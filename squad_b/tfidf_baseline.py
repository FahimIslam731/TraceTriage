from __future__ import annotations

"""TF-IDF + classifier baselines for Squad B.

Trains Logistic Regression and Random Forest on TF-IDF text features
optionally combined with structured numerical features.

Reproducibility: all models use random_state = RANDOM_SEED.
"""
import numpy as np
from collections import Counter
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loader import RANDOM_SEED, build_dataset, get_train_test_indices
from .evaluator import (
    confusion_matrix_str,
    confusion_matrices_by_group,
    evaluate,
    evaluate_by_group,
    print_group_summary,
    print_report,
    save_results,
)


def run_tfidf_baseline(
    use_structured: bool = True,
    test_size: float = 0.2,
    input_variant: str = "full_trace",
    result_prefix: str = "",
) -> dict:
    """Run TF-IDF baseline, domain-only baseline, and majority baseline."""
    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=use_structured, input_variant=input_variant)
    texts, labels, structured = ds["texts"], ds["labels"], ds["structured"]
    domains = np.array([t.get("domain", "") for t in ds["traces"]])

    train_idx, test_idx, dev_idx = get_train_test_indices(
        ds, labels, test_size, RANDOM_SEED
    )
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    train_domains, test_domains = domains[train_idx], domains[test_idx]
    # TF-IDF
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=10_000, sublinear_tf=True,
                            ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_tr_tf = tfidf.fit_transform(train_texts)
    X_te_tf = tfidf.transform(test_texts)

    if use_structured:
        scaler = StandardScaler()
        s_tr = scaler.fit_transform(structured[train_idx])
        s_te = scaler.transform(structured[test_idx])
        X_train = hstack([X_tr_tf, csr_matrix(s_tr)])
        X_test = hstack([X_te_tf, csr_matrix(s_te)])
        print(f"  TF-IDF: {X_tr_tf.shape[1]}, Structured: {s_tr.shape[1]}, Total: {X_train.shape[1]}")
    else:
        X_train, X_test = X_tr_tf, X_te_tf
        print(f"  TF-IDF features: {X_train.shape[1]}")

    if dev_idx is not None:
        print(f"  Train: {X_train.shape[0]}, Dev: {len(dev_idx)}, Test: {X_test.shape[0]}")
    else:
        print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_SEED, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_res = evaluate(y_test, y_pred_lr)
    lr_by_domain = evaluate_by_group(y_test, y_pred_lr, test_domains)
    lr_cm_by_domain = confusion_matrices_by_group(
        y_test, y_pred_lr, test_domains, sorted(set(y_test) | set(y_pred_lr))
    )
    print_report(lr_res, "TF-IDF + Logistic Regression")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_lr))
    print_group_summary(lr_by_domain, "Per-domain summary: TF-IDF + Logistic Regression")
    name_prefix = f"{result_prefix}_" if result_prefix else ""

    save_results(lr_res, f"{name_prefix}tfidf_logreg", extra={"use_structured": use_structured,
                 "random_seed": RANDOM_SEED, "test_size": test_size,
                 "split_source": "squad_a_frozen" if dev_idx is not None else "random",
                 "input_variant": input_variant,
                 "per_domain": lr_by_domain,
                 "per_domain_confusion_matrices": lr_cm_by_domain})

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_res = evaluate(y_test, y_pred_rf)
    rf_by_domain = evaluate_by_group(y_test, y_pred_rf, test_domains)
    rf_cm_by_domain = confusion_matrices_by_group(
        y_test, y_pred_rf, test_domains, sorted(set(y_test) | set(y_pred_rf))
    )
    print_report(rf_res, "TF-IDF + Random Forest")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_rf))
    print_group_summary(rf_by_domain, "Per-domain summary: TF-IDF + Random Forest")
    save_results(rf_res, f"{name_prefix}tfidf_rf", extra={
        "random_seed": RANDOM_SEED,
        "split_source": "squad_a_frozen" if dev_idx is not None else "random",
        "input_variant": input_variant,
        "per_domain": rf_by_domain,
        "per_domain_confusion_matrices": rf_cm_by_domain,
    })

    # Majority baseline
    majority = Counter(y_train).most_common(1)[0][0]
    y_pred_maj = np.full_like(y_test, majority)
    maj_res = evaluate(y_test, y_pred_maj)
    maj_by_domain = evaluate_by_group(y_test, y_pred_maj, test_domains)
    print_report(maj_res, f"Majority Baseline (always '{majority}')")
    save_results(maj_res, f"{name_prefix}majority_baseline", extra={
        "split_source": "squad_a_frozen" if dev_idx is not None else "random",
        "input_variant": input_variant,
        "per_domain": maj_by_domain,
    })

    # Domain-only baseline: predict the modal training action for each domain.
    domain_modes = {}
    for domain in sorted(set(train_domains)):
        domain_labels = y_train[train_domains == domain]
        domain_modes[domain] = Counter(domain_labels).most_common(1)[0][0]

    y_pred_domain = np.array([
        domain_modes.get(domain, majority) for domain in test_domains
    ])
    domain_res = evaluate(y_test, y_pred_domain)
    domain_by_domain = evaluate_by_group(y_test, y_pred_domain, test_domains)
    domain_cm_by_domain = confusion_matrices_by_group(
        y_test, y_pred_domain, test_domains, sorted(set(y_test) | set(y_pred_domain))
    )
    print_report(domain_res, "Domain-Only Baseline")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_domain))
    print_group_summary(domain_by_domain, "Per-domain summary: Domain-Only Baseline")
    save_results(domain_res, f"{name_prefix}domain_only_baseline", extra={
        "domain_modes": domain_modes,
        "fallback_action": majority,
        "random_seed": RANDOM_SEED,
        "test_size": test_size,
        "split_source": "squad_a_frozen" if dev_idx is not None else "random",
        "input_variant": input_variant,
        "per_domain": domain_by_domain,
        "per_domain_confusion_matrices": domain_cm_by_domain,
    })

    print(
        "Macro F1 gap vs domain-only: "
        f"LogReg {lr_res['macro_f1'] - domain_res['macro_f1']:+.4f}, "
        f"RandomForest {rf_res['macro_f1'] - domain_res['macro_f1']:+.4f}"
    )

    return {
        "tfidf_logreg": lr_res,
        "tfidf_rf": rf_res,
        "majority": maj_res,
        "domain_only": domain_res,
    }
