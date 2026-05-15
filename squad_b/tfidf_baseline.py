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

from .data_loader import RANDOM_SEED, build_dataset, get_index_splits
from .evaluator import confusion_matrix_str, evaluate, print_report, save_results


def run_tfidf_baseline(use_structured: bool = True, test_size: float = 0.2) -> dict:
    """Run TF-IDF baseline: LogReg, RandomForest, and majority baseline."""
    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=use_structured)
    texts, labels, structured = ds["texts"], ds["labels"], ds["structured"]
    feature_names = ds["feature_names"]

    train_idx, test_idx = get_index_splits(len(labels), labels, test_size, RANDOM_SEED)
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

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

    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_SEED, solver="lbfgs",
                            multi_class="multinomial")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_res = evaluate(y_test, y_pred_lr)
    print_report(lr_res, "TF-IDF + Logistic Regression")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_lr))
    save_results(lr_res, "tfidf_logreg", extra={"use_structured": use_structured,
                 "random_seed": RANDOM_SEED, "test_size": test_size})

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_res = evaluate(y_test, y_pred_rf)
    print_report(rf_res, "TF-IDF + Random Forest")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_rf))
    save_results(rf_res, "tfidf_rf", extra={"random_seed": RANDOM_SEED})

    # Majority baseline
    majority = Counter(y_train).most_common(1)[0][0]
    y_pred_maj = np.full_like(y_test, majority)
    maj_res = evaluate(y_test, y_pred_maj)
    print_report(maj_res, f"Majority Baseline (always '{majority}')")
    save_results(maj_res, "majority_baseline")

    return lr_res
