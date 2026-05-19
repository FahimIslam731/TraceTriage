from __future__ import annotations

"""Leave-one-domain-out evaluation for Squad B Experiment 3."""

import json
from collections import Counter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data_loader import RANDOM_SEED, build_dataset
from .evaluator import RESULTS_DIR, confusion_matrix_str, evaluate, print_report, save_results


def _build_features(
    train_texts: list[str],
    test_texts: list[str],
    structured: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    use_structured: bool,
) -> tuple[Any, Any]:
    """Fit text/structured featurizers on train only and transform held-out domain."""
    tfidf = TfidfVectorizer(
        max_features=10_000,
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_tr_tf = tfidf.fit_transform(train_texts)
    X_te_tf = tfidf.transform(test_texts)

    if not use_structured:
        return X_tr_tf, X_te_tf

    scaler = StandardScaler()
    s_tr = scaler.fit_transform(structured[train_idx])
    s_te = scaler.transform(structured[test_idx])
    return hstack([X_tr_tf, csr_matrix(s_tr)]), hstack([X_te_tf, csr_matrix(s_te)])


def _append_summary_row(
    rows: list[dict],
    domain: str,
    method: str,
    result: dict,
    n_train: int,
    n_test: int,
) -> None:
    rows.append({
        "heldout_domain": domain,
        "method": method,
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "n_train": n_train,
        "n_test": n_test,
    })


def _summarize_per_action_transfer(all_results: dict[str, dict]) -> dict:
    """Average per-action F1 across held-out domains for each method."""
    action_scores: dict[str, dict[str, list[float]]] = {}
    for domain_results in all_results.values():
        for method, result in domain_results.items():
            action_scores.setdefault(method, {})
            for action, metrics in result.get("per_class", {}).items():
                action_scores[method].setdefault(action, []).append(metrics["f1"])

    return {
        method: {
            action: {
                "mean_f1": round(float(np.mean(scores)), 4),
                "num_domains": len(scores),
            }
            for action, scores in sorted(actions.items())
        }
        for method, actions in sorted(action_scores.items())
    }


def _load_in_domain_reference() -> dict:
    """Load in-domain Experiment 2 metrics if TF-IDF results have been run."""
    references = {}
    for method, filename in {
        "tfidf_logreg": "tfidf_logreg_results.json",
        "tfidf_rf": "tfidf_rf_results.json",
    }.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        references[method] = payload.get("metrics", {})
    return references


def _summarize_transfer_gap(summary_rows: list[dict], in_domain_reference: dict) -> dict:
    """Compare held-out-domain macro F1 to in-domain macro F1 when available."""
    gaps = {}
    for method, reference in in_domain_reference.items():
        in_domain_f1 = reference.get("macro_f1")
        if in_domain_f1 is None:
            continue
        heldout = [row["macro_f1"] for row in summary_rows if row["method"] == method]
        if not heldout:
            continue
        mean_heldout = float(np.mean(heldout))
        gaps[method] = {
            "in_domain_macro_f1": in_domain_f1,
            "mean_heldout_macro_f1": round(mean_heldout, 4),
            "transfer_gap": round(float(in_domain_f1 - mean_heldout), 4),
        }
    return gaps


def run_cross_domain_evaluation(
    use_structured: bool = True,
    min_domain_samples: int = 10,
) -> dict:
    """Run leave-one-domain-out evaluation for Experiment 3.

    For each domain with at least ``min_domain_samples`` examples, train on all
    other domains and evaluate on the held-out domain. The non-oracle majority
    baseline predicts the global training majority. The oracle domain baseline
    predicts the held-out domain's modal action and is included only as a
    shortcut-detection reference.
    """
    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=use_structured)
    texts = ds["texts"]
    labels = ds["labels"]
    structured = ds["structured"]
    domains = np.array([t.get("domain", "") for t in ds["traces"]])

    domain_counts = Counter(domains)
    heldout_domains = [
        d for d, count in sorted(domain_counts.items())
        if count >= min_domain_samples
    ]
    skipped = {
        d: count for d, count in sorted(domain_counts.items())
        if count < min_domain_samples
    }

    if skipped:
        print(f"Skipping small domains (<{min_domain_samples} examples): {skipped}")

    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for domain in heldout_domains:
        print(f"\n{'=' * 72}")
        print(f"  Held-out domain: {domain}")
        print(f"{'=' * 72}")

        test_idx = np.where(domains == domain)[0]
        train_idx = np.where(domains != domain)[0]
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
        print(f"  Test label distribution: {dict(Counter(y_test))}")

        X_train, X_test = _build_features(
            train_texts,
            test_texts,
            structured,
            train_idx,
            test_idx,
            use_structured,
        )

        domain_results = {}

        majority = Counter(y_train).most_common(1)[0][0]
        y_pred_majority = np.full_like(y_test, majority)
        majority_res = evaluate(y_test, y_pred_majority)
        print_report(majority_res, f"Cross-Domain Majority ({domain})")
        domain_results["majority"] = majority_res
        _append_summary_row(
            summary_rows, domain, "majority", majority_res, len(train_idx), len(test_idx)
        )

        oracle_domain_mode = Counter(y_test).most_common(1)[0][0]
        y_pred_oracle_domain = np.full_like(y_test, oracle_domain_mode)
        oracle_res = evaluate(y_test, y_pred_oracle_domain)
        print_report(oracle_res, f"Oracle Domain-Mode Baseline ({domain})")
        domain_results["oracle_domain_mode"] = oracle_res
        _append_summary_row(
            summary_rows,
            domain,
            "oracle_domain_mode",
            oracle_res,
            len(train_idx),
            len(test_idx),
        )

        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        lr_res = evaluate(y_test, y_pred_lr)
        print_report(lr_res, f"Cross-Domain TF-IDF + LogReg ({domain})")
        print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_lr))
        domain_results["tfidf_logreg"] = lr_res
        _append_summary_row(
            summary_rows, domain, "tfidf_logreg", lr_res, len(train_idx), len(test_idx)
        )

        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        rf_res = evaluate(y_test, y_pred_rf)
        print_report(rf_res, f"Cross-Domain TF-IDF + Random Forest ({domain})")
        print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_rf))
        domain_results["tfidf_rf"] = rf_res
        _append_summary_row(
            summary_rows, domain, "tfidf_rf", rf_res, len(train_idx), len(test_idx)
        )

        all_results[domain] = domain_results

    per_action_transfer = _summarize_per_action_transfer(all_results)
    in_domain_reference = _load_in_domain_reference()
    transfer_gap = _summarize_transfer_gap(summary_rows, in_domain_reference)

    output = {
        "heldout_domains": heldout_domains,
        "skipped_domains": skipped,
        "summary": summary_rows,
        "per_action_transfer": per_action_transfer,
        "in_domain_reference": in_domain_reference,
        "transfer_gap": transfer_gap,
        "by_domain": all_results,
    }
    save_results(output, "cross_domain", extra={
        "use_structured": use_structured,
        "min_domain_samples": min_domain_samples,
        "random_seed": RANDOM_SEED,
    })

    print("\nCross-domain summary:")
    for row in summary_rows:
        print(
            f"  {row['heldout_domain']:<14} {row['method']:<20} "
            f"macro_f1={row['macro_f1']:.4f} accuracy={row['accuracy']:.4f}"
        )

    print("\nPer-action transfer summary (mean F1 across held-out domains):")
    for method, actions in per_action_transfer.items():
        action_text = ", ".join(
            f"{action}={stats['mean_f1']:.4f}" for action, stats in actions.items()
        )
        print(f"  {method:<20} {action_text}")

    if transfer_gap:
        print("\nTransfer gap vs in-domain TF-IDF results:")
        for method, gap in transfer_gap.items():
            print(
                f"  {method:<20} in_domain={gap['in_domain_macro_f1']:.4f} "
                f"heldout_mean={gap['mean_heldout_macro_f1']:.4f} "
                f"gap={gap['transfer_gap']:.4f}"
            )

    return output
