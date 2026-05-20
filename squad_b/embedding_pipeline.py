from __future__ import annotations

"""Embedding-based classification pipeline for Squad B.

Embeds flattened trace text via OpenAI API (text-embedding-3-small),
caches embeddings to disk, and trains SVM, LogReg, and optional XGBoost classifiers.

Reproducibility: deterministic splits via RANDOM_SEED, cached embeddings.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import RANDOM_SEED, build_dataset, get_train_test_indices
from .evaluator import (
    confusion_matrix_str,
    evaluate,
    evaluate_by_group,
    print_group_summary,
    print_report,
    save_results,
)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
EMBED_CACHE = CACHE_DIR / "embeddings.npy"
EMBED_IDS_CACHE = CACHE_DIR / "embedding_ids.json"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "openai/text-embedding-3-small"
BATCH_SIZE = 64  # OpenAI embedding batch limit


def _get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL,
                  default_headers={"HTTP-Referer": "http://localhost",
                                   "X-Title": "TraceTriage Squad B"})


def embed_texts(texts: list[str], client) -> np.ndarray:
    """Embed a list of texts using the OpenAI API. Returns (N, D) array."""
    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # Truncate to avoid token limits (8191 tokens ≈ 30k chars)
        batch = [t[:25_000] for t in batch]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for item in resp.data:
            all_embeddings.append(item.embedding)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return np.array(all_embeddings, dtype=np.float32)


def _cache_paths(input_variant: str) -> tuple[Path, Path]:
    if input_variant == "full_trace":
        return EMBED_CACHE, EMBED_IDS_CACHE
    return (
        CACHE_DIR / f"embeddings_{input_variant}.npy",
        CACHE_DIR / f"embedding_ids_{input_variant}.json",
    )


def save_cache(embeddings: np.ndarray, trace_ids: list[str], input_variant: str = "full_trace") -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embed_cache, ids_cache = _cache_paths(input_variant)
    np.save(str(embed_cache), embeddings)
    ids_cache.write_text(json.dumps(trace_ids), encoding="utf-8")
    print(f"  Cached {embeddings.shape} embeddings to {embed_cache}")


def load_cache(input_variant: str = "full_trace") -> tuple[np.ndarray, list[str]] | None:
    embed_cache, ids_cache = _cache_paths(input_variant)
    if embed_cache.exists() and ids_cache.exists():
        embeddings = np.load(str(embed_cache))
        ids = json.loads(ids_cache.read_text(encoding="utf-8"))
        print(f"  Loaded cached embeddings: {embeddings.shape}")
        return embeddings, ids
    return None


def run_embedding_pipeline(test_size: float = 0.2, input_variant: str = "full_trace") -> dict:
    """Run embedding-based classification (SVM, LogReg, optional XGBoost)."""
    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=False, input_variant=input_variant)
    texts, labels, trace_ids = ds["texts"], ds["labels"], ds["trace_ids"]
    domains = np.array([t.get("domain", "") for t in ds["traces"]])

    # Try cache first
    cached = load_cache(input_variant)
    if cached is not None:
        embeddings, cached_ids = cached
        if cached_ids == trace_ids and embeddings.shape[0] == len(trace_ids):
            print("  Cache hit — using cached embeddings.")
        else:
            print("  Cache mismatch — re-embedding.")
            cached = None

    if cached is None:
        client = _get_client()
        if client is None:
            print("ERROR: OPENROUTER_API_KEY not set and no cached embeddings.")
            print("Set the env var and re-run, or provide cached embeddings.")
            sys.exit(1)
        print(f"Embedding {len(texts)} traces with {EMBED_MODEL}...")
        embeddings = embed_texts(texts, client)
        save_cache(embeddings, trace_ids, input_variant)

    # Split
    train_idx, test_idx, dev_idx = get_train_test_indices(
        ds, labels, test_size, RANDOM_SEED
    )
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    test_domains = domains[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if dev_idx is not None:
        print(f"  Train: {X_train.shape}, Dev: {len(dev_idx)}, Test: {X_test.shape}")
    else:
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # SVM
    print("\nTraining SVM (RBF)...")
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_res = evaluate(y_test, y_pred_svm)
    svm_by_domain = evaluate_by_group(y_test, y_pred_svm, test_domains)
    print_report(svm_res, "Embedding + SVM (RBF)")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_svm))
    print_group_summary(svm_by_domain, "Per-domain summary: Embedding + SVM")
    save_results(svm_res, "embedding_svm", extra={"embed_model": EMBED_MODEL,
                 "random_seed": RANDOM_SEED,
                 "split_source": "squad_a_frozen" if dev_idx is not None else "random",
                 "input_variant": input_variant,
                 "per_domain": svm_by_domain})

    # Logistic Regression
    print("\nTraining Logistic Regression on embeddings...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_res = evaluate(y_test, y_pred_lr)
    lr_by_domain = evaluate_by_group(y_test, y_pred_lr, test_domains)
    print_report(lr_res, "Embedding + Logistic Regression")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_lr))
    print_group_summary(lr_by_domain, "Per-domain summary: Embedding + Logistic Regression")
    save_results(lr_res, "embedding_logreg", extra={"embed_model": EMBED_MODEL,
                 "random_seed": RANDOM_SEED,
                 "split_source": "squad_a_frozen" if dev_idx is not None else "random",
                 "input_variant": input_variant,
                 "per_domain": lr_by_domain})

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("\nSkipping Embedding + XGBoost: install xgboost to enable it.")
    else:
        print("\nTraining XGBoost on embeddings...")
        label_encoder = LabelEncoder()
        y_train_enc = label_encoder.fit_transform(y_train)

        xgb = XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        xgb.fit(X_train, y_train_enc)
        y_pred_xgb = label_encoder.inverse_transform(xgb.predict(X_test))
        xgb_res = evaluate(y_test, y_pred_xgb)
        xgb_by_domain = evaluate_by_group(y_test, y_pred_xgb, test_domains)
        print_report(xgb_res, "Embedding + XGBoost")
        print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_xgb))
        print_group_summary(xgb_by_domain, "Per-domain summary: Embedding + XGBoost")
        save_results(xgb_res, "embedding_xgboost", extra={
            "embed_model": EMBED_MODEL,
            "random_seed": RANDOM_SEED,
            "split_source": "squad_a_frozen" if dev_idx is not None else "random",
            "input_variant": input_variant,
            "per_domain": xgb_by_domain,
        })

    return svm_res
