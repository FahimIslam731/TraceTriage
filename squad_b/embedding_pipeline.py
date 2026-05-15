from __future__ import annotations

"""Embedding-based classification pipeline for Squad B.

Embeds flattened trace text via OpenAI API (text-embedding-3-small),
caches embeddings to disk, and trains SVM + LogReg classifiers.

Reproducibility: deterministic splits via RANDOM_SEED, cached embeddings.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_loader import RANDOM_SEED, build_dataset, get_index_splits
from .evaluator import confusion_matrix_str, evaluate, print_report, save_results

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


def save_cache(embeddings: np.ndarray, trace_ids: list[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(EMBED_CACHE), embeddings)
    EMBED_IDS_CACHE.write_text(json.dumps(trace_ids), encoding="utf-8")
    print(f"  Cached {embeddings.shape} embeddings to {EMBED_CACHE}")


def load_cache() -> tuple[np.ndarray, list[str]] | None:
    if EMBED_CACHE.exists() and EMBED_IDS_CACHE.exists():
        embeddings = np.load(str(EMBED_CACHE))
        ids = json.loads(EMBED_IDS_CACHE.read_text(encoding="utf-8"))
        print(f"  Loaded cached embeddings: {embeddings.shape}")
        return embeddings, ids
    return None


def run_embedding_pipeline(test_size: float = 0.2) -> dict:
    """Run embedding-based classification (SVM + LogReg)."""
    print("Loading dataset...")
    ds = build_dataset(use_sqlite_features=False)
    texts, labels, trace_ids = ds["texts"], ds["labels"], ds["trace_ids"]

    # Try cache first
    cached = load_cache()
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
        save_cache(embeddings, trace_ids)

    # Split
    train_idx, test_idx = get_index_splits(len(labels), labels, test_size, RANDOM_SEED)
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # SVM
    print("\nTraining SVM (RBF)...")
    svm = SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_res = evaluate(y_test, y_pred_svm)
    print_report(svm_res, "Embedding + SVM (RBF)")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_svm))
    save_results(svm_res, "embedding_svm", extra={"embed_model": EMBED_MODEL,
                 "random_seed": RANDOM_SEED})

    # Logistic Regression
    print("\nTraining Logistic Regression on embeddings...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_SEED)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_res = evaluate(y_test, y_pred_lr)
    print_report(lr_res, "Embedding + Logistic Regression")
    print("Confusion Matrix:\n" + confusion_matrix_str(y_test, y_pred_lr))
    save_results(lr_res, "embedding_logreg", extra={"embed_model": EMBED_MODEL,
                 "random_seed": RANDOM_SEED})

    return svm_res
