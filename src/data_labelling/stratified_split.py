#!/usr/bin/env python3
"""
stratified_split.py
====================
Atomically stratifies consolidated_labels.csv into 80/10/10 train/dev/test
splits with special handling for rare labels, then generates a SHA-256
hash manifest to lock all output files.

Usage:
    python stratified_split.py [--input PATH] [--output-dir DIR] [--seed INT]

Design decisions
----------------
- Stratifies on `human_majority` (the action label column).
- Rare labels (fewer than MIN_PER_SPLIT * 3 = 9 examples) get at least one
  example in each split; remaining examples go to train.
- ESCALATE (n=1) is too small to split 3 ways — placed entirely in train
  with a warning.
- All file writes happen together (atomic-style: write to temp then rename)
  so a partial failure never leaves inconsistent outputs.
- Hash manifest uses SHA-256 over the raw CSV bytes, recorded in
  manifest.json alongside the splits.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
DEV_RATIO   = 0.10
TEST_RATIO  = 0.10
MIN_PER_SPLIT = 1          # minimum examples per split for rare labels
LABEL_COL   = "human_majority"
DEFAULT_SEED = 42

RARE_THRESHOLD = 10        # labels with fewer examples get special handling


# ── Helpers ───────────────────────────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_csv(df: pd.DataFrame, dest: Path, tmp_dir: Path) -> None:
    """Write df to a temp file, then rename into place (atomic on POSIX)."""
    tmp = tmp_dir / (dest.name + ".tmp")
    df.to_csv(tmp, index=False)
    shutil.move(str(tmp), str(dest))


def split_label_group(
    group: pd.DataFrame,
    label: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single-label group into (train, dev, test).

    Rare-label strategy
    -------------------
    If n < 3  → all go to train, emit warning.
    If 3 <= n < RARE_THRESHOLD → guarantee 1 in dev and 1 in test,
                                  rest to train.
    Otherwise → standard proportional split via sklearn StratifiedShuffleSplit
                (or manual floor/ceil arithmetic to avoid sklearn dependency).
    """
    n = len(group)
    shuffled = group.sample(frac=1, random_state=seed).reset_index(drop=True)

    if n < 3:
        print(f"  ⚠  '{label}' has only {n} example(s) — assigning all to TRAIN.")
        return shuffled, pd.DataFrame(columns=group.columns), pd.DataFrame(columns=group.columns)

    if n < RARE_THRESHOLD:
        print(f"  ⚠  '{label}' is rare (n={n}) — guaranteeing 1 in DEV and 1 in TEST.")
        test  = shuffled.iloc[[n - 1]]
        dev   = shuffled.iloc[[n - 2]]
        train = shuffled.iloc[:n - 2]
        return train, dev, test

    # Standard split
    n_test  = max(MIN_PER_SPLIT, round(n * TEST_RATIO))
    n_dev   = max(MIN_PER_SPLIT, round(n * DEV_RATIO))
    n_train = n - n_dev - n_test

    test  = shuffled.iloc[:n_test]
    dev   = shuffled.iloc[n_test : n_test + n_dev]
    train = shuffled.iloc[n_test + n_dev :]

    return train, dev, test


# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",      default="squad_a/audit_results/all_1212_labels.csv")
    parser.add_argument("--output-dir", default="squad_a")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    seed       = args.seed

    print(f"📂  Input : {input_path}")
    print(f"📁  Output: {output_dir}")
    print(f"🌱  Seed  : {seed}\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    if LABEL_COL not in df.columns:
        sys.exit(f"ERROR: column '{LABEL_COL}' not found in {input_path}.\n"
                 f"Available columns: {df.columns.tolist()}")

    print(f"Loaded {len(df)} rows, {df[LABEL_COL].nunique()} unique labels.\n")
    print("Label distribution:")
    for label, cnt in df[LABEL_COL].value_counts().items():
        print(f"  {label:<20} {cnt:>4}")
    print()

    # ── Stratified split per label ────────────────────────────────────────────
    trains, devs, tests = [], [], []

    for label, group in df.groupby(LABEL_COL, sort=False):
        tr, dv, te = split_label_group(group, label, seed)
        trains.append(tr)
        devs.append(dv)
        tests.append(te)

    train_df = pd.concat(trains).sample(frac=1, random_state=seed).reset_index(drop=True)
    dev_df   = pd.concat(devs).sample(frac=1,   random_state=seed).reset_index(drop=True)
    test_df  = pd.concat(tests).sample(frac=1,  random_state=seed).reset_index(drop=True)

    total = len(train_df) + len(dev_df) + len(test_df)
    assert total == len(df), f"Row count mismatch: {total} != {len(df)}"

    # ── Print split summary ───────────────────────────────────────────────────
    print("Split summary:")
    print(f"  {'Set':<8} {'N':>5}  {'%':>6}  per-label counts")
    for name, split in [("TRAIN", train_df), ("DEV", dev_df), ("TEST", test_df)]:
        per = split[LABEL_COL].value_counts().to_dict()
        pct = 100 * len(split) / len(df)
        detail = "  ".join(f"{k}={v}" for k, v in sorted(per.items()))
        print(f"  {name:<8} {len(split):>5}  {pct:>5.1f}%  {detail}")
    print()

    # ── Verify no label is missing from all three sets ────────────────────────
    all_labels = set(df[LABEL_COL].unique())
    represented = (
        set(train_df[LABEL_COL].unique()) |
        set(dev_df[LABEL_COL].unique())   |
        set(test_df[LABEL_COL].unique())
    )
    missing = all_labels - represented
    if missing:
        print(f"  ⚠  Labels present in input but missing from all outputs: {missing}")

    # ── Atomic write to output dir ────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp())

    out_files = {
        "train.csv": train_df,
        "dev.csv":   dev_df,
        "test.csv":  test_df,
    }

    written_paths = {}
    try:
        for fname, split_df in out_files.items():
            dest = output_dir / fname
            atomic_write_csv(split_df, dest, tmp_dir)
            written_paths[fname] = dest
            print(f"  ✓  Wrote {dest}  ({len(split_df)} rows)")
    except Exception as e:
        print(f"\nERROR during write: {e}")
        print("Cleaning up partial outputs…")
        for p in written_paths.values():
            p.unlink(missing_ok=True)
        sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Hash manifest ─────────────────────────────────────────────────────────
    manifest = {
        "schema_version": "1.0",
        "seed": seed,
        "source_file": str(input_path),
        "source_sha256": sha256_file(input_path),
        "total_rows": len(df),
        "splits": {},
    }

    for fname, dest in written_paths.items():
        split_name = fname.replace(".csv", "")
        split_df   = out_files[fname]
        manifest["splits"][split_name] = {
            "file":    fname,
            "sha256":  sha256_file(dest),
            "rows":    len(split_df),
            "pct":     round(100 * len(split_df) / len(df), 2),
            "label_counts": split_df[LABEL_COL].value_counts().to_dict(),
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✓  Manifest written → {manifest_path}")

    # ── Verification helper reminder ──────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────")
    print("To verify integrity later, run:")
    print(f"  python stratified_split.py --verify {manifest_path}")
    print("──────────────────────────────────────────────────────────────\n")

    return manifest


def verify(manifest_path: str):
    """Re-hash all files listed in the manifest and report any mismatches."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(manifest_path).parent
    ok = True

    print(f"Verifying splits in {output_dir} …\n")
    for split_name, info in manifest["splits"].items():
        fpath = output_dir / info["file"]
        if not fpath.exists():
            print(f"  ✗  MISSING: {fpath}")
            ok = False
            continue
        actual = sha256_file(fpath)
        if actual == info["sha256"]:
            print(f"  ✓  {info['file']}  ({info['rows']} rows) — OK")
        else:
            print(f"  ✗  {info['file']}  HASH MISMATCH")
            print(f"       expected: {info['sha256']}")
            print(f"       got:      {actual}")
            ok = False

    if ok:
        print("\n✅  All files intact.")
    else:
        print("\n❌  Integrity check FAILED — do not use these splits.")
        sys.exit(2)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--verify":
        if len(sys.argv) < 3:
            sys.exit("Usage: python stratified_split.py --verify <manifest.json>")
        verify(sys.argv[2])
    else:
        main()
