#!/usr/bin/env python3
"""Reviewer-facing wrapper for the Squad A human-vs-LLM audit script.

The canonical implementation lives at ``squad_a/human_vs_llm_audit.py``.
Run this file from the repository root to reproduce the audit metrics:

    python src/data_labelling/human_vs_llm_audit.py
"""

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = PROJECT_ROOT / "squad_a" / "human_vs_llm_audit.py"


if __name__ == "__main__":
    runpy.run_path(str(AUDIT_SCRIPT), run_name="__main__")
