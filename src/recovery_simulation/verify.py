"""Answer verification helpers for each benchmark domain.

These are intentionally simple — they trade recall for precision so false
positives (claiming success when the answer is wrong) are rare.
"""
import re
import string


def verify_answer(domain: str, gold_answer: str, recovered_answer: str) -> bool:
    """Return True if recovered_answer is considered correct for the given domain."""
    if not recovered_answer or not gold_answer:
        return False
    domain_upper = domain.upper()
    if domain_upper == "GSM8K":
        return _verify_gsm8k(gold_answer, recovered_answer)
    if domain_upper == "MBPP":
        return _verify_mbpp(gold_answer, recovered_answer)
    # SealQA, MedBrowseComp, BrowseComp all use fuzzy text match
    return _verify_text(gold_answer, recovered_answer)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_numbers(text: str) -> list[str]:
    # Handles integers, decimals, and comma-separated thousands (e.g. 1,000)
    return re.findall(r"-?\d[\d,]*\.?\d*", text)


def _verify_gsm8k(gold: str, recovered: str) -> bool:
    """Extract the final numeric value from both answers and compare."""
    gold_nums = _extract_numbers(gold.replace(",", ""))
    rec_nums = _extract_numbers(recovered.replace(",", ""))
    if not gold_nums or not rec_nums:
        return False
    # Compare the last number in each (final answer position)
    try:
        return float(gold_nums[-1]) == float(rec_nums[-1])
    except ValueError:
        return False


def _verify_mbpp(gold: str, recovered: str) -> bool:
    """Check whether the recovered code contains the expected function signature or output.

    Full code execution is expensive and environment-dependent; this check
    catches obvious misses without running arbitrary code.
    """
    gold_norm = _normalize(gold)
    rec_norm = _normalize(recovered)
    # If the gold answer is a short expected output value, look for it
    if len(gold_norm) <= 30:
        return gold_norm in rec_norm
    # Otherwise overlap on key tokens
    gold_tokens = set(gold_norm.split())
    rec_tokens = set(rec_norm.split())
    overlap = gold_tokens & rec_tokens
    if not gold_tokens:
        return False
    return len(overlap) / len(gold_tokens) >= 0.7


def _verify_text(gold: str, recovered: str) -> bool:
    """Fuzzy match: normalized gold must appear as a substring, or token overlap >= 0.8."""
    gold_norm = _normalize(gold)
    rec_norm = _normalize(recovered)
    if gold_norm in rec_norm:
        return True
    gold_tokens = set(gold_norm.split())
    rec_tokens = set(rec_norm.split())
    if not gold_tokens:
        return False
    overlap = gold_tokens & rec_tokens
    return len(overlap) / len(gold_tokens) >= 0.8
