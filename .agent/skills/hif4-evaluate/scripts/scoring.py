from __future__ import annotations


LINEAR_SCORE_WEIGHT = 1.0
ATTENTION_SCORE_WEIGHT = 5.0
PERCENTAGE_POINTS = 100.0
SCORE_SCALE_CASES = 300


def weighted_total_score(
    linear_scores: list[float], attention_scores: list[float],
) -> float:
    """Match the official 1:5 case ratio while retaining the local score scale."""
    if not linear_scores or not attention_scores:
        raise ValueError("both Linear and Attention scores are required")
    linear_mean = sum(linear_scores) / len(linear_scores)
    attention_mean = sum(attention_scores) / len(attention_scores)
    weighted_mean = (
        LINEAR_SCORE_WEIGHT * linear_mean
        + ATTENTION_SCORE_WEIGHT * attention_mean
    ) / (LINEAR_SCORE_WEIGHT + ATTENTION_SCORE_WEIGHT)
    # The formal dataset contains 50 Linear and 250 Attention cases. Weighting
    # the two category means 1:5 and multiplying by 300 is algebraically the
    # same as summing all 300 per-case improvement ratios.
    return PERCENTAGE_POINTS * SCORE_SCALE_CASES * weighted_mean
