from __future__ import annotations


LINEAR_SCORE_WEIGHT = 1.0
ATTENTION_SCORE_WEIGHT = 5.0
PERCENTAGE_POINTS = 100.0


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
    # The task document defines the final score as the sum of each case's
    # MSE-improvement percentage, so ratios must be converted to percentage
    # points. With 50 Linear and 250 Attention cases, this is exactly the
    # direct sum over all 300 case percentages.
    return PERCENTAGE_POINTS * (len(linear_scores) + len(attention_scores)) * weighted_mean
