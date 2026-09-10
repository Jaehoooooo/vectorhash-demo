"""
metrics.py
Simple retrieval-quality metrics used across experiments.
"""
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def exact_match_rate(recovered: np.ndarray, target: np.ndarray) -> float:
    """Fraction of matching signs -- appropriate for bipolar patterns."""
    return float(np.mean(np.sign(recovered) == np.sign(target)))


def capacity_curve(model_fit_and_recall_fn, item_counts: list, **kwargs) -> list:
    """Generic helper: for each n in item_counts, call
    model_fit_and_recall_fn(n, **kwargs) -> average recall score."""
    return [model_fit_and_recall_fn(n, **kwargs) for n in item_counts]


def binary_mutual_information(bit_error_rate: float) -> float:
    """Mutual information (bits) retained per stored bit, given a binary
    symmetric channel with the observed bit_error_rate. Used by the paper
    (Figs 3/7) to compare recall quality on a common scale that doesn't
    saturate as fast as raw accuracy. MI = 1 - H(bit_error_rate)."""
    p = np.clip(bit_error_rate, 1e-12, 1 - 1e-12)
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return float(max(1.0 - H, 0.0))


def is_novel(residual_error: float, familiarity_threshold: float) -> bool:
    """Fig 3 novelty-detection readout: an item is judged 'novel' (never
    stored) if the scaffold's hippocampal-state reconstruction error after
    cleanup exceeds familiarity_threshold, and 'familiar' otherwise."""
    return residual_error > familiarity_threshold
