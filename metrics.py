"""The two leaderboard metrics, plus clipping to the valid score range."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error (lower is better)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true, y_pred) -> float:
    """Pearson correlation (higher is better). NaN if either vector is constant."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def clip_scores(pred, lo: float = 0.0, hi: float = 5.0):
    """Predictions are clipped to the valid label range [0, 5]."""
    return np.clip(np.asarray(pred, float), lo, hi)
