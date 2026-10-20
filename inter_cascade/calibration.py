"""Deferral-threshold calibration (Jung et al., 2024, Algorithm 1).

Given calibration confidences and correctness labels, find the smallest
threshold lambda such that, with probability at least 1 - delta, the error
rate among answers with confidence >= lambda stays below alpha. The search is
fixed-sequence testing from high to low thresholds with a Clopper-Pearson
upper confidence bound at each step.

This is an exact port of ``threshold()`` from the original research code (the
function the paper's lambda values came from), including two idiosyncrasies:
grid points are skipped while the slightly-wider window still contains no
errors, and windows where *every* answer above the threshold is wrong are
passed through untested (the original produced a NaN bound there, which never
compared as a failure). The returned value is the first grid point at which
the Clopper-Pearson bound exceeds alpha, i.e. one step below the last
certified threshold.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import beta


def calibrate_threshold(
    confidences: Sequence[float],
    labels: Sequence[int],
    alpha: float,
    delta: float,
    grid_size: int = 1000,
) -> float:
    """Largest-coverage threshold with a (1 - delta) guarantee of risk <= alpha.

    Raises ValueError when even the most-confident answers violate the bound.
    """
    confidences = np.asarray(confidences, dtype=float)
    disagreements = 1 - np.asarray(labels, dtype=int)

    grid = np.linspace(1.0, 0.001, grid_size)
    step = grid[0] - grid[1] if grid_size > 1 else 0.001
    passed = 0
    for tau in grid:
        idx = confidences >= tau
        n = int(idx.sum())
        if n == 0:
            continue
        # Wait until the slightly-wider window contains at least one error.
        if int(disagreements[confidences >= tau - step].sum()) == 0:
            continue
        k = int(disagreements[idx].sum())
        if k >= n:
            # Degenerate window (all wrong): no valid bound. The original
            # walked past these via a NaN comparison; we do the same but do
            # not count them as certified, so all-wrong data still raises.
            continue

        upper = beta.ppf(1 - delta, k + 1, n - k)
        if upper > alpha:
            if passed == 0:
                raise ValueError(
                    f"No threshold guarantees error <= {alpha} at confidence {1 - delta}; "
                    "relax alpha/delta or use a larger calibration set."
                )
            return float(tau)
        passed += 1

    if passed == 0 and int(disagreements.sum()) > 0:
        # Errors exist but no window was ever certifiable (e.g. every answer
        # above every threshold was wrong).
        raise ValueError(
            f"No threshold guarantees error <= {alpha} at confidence {1 - delta}; "
            "relax alpha/delta or use a larger calibration set."
        )
    return 0.0


def coverage(confidences: Sequence[float], threshold: float) -> float:
    """Fraction of calibration answers that would be accepted at ``threshold``."""
    x = np.asarray(confidences, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float((x >= threshold).mean())


def scan_alpha_delta(
    confidences: Sequence[float],
    labels: Sequence[int],
    alphas: Optional[Sequence[float]] = None,
    deltas: Optional[Sequence[float]] = None,
) -> List[dict]:
    """Grid-scan (alpha, delta) -> (threshold, coverage), for picking settings.

    This is the hyperparameter exploration the paper ran per benchmark;
    combinations with no feasible threshold are omitted.
    """
    alphas = alphas if alphas is not None else np.arange(0.1, 0.9, 0.1)
    deltas = deltas if deltas is not None else np.arange(0.1, 0.9, 0.1)
    rows = []
    for a in alphas:
        for d in deltas:
            try:
                thr = calibrate_threshold(confidences, labels, float(a), float(d))
            except ValueError:
                continue
            rows.append(
                {
                    "alpha": round(float(a), 3),
                    "delta": round(float(d), 3),
                    "threshold": round(thr, 4),
                    "coverage": round(coverage(confidences, thr), 4),
                }
            )
    return rows
