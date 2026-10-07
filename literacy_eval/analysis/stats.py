"""Paired statistics for the HiMed-Trad evaluation (pure functions, no I/O).

WHY PAIRED TESTS:
The same 60 items are asked at every level, so each item is its own control.
Paired tests compare low vs high WITHIN an item, which removes item difficulty
from the comparison and gives far more power than unpaired tests.

  page_trend        ordered hypothesis low < mid < high (Page's L test)
  wilcoxon_paired   low vs high, with matched-pairs rank-biserial effect size
  cochran_q         paired binary outcome across 3 levels (e.g. usable)
  mcnemar_exact     paired binary outcome, 2 levels
  holm              multiple-comparison correction within a family of tests
  wilson_ci         95% CI for a proportion
  bootstrap_ci      95% CI for any statistic, resampling items
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

MIN_PAIRED_N = 10   # below this, report "n too small" instead of a p-value


def page_trend(matrix: np.ndarray, increasing: bool = True) -> dict:
    """Page's L test for an ordered trend across columns (low, mid, high).

    matrix: shape (n_items, 3), columns ordered [low, mid, high], no NaNs.
    increasing=False tests low > mid > high instead.
    """
    n = len(matrix)
    if n < MIN_PAIRED_N:
        return {"n": n, "L": float("nan"), "p": float("nan")}
    res = stats.page_trend_test(matrix, predicted_ranks=[1, 2, 3] if increasing else [3, 2, 1])
    return {"n": n, "L": float(res.statistic), "p": float(res.pvalue)}


def wilcoxon_paired(low: np.ndarray, high: np.ndarray) -> dict:
    """Wilcoxon signed-rank test, high vs low, plus effect size.

    rank_biserial is in [-1, 1]: +1 means high > low for every item.
    share_low_lt_high is the plain-language version: the fraction of items
    whose low answer scored lower than its high answer.
    """
    low, high = np.asarray(low, float), np.asarray(high, float)
    diff = high - low
    n = len(diff)
    share = float(np.mean(diff > 0)) if n else float("nan")
    nonzero = diff[diff != 0]
    if n < MIN_PAIRED_N or len(nonzero) == 0:
        return {"n": n, "p": float("nan"), "rank_biserial": float("nan"),
                "share_low_lt_high": share, "median_diff": float(np.median(diff)) if n else float("nan")}
    ranks = stats.rankdata(np.abs(nonzero))
    w_pos, w_neg = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    res = stats.wilcoxon(high, low, zero_method="wilcox")
    return {
        "n": n,
        "p": float(res.pvalue),
        "rank_biserial": float((w_pos - w_neg) / (w_pos + w_neg)),
        "share_low_lt_high": share,
        "median_diff": float(np.median(diff)),
    }


def cochran_q(matrix: np.ndarray) -> dict:
    """Cochran's Q for a binary (0/1) matrix of shape (n_items, k_levels)."""
    x = np.asarray(matrix, float)
    n, k = x.shape
    col, row = x.sum(axis=0), x.sum(axis=1)
    denom = k * row.sum() - (row ** 2).sum()
    if n < MIN_PAIRED_N or denom == 0:
        return {"n": n, "Q": float("nan"), "p": float("nan")}
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denom
    return {"n": n, "Q": float(q), "p": float(stats.chi2.sf(q, k - 1))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test for paired binary outcomes a, b (0/1 arrays)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    discordant = only_a + only_b
    if discordant == 0:
        return {"n": len(a), "only_a": only_a, "only_b": only_b, "p": 1.0}
    p = stats.binomtest(only_a, discordant, 0.5).pvalue
    return {"n": len(a), "only_a": only_a, "only_b": only_b, "p": float(p)}


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaNs are passed through)."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvalues[i])
    m = len(order)
    adjusted = list(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 2000,
                 seed: int = 42) -> tuple[float, float]:
    """95% percentile bootstrap CI, resampling items with replacement."""
    values = np.asarray(values)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
