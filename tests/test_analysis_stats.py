"""Tests for literacy_eval/analysis/stats.py — paired statistics."""

import math

import numpy as np
import pytest

from literacy_eval.analysis import stats as S


def test_page_trend_detects_increase_and_respects_min_n():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 1, size=30)
    increasing = np.column_stack([base, base + 1, base + 2])
    assert S.page_trend(increasing)["p"] < 0.001
    assert math.isnan(S.page_trend(increasing[:5])["p"])


def test_wilcoxon_effect_size_signs():
    low = np.arange(20, dtype=float)
    up = S.wilcoxon_paired(low, low + np.linspace(1, 3, 20))
    assert up["rank_biserial"] == pytest.approx(1.0)
    assert up["share_low_lt_high"] == 1.0
    down = S.wilcoxon_paired(low, low - 1)
    assert down["rank_biserial"] == pytest.approx(-1.0)


def test_cochran_q_matches_hand_computation():
    # 12 items x 3 levels; levels succeed 10, 6, 2 times
    x = np.zeros((12, 3), int)
    x[:10, 0] = 1
    x[:6, 1] = 1
    x[:2, 2] = 1
    res = S.cochran_q(x)
    col, row = x.sum(0), x.sum(1)
    q = 2 * (3 * (col ** 2).sum() - col.sum() ** 2) / (3 * row.sum() - (row ** 2).sum())
    assert res["Q"] == pytest.approx(q)
    assert res["p"] < 0.01


def test_mcnemar_counts_discordant_pairs():
    a = np.array([1, 1, 1, 1, 0, 0, 1, 0])
    b = np.array([0, 0, 0, 1, 0, 1, 1, 0])
    res = S.mcnemar_exact(a, b)
    assert (res["only_a"], res["only_b"]) == (3, 1)
    assert res["p"] == pytest.approx(0.625)


def test_mcnemar_no_discordant_pairs_is_p_one():
    assert S.mcnemar_exact(np.ones(5), np.ones(5))["p"] == 1.0


def test_holm_adjustment():
    assert S.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = S.holm([0.01, float("nan")])
    assert out[0] == pytest.approx(0.01) and math.isnan(out[1])


def test_wilson_ci_zero_successes_has_nonzero_upper_bound():
    lo, hi = S.wilson_ci(0, 60)
    assert lo == 0.0 and 0.05 < hi < 0.07


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    values = np.array([0, 1] * 25, float)
    ci = S.bootstrap_ci(values)
    assert ci[0] < 0.5 < ci[1]
    assert ci == S.bootstrap_ci(values)


def test_page_trend_can_test_a_decreasing_trend():
    base = np.random.default_rng(1).normal(10, 1, size=30)
    decreasing = np.column_stack([base + 2, base + 1, base])
    assert S.page_trend(decreasing, increasing=False)["p"] < 0.001
    assert S.page_trend(decreasing, increasing=True)["p"] > 0.5


def test_weighted_kappa_perfect_agreement_is_one():
    a = np.array([0, 1, 2, 2, 1, 0])
    assert S.weighted_kappa(a, a) == pytest.approx(1.0)


def test_weighted_kappa_hand_computed_quadratic():
    # observed weighted disagreement (0+0+0+.25+.25+1)/6 = .25; expected (uniform marginals) = 1/3
    a, b = np.array([0, 1, 2, 0, 1, 2]), np.array([0, 1, 2, 1, 2, 0])
    assert S.weighted_kappa(a, b) == pytest.approx(0.25, abs=1e-9)


def test_weighted_kappa_linear_weights():
    # linear: observed (0+0+0+.5+.5+1)/6 = 1/3; expected = (1/9) * (4*.5 + 2*1) = 4/9
    a, b = np.array([0, 1, 2, 0, 1, 2]), np.array([0, 1, 2, 1, 2, 0])
    assert S.weighted_kappa(a, b, weights="linear") == pytest.approx(1 - (1 / 3) / (4 / 9), abs=1e-9)


def test_weighted_kappa_constant_ratings_is_nan():
    assert math.isnan(S.weighted_kappa(np.array([1, 1, 1]), np.array([1, 1, 1])))
