"""Tests for literacy_eval/register/train.py — the ordinal (CORAL) head's targets, decoding and metrics."""

import numpy as np
import pytest
import torch

from literacy_eval.register import train as T


def test_ordinal_targets():
    assert T.ordinal_targets(torch.tensor([0, 1, 2])).tolist() == [[0, 0], [1, 0], [1, 1]]


def test_decode_gives_valid_probabilities_and_expected_level():
    # very negative -> grade5; very positive -> professional; P(>0) high, P(>1) low -> grade10
    logits = torch.tensor([[-9.0, -9.0], [9.0, -9.0], [9.0, 9.0], [0.0, 3.0]])
    pred, probs, level = T.decode(logits)
    assert pred.tolist()[:3] == [0, 1, 2]
    assert np.allclose(probs.sum(1), 1) and (probs >= 0).all()      # last row: P(>1) clipped to P(>0)
    assert level[0] == pytest.approx(0, abs=1e-3) and level[2] == pytest.approx(2, abs=1e-3)


def test_metrics_macro_f1_and_confusion():
    m = T.metrics(np.array([0, 1, 2, 2]), [0, 1, 1, 2])
    assert m["accuracy"] == 0.75
    assert m["confusion_gold_x_pred"] == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]
    assert m["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)


def test_middle_class_can_win_with_the_initial_threshold_gap():
    # Regression: the first training run started with the thresholds only 1.0 apart, which caps P(10th) at 0.245,
    # so the middle class was NEVER predicted (all 4 runs). Score s = 0 sits midway between the thresholds.
    model = T.OrdinalClassifier("hf-internal-testing/tiny-random-bert")
    thresholds = torch.stack([model.b1, model.b1 + torch.exp(model.log_gap)]).detach()
    logits = (0.0 - thresholds).unsqueeze(0)
    pred, probs, _ = T.decode(logits)
    assert pred.tolist() == [1], probs          # the middle class wins when the score is between the thresholds


def test_head_and_thresholds_get_the_larger_learning_rate():
    model = T.OrdinalClassifier("hf-internal-testing/tiny-random-bert")
    opt = T.build_optimizer(model, lr=2e-5, head_lr=1e-3)
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs == [2e-5, 1e-3, 1e-3]
    assert opt.param_groups[2]["weight_decay"] == 0.0          # thresholds: no weight decay
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in model.parameters()}   # nothing left out
