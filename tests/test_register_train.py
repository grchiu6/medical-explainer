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
