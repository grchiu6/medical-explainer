"""Tests for literacy_eval/judge/parse.py — reading the score out of an M-Prometheus reply."""

import pytest

from literacy_eval.judge.parse import ParseError, parse_result


@pytest.mark.parametrize("text,score", [
    ("Feedback: Simple words. [RESULT] 2", 2),
    ("Feedback: ok [[RESULT]] 4", 4),          # the paper's spelling
    ("Feedback: ok [RESULT] 5.", 5),
    ("Feedback: ok [RESULT]: 3\n", 3),
    ("Feedback: mentions [RESULT] 1 as format. Final: [RESULT] 4", 4),   # last tag wins
    ("No prefix here [RESULT] 2", 2),
])
def test_score_variants(text, score):
    assert parse_result(text)[1] == score


def test_feedback_is_stripped_of_prefix_and_tag():
    assert parse_result("Feedback: Uses everyday words. [RESULT] 1") == ("Uses everyday words.", 1)


@pytest.mark.parametrize("text", ["Feedback: good. Score: 4", "Feedback: [RESULT] 7",
                                  "Feedback: [RESULT] 0", "", "Feedback: cut off mid-sent",
                                  "Feedback: ok [RESULT] 45"])
def test_unparsable_raises(text):
    with pytest.raises(ParseError):
        parse_result(text)
