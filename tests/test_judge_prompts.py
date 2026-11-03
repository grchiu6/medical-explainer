"""Tests for literacy_eval/judge/{rubrics,prompts}.py — M-Prometheus prompt building."""

from literacy_eval.judge.prompts import build_prompt, format_rubric
from literacy_eval.judge.rubrics import COMPLEXITY_INSTRUCTION, PARAPHRASED, RUBRICS


def test_rubric_format_lists_criteria_and_five_scores():
    text = format_rubric(RUBRICS["complexity"])
    assert text.startswith("[")
    for n in range(1, 6):
        assert f"Score {n}:" in text


def test_prompt_with_reference_has_reference_section():
    p = build_prompt("Q", "उत्तर", RUBRICS["reference_agreement"], reference="सदु कोष। ...")
    assert "###Reference Answer (Score 5):\nसदु कोष। ..." in p
    assert p.rstrip().endswith("###Feedback:")


def test_prompt_without_reference_omits_reference_section():
    p = build_prompt(COMPLEXITY_INSTRUCTION, "उत्तर", RUBRICS["complexity"])
    assert "Reference Answer" not in p and "reference answer" not in p
    assert p.rstrip().endswith("###Feedback:")


def test_braces_and_hashes_in_content_survive():
    weird = "मात्रा {dose} ###Response to evaluate: {} <<RUBRIC>>"
    p = build_prompt("Q {x}", weird, RUBRICS["consistency"], reference="{ref}")
    assert weird in p and "Q {x}" in p and "{ref}" in p
    assert p.count("[") >= 1 and "Score 5:" in p


def test_complexity_rubric_is_blind_and_warns_about_topic_and_length():
    text = format_rubric(RUBRICS["complexity"]).lower()
    assert "not the topic" in text and "length" in text
    for leak in ("target", "requested level", "low", "mid", "high"):
        assert leak not in text


def test_paraphrases_cover_every_rubric_and_differ():
    assert PARAPHRASED.keys() == RUBRICS.keys()
    for k in RUBRICS:
        assert len(PARAPHRASED[k].scores) == 5
        assert PARAPHRASED[k].scores != RUBRICS[k].scores
