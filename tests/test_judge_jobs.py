"""Tests for literacy_eval/judge/jobs.py — one judge prompt per (task, response)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from literacy_eval.judge.jobs import llm_jobs, read_jsonl, reference_answer, write_jsonl


def _item(i):
    return {"id": i, "topic_hi": f"आयुर्वेद के अनुसार: प्रश्न {i}?", "system": "Ayurveda",
            "correct_option_text": f"सही {i}", "cot": f"व्याख्या {i}।",
            "key_facts": [f"तथ्य-एक-{i}", f"तथ्य-दो-{i}"],
            "options": {"A": f"सही {i}", "B": "गलत"}, "correct_answer": "A"}


def _row(level, item, usable=True, answer="यह दवा वर्ष 2015 से दी जाती है।", model="himed"):
    return {"model": model, "level": level, "item_id": item, "topic_hi": f"आयुर्वेद के अनुसार: प्रश्न {item}?",
            "answer": answer, "usable_rules": usable, "failure_type": "ok" if usable else "degenerate"}


@pytest.fixture
def toy():
    rows = [
        _row("low", "i1"), _row("mid", "i1", usable=False), _row("high", "i1"),
        _row("low", "i2"), _row("mid", "i2"),                       # i2 has no high answer
        _row("low", "unusable_item", usable=False), _row("high", "unusable_item", usable=False),
        _row("high", "empty_item", answer=""),
    ]
    ref = {i: _item(i) for i in ("i1", "i2", "unusable_item", "empty_item")}
    return SimpleNamespace(clean=pd.DataFrame(rows), ref=ref)


def test_every_response_with_text_becomes_a_job_but_generation_failures_do_not(toy):
    # poor responses (rule-unusable: loops, echoes, refusals...) ARE judged; empty output is a generation failure
    items = {j["item_id"] for j in llm_jobs(toy.clean, toy.ref)}
    assert items == {"i1", "i2", "unusable_item"}


def test_consistency_pairs_need_both_levels_to_have_text(toy):
    ids = {j["job_id"] for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "consistency"}
    assert ids == {"consistency|himed|low-high|i1", "consistency|himed|mid-high|i1",
                   "consistency|himed|low-high|unusable_item"}


def test_term_explanation_only_low_and_mid(toy):
    levels = {j["level"] for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "term_explanation"}
    assert levels == {"low", "mid"}


def test_job_ids_and_counts(toy):
    jobs = llm_jobs(toy.clean, toy.ref)
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert "complexity|himed|high|i1" in {j["job_id"] for j in jobs}
    by_task = pd.Series([j["task"] for j in jobs]).value_counts().to_dict()
    assert by_task == {"complexity": 7, "reference_agreement": 7, "term_explanation": 5, "consistency": 3}


def test_complexity_is_blind_and_normalised(toy):
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "complexity")
    assert "प्रश्न" not in j["prompt"]                  # no topic
    assert "Reference Answer" not in j["prompt"] and "himed" not in j["prompt"]
    assert "2015" not in j["prompt"] and "वर्ष 0 से" in j["prompt"]


def test_judge2_keeps_raw_numbers(toy):
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "reference_agreement")
    assert "2015" in j["prompt"]


def test_reference_uses_option_and_cot_never_key_facts(toy):
    assert reference_answer(toy.ref["i1"]) == "सही i1। व्याख्या i1।"
    for j in llm_jobs(toy.clean, toy.ref):
        assert all(f not in j["prompt"] for f in toy.ref[j["item_id"]]["key_facts"])
        if j["task"] == "reference_agreement":
            assert reference_answer(toy.ref[j["item_id"]]) in j["prompt"]


def test_consistency_reference_is_the_high_answer(toy):
    clean = toy.clean.copy()
    clean.loc[(clean.item_id == "i1") & (clean.level == "high"), "answer"] = "उच्च स्तर का उत्तर।"
    j = next(j for j in llm_jobs(clean, toy.ref) if j["task"] == "consistency")
    assert "###Reference Answer (Score 5):\nउच्च स्तर का उत्तर।" in j["prompt"]
    assert j["pair"] == "low-high"


def test_jobs_carry_failure_type_and_rule_usability(toy):
    jobs = {j["job_id"]: j for j in llm_jobs(toy.clean, toy.ref)}
    assert jobs["complexity|himed|low|unusable_item"]["failure_type"] == "degenerate"
    assert jobs["complexity|himed|low|unusable_item"]["usable_rules"] is False
    assert jobs["complexity|himed|low|i1"]["failure_type"] == "ok"
    clean = toy.clean.assign(usable_rules=toy.clean["usable_rules"].astype(str))   # as read from CSV
    assert {j["job_id"]: j for j in llm_jobs(clean, toy.ref)}["complexity|himed|low|i1"]["usable_rules"] is True


def test_jsonl_roundtrip_skips_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl([{"a": 1, "t": "हिंदी"}], p)
    p.write_text(p.read_text(encoding="utf-8") + '{"a": 2', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1, "t": "हिंदी"}]


def test_jsonl_broken_middle_line_raises(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(p)


def test_harm_task_was_removed(toy):
    from literacy_eval.judge.jobs import TASKS
    from literacy_eval.judge.rubrics import PARAPHRASED, RUBRICS
    assert "harm" not in TASKS and "harm" not in RUBRICS and "harm" not in PARAPHRASED
    assert all(j["task"] != "harm" for j in llm_jobs(toy.clean, toy.ref))
