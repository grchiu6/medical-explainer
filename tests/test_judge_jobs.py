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
            "answer": answer, "usable_rules": usable}


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


def test_only_usable_nonempty_rows_become_jobs(toy):
    items = {j["item_id"] for j in llm_jobs(toy.clean, toy.ref)}
    assert items == {"i1", "i2"}


def test_consistency_needs_both_levels_usable(toy):
    ids = {j["job_id"] for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "consistency"}
    assert ids == {"consistency|himed|low-high|i1"}


def test_term_explanation_only_low_and_mid(toy):
    levels = {j["level"] for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "term_explanation"}
    assert levels == {"low", "mid"}


def test_job_ids_and_counts(toy):
    jobs = llm_jobs(toy.clean, toy.ref)
    assert len({j["job_id"] for j in jobs}) == len(jobs)
    assert "complexity|himed|high|i1" in {j["job_id"] for j in jobs}
    by_task = pd.Series([j["task"] for j in jobs]).value_counts().to_dict()
    assert by_task == {"complexity": 4, "reference_agreement": 4, "harm": 4, "term_explanation": 3,
                       "consistency": 1}


def test_complexity_is_blind_and_normalised(toy):
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "complexity")
    assert "प्रश्न" not in j["prompt"]                  # no topic
    assert "Reference Answer" not in j["prompt"] and "himed" not in j["prompt"]
    assert "2015" not in j["prompt"] and "वर्ष 0 से" in j["prompt"]


def test_judge2_keeps_raw_numbers(toy):
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "harm")
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


def test_usable_column_as_strings_is_handled(toy):
    clean = toy.clean.assign(usable_rules=toy.clean["usable_rules"].astype(str))
    assert {j["item_id"] for j in llm_jobs(clean, toy.ref)} == {"i1", "i2"}


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
