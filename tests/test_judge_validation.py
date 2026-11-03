"""Tests for literacy_eval/judge/validation.py — validation jobs and score -> level cut-points."""

import json

import numpy as np
import pandas as pd
import pytest

from literacy_eval.judge.jobs import reference_answer
from literacy_eval.judge.prompts import build_prompt, format_rubric
from literacy_eval.judge.rubrics import PARAPHRASED, RUBRICS
from literacy_eval.judge.validation import (EXPECTED_BAND, apply_cutpoints, controlled_cases,
                                            fit_cutpoints, readmepp_sample, validation_jobs)


def _item(i, correct="सही विकल्प", cot=None):
    return {"id": i, "topic_hi": f"आयुर्वेद के अनुसार: प्रश्न {i}?", "system": "Ayurveda",
            "options": {"A": "पहला गलत", "B": correct, "C": "दूसरा गलत"}, "correct_answer": "B",
            "correct_option_text": correct,
            "cot": cot if cot is not None else f"{correct} इसका उत्तर है ({i})।", "key_facts": []}


@pytest.fixture
def ref2():
    return {"i1": _item("i1"), "i2": _item("i2", cot="व्याख्या में उत्तर का नाम नहीं।")}


def test_controlled_cases_four_per_item(ref2):
    cases = controlled_cases(ref2)
    assert len(cases) == 8
    assert {c["case"] for c in cases} == set(EXPECTED_BAND)


def test_positive_is_the_reference_answer(ref2):
    pos = [c for c in controlled_cases(ref2) if c["case"] == "positive" and c["item_id"] == "i1"][0]
    assert pos["response"] == reference_answer(ref2["i1"]) == pos["positive_text"]


def test_swap_replaces_correct_option_with_first_wrong_option(ref2):
    sw = {c["item_id"]: c for c in controlled_cases(ref2) if c["case"] == "swapped"}
    assert "सही विकल्प" not in sw["i1"]["response"] and "पहला गलत" in sw["i1"]["response"]
    # i2's CoT never names the answer: wrong option is prepended to the CoT
    assert sw["i2"]["response"] == "पहला गलत। व्याख्या में उत्तर का नाम नहीं।"


def test_unrelated_case_uses_the_next_item(ref2):
    un = {c["item_id"]: c["response"] for c in controlled_cases(ref2) if c["case"] == "unrelated"}
    assert un["i1"] == reference_answer(ref2["i2"]) and un["i2"] == reference_answer(ref2["i1"])


def test_non_answer_is_the_question_stem(ref2):
    na = [c for c in controlled_cases(ref2) if c["case"] == "non_answer" and c["item_id"] == "i1"][0]
    assert na["response"] == "प्रश्न i1?"


def test_readmepp_sample_is_stratified_and_handles_string_ratings():
    rows = [{"id": f"r{k}", "text": "वाक्य", "rating": f"{(k % 6) + 1}.0", "domain": "Wikipedia"}
            for k in range(600)]
    sample = readmepp_sample(rows, n=300, seed=42)
    assert len(sample) == 300 and len({r["id"] for r in sample}) == 300
    counts = pd.Series([round(float(r["rating"])) for r in sample]).value_counts()
    assert set(counts.index) == {1, 2, 3, 4, 5, 6} and counts.min() == 50


def test_readmepp_sample_fills_from_other_ratings_when_one_is_thin():
    rows = [{"id": f"a{k}", "rating": "1.0", "text": "x"} for k in range(10)] + \
           [{"id": f"b{k}", "rating": "2.0", "text": "x"} for k in range(200)]
    sample = readmepp_sample(rows, n=100, seed=42)
    assert len(sample) == 100 and sum(r["rating"] == "1.0" for r in sample) == 10


def test_fit_cutpoints_recovers_clean_separation():
    s = np.array([1, 1, 2, 3, 3, 4, 5, 5]); lv = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    assert fit_cutpoints(s, lv) == (2.5, 3.5)
    assert apply_cutpoints(s, (2.5, 3.5)).tolist() == lv.tolist()


def test_fit_cutpoints_finds_shifted_boundaries():
    s = np.array([1, 2, 3, 4, 4, 5]); lv = np.array([0, 0, 0, 1, 1, 2])
    assert fit_cutpoints(s, lv) == (3.5, 4.5)


def test_fit_cutpoints_tie_breaks_toward_2_5_and_3_5():
    assert fit_cutpoints(np.array([3, 3, 3]), np.array([0, 1, 2])) == (2.5, 3.5)


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def test_validation_jobs_ids_unique_and_complete(tmp_path, ref2):
    corpus = tmp_path / "corpus"; corpus.mkdir()
    _write(corpus / "val.jsonl", [{"text": "आसान", "label": "grade5", "source": "ncert"}])
    _write(corpus / "test.jsonl", [{"text": "रोग का लक्षण", "label": "grade5", "source": "ncert"},
                                   {"text": "जटिल", "label": "professional", "source": "icmr"}])
    evald = tmp_path / "eval"; evald.mkdir()
    _write(evald / "readmepp_hi.jsonl", [{"id": "r1", "text": "वाक्य", "rating": "2.0", "domain": "Wikipedia"}])
    _write(evald / "miracl_medical.jsonl", [{"id": "m1", "text": "रोग", "title": "t"}])
    llm = [{"job_id": f"term_explanation|himed|low|i{k}", "task": "term_explanation", "model": "himed", "level": "low",
            "item_id": "i1", "prompt": build_prompt("q", "उत्तर", RUBRICS["term_explanation"]), "prompt_version": "v1"}
           for k in range(20)]

    jobs = validation_jobs(ref2, llm, corpus_dir=corpus, eval_dir=evald)
    ids = [j["job_id"] for j in jobs]
    assert len(ids) == len(set(ids))
    assert "complexity|corpus:val|0" in ids and "complexity|corpus:test|1" in ids
    assert "complexity|readmepp|r1" in ids and "complexity|miracl|m1" in ids
    assert "reference_agreement|controlled:swapped|i1" in ids and "consistency|controlled:positive|i2" in ids
    assert sum(i.startswith("para:") for i in ids) == 2           # 10% of 20 LLM jobs
    test0 = next(j for j in jobs if j["job_id"] == "complexity|corpus:test|0")
    assert test0["label"] == "grade5" and test0["health"] is True
    assert test0["n_words"] == 3
    para = next(j for j in jobs if j["job_id"].startswith("para:"))
    assert format_rubric(PARAPHRASED["term_explanation"]) in para["prompt"]
    assert format_rubric(RUBRICS["term_explanation"]) not in para["prompt"]
    cons = next(j for j in jobs if j["job_id"] == "consistency|controlled:swapped|i1")
    assert "###Reference Answer (Score 5):\n" + reference_answer(ref2["i1"]) in cons["prompt"]


def test_paraphrase_of_a_prompt_without_its_rubric_raises(ref2, tmp_path):
    llm = [{"job_id": "term_explanation|himed|low|i1", "task": "term_explanation", "prompt": "no rubric here"}] * 10
    with pytest.raises(ValueError):
        validation_jobs(ref2, llm, corpus_dir=tmp_path, eval_dir=tmp_path, include_sets=False)


def test_health_set_pools_grade5_health_passages_from_all_splits(tmp_path, ref2):
    corpus = tmp_path / "corpus"; corpus.mkdir(); evald = tmp_path / "eval"; evald.mkdir()
    _write(corpus / "train.jsonl", [{"text": "बुखार और रोग", "label": "grade5"}, {"text": "खेल", "label": "grade5"},
                                    {"text": "रोग", "label": "professional"}])
    _write(corpus / "val.jsonl", [{"text": "रोग से बचाव", "label": "grade5"}])
    _write(corpus / "test.jsonl", [{"text": "आसान", "label": "grade5"}])
    _write(evald / "readmepp_hi.jsonl", [{"id": "r1", "text": "वाक्य", "rating": "2.0"}])
    _write(evald / "miracl_medical.jsonl", [{"id": "m1", "text": "रोग", "title": "t"}])
    ids = {j["job_id"] for j in validation_jobs(ref2, [], corpus_dir=corpus, eval_dir=evald)
           if j.get("set") == "corpus:health"}
    assert ids == {"complexity|corpus:health|train:0", "complexity|corpus:health|val:0"}
