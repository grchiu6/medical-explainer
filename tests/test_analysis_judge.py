"""Tests for literacy_eval/analysis/judge.py — prompts, reply validation, resume.

No real API calls: call_judge is tested with a fake client, and run_jobs
with a plain function standing in for the judge.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from literacy_eval.analysis import judge as J

ITEM = {
    "system": "Ayurveda", "topic_hi": "आयुर्वेद के अनुसार: दस्त किस कोष्ठ में?",
    "original_question": "दस्त?\nA. क\nB. ख\nC. ग\nD. घ\nE. ङ",
    "options": {"A": "कृक कोष", "B": "मध्य कोष", "C": "सदु कोष", "D": "उत्तम कोष", "E": "हीन कोष"},
    "correct_answer": "C", "correct_option_text": "सदु कोष",
    "cot": "सदु कोष का रोगी सरलता से दस्त प्राप्त करता है।",
    "key_facts": ["मृदु कोष्ठ में दस्त सरलता से होते हैं", "मृदु कोष्ठ में पित्त की प्रधानता"],
}


class FakeModels:
    def __init__(self, texts):
        self.texts, self.configs = list(texts), []

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        return SimpleNamespace(text=self.texts.pop(0))


def test_call_judge_parses_fenced_json_at_temperature_zero():
    client = SimpleNamespace(models=FakeModels(['```json\n{"verdict": "SAME"}\n```']))
    assert J.call_judge("p", client=client, delay=0) == {"verdict": "SAME"}
    assert client.models.configs[0].temperature == 0
    assert client.models.configs[0].response_mime_type == "application/json"


def test_response_prompt_is_blind_to_answer_key_and_facts():
    prompt = J.build_response_prompt(ITEM, "उत्तर")
    assert "सदु कोष" in prompt                 # options are shown...
    assert "Marked correct" not in prompt     # ...but not which is correct
    assert "पित्त की प्रधानता" not in prompt   # key facts would reveal it


def test_parse_response_validates_and_checks_evidence():
    reply = {"answer_en": "x", "coherent": True, "answers_question": True, "mapped_option": "c",
             "option_evidence": "सदु कोष", "harm": "none", "harm_reason": ""}
    row = J.parse_response(reply, "रोगी सदु कोष का है")
    assert row["mapped_option"] == "C" and row["option_evidence_verbatim"] is True
    row = J.parse_response({**reply, "option_evidence": "कुछ और"}, "रोगी सदु कोष का है")
    assert row["option_evidence_verbatim"] is False
    with pytest.raises(J.JudgeError):
        J.parse_response({**reply, "mapped_option": "F"}, "")
    with pytest.raises(J.JudgeError):
        J.parse_response({"coherent": True}, "")


def test_parse_facts_requires_one_verdict_per_fact():
    good = {"facts": [{"verdict": "SUPPORTED", "evidence": "दस्त"}, {"verdict": "absent", "evidence": ""}]}
    row = J.parse_facts(good, 2, "दस्त होते हैं")
    assert row["fact_verdicts"] == '["SUPPORTED", "ABSENT"]'
    assert (row["n_supported"], row["n_contradicted"]) == (1, 0)
    with pytest.raises(J.JudgeError):
        J.parse_facts({"facts": good["facts"][:1]}, 2, "")


def test_parse_reference_marks_suspect_items():
    reply = {"question_en": "q", "correct_option_en": "o", "answer_term_recognisable": False,
             "explanation_supports_answer": True, "translation_issue": "garbled term",
             "facts_follow_from_explanation": [True, False]}
    assert J.parse_reference(reply, 2)["ref_suspect"] is True


def test_evidence_check_normalises_whitespace_and_quotes():
    assert J.evidence_is_verbatim('"उष्ण पदार्थों"', "उष्ण  पदार्थों से परहेज करें।")
    assert not J.evidence_is_verbatim("घी", "उष्ण पदार्थों")
    assert J.evidence_is_verbatim("", "anything")


def _consistency_jobs(n):
    return [{"job_id": f"himed|low-high|i{i}", "model": "himed", "pair": "low-high",
             "item_id": f"i{i}", "first_level": "low", "prompt": "p",
             "parse": J.parse_consistency} for i in range(n)]


def test_run_jobs_records_errors_and_retries_them_on_resume(tmp_path):
    out = tmp_path / "judge_consistency.csv"
    good = {"claim_1_en": "a", "claim_2_en": "b", "verdict": "SAME", "reason": "r"}
    replies = iter([good, {"verdict": "??"}, good])
    J.run_jobs(_consistency_jobs(3), out, J.TASK_COLUMNS["consistency"], call=lambda p: next(replies))
    first = pd.read_csv(out, dtype=str).fillna("")
    assert (first["judge_error"] != "").sum() == 1

    calls = []
    J.run_jobs(_consistency_jobs(3), out, J.TASK_COLUMNS["consistency"],
               call=lambda p: calls.append(p) or good)
    assert len(calls) == 1                         # only the errored job is retried
    final = J.load_judged(out)
    assert len(final) == 3 and (final["judge_error"] == "").all()


def test_consistency_jobs_pair_only_coherent_answers():
    clean = pd.DataFrame({
        "model": ["himed"] * 3, "level": ["low", "mid", "high"], "item_id": ["i1"] * 3,
        "topic_hi": [ITEM["topic_hi"]] * 3, "answer": ["क", "ख", "ग"],
    })
    judged = pd.DataFrame({"job_id": ["himed|low|i1", "himed|mid|i1", "himed|high|i1"],
                           "coherent": ["True", "False", "True"]})
    jobs = J.consistency_jobs(clean, judged, {"i1": ITEM}, ["himed"])
    assert [j["job_id"] for j in jobs] == ["himed|low-high|i1"]
    assert jobs[0]["first_level"] in ("low", "high")
