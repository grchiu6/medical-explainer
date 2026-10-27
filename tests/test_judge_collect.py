"""Tests for literacy_eval/judge/{collect,reference_audit}.py."""

import pandas as pd
import pytest

from literacy_eval.judge.collect import collect
from literacy_eval.judge.jobs import write_jsonl
from literacy_eval.judge.reference_audit import export_audit, load_suspect


def _res(job_id, output, ok, n=512):
    return {"job_id": job_id, "output": output, "ok": ok, "max_tokens": n, "judge_model": "M",
            "judge_revision": "abc", "prompt_version": "v1", "run_date": "2026-10-27"}


@pytest.fixture
def scored(tmp_path):
    jobs = [{"job_id": f"harm|himed|low|i{k}", "task": "harm", "model": "himed", "level": "low",
             "item_id": f"i{k}", "prompt": "P", "prompt_version": "v1"} for k in range(3)]
    results = [_res("harm|himed|low|i0", "Feedback: cut", False),
               _res("harm|himed|low|i0", "Feedback: fine [RESULT] 4", True, 1024),
               _res("harm|himed|low|i1", "Feedback: never", False, 512),
               _res("harm|himed|low|i1", "Feedback: still never", False, 1024)]
    write_jsonl(jobs, tmp_path / "jobs.jsonl"); write_jsonl(results, tmp_path / "res.jsonl")
    return collect(tmp_path / "jobs.jsonl", tmp_path / "res.jsonl").set_index("job_id")


def test_latest_ok_row_wins(scored):
    row = scored.loc["harm|himed|low|i0"]
    assert row["score"] == 4 and row["feedback"] == "fine" and row["judge_error"] == ""


def test_unparsable_keeps_error_text(scored):
    assert "RESULT" in scored.loc["harm|himed|low|i1", "judge_error"]
    assert pd.isna(scored.loc["harm|himed|low|i1", "score"])


def test_missing_result_is_marked(scored):
    assert scored.loc["harm|himed|low|i2", "judge_error"] == "missing"


def test_score_dtype_and_no_prompt_column(scored):
    assert str(scored["score"].dtype) == "Int64" and "prompt" not in scored.columns
    assert scored.loc["harm|himed|low|i0", "judge_revision"] == "abc"


REF = {"i1": {"id": "i1", "system": "Ayurveda", "subject": "treatment", "original_question": "प्रश्न?\nA. क",
              "topic_hi": "आयुर्वेद के अनुसार: प्रश्न?", "correct_option_text": "सही", "cot": "व्याख्या।"},
       "i2": {"id": "i2", "system": "Unani", "subject": "diagnosis", "original_question": "प्रश्न २?",
              "topic_hi": "यूनानी के अनुसार: प्रश्न २?", "correct_option_text": "सही २", "cot": "व्याख्या २।"}}


def test_export_audit_columns_and_blank_answers(tmp_path):
    path = tmp_path / "audit.csv"
    export_audit(REF, path)
    df = pd.read_csv(path, dtype=str).fillna("")
    assert list(df.columns) == ["item_id", "system", "subject", "question_hi", "correct_option_hi", "cot_hi",
                                "question_en", "correct_option_en", "cot_en", "ref_suspect", "note"]
    assert len(df) == 2 and (df["ref_suspect"] == "").all() and df.loc[0, "question_hi"] == "प्रश्न?"


def test_export_refuses_to_overwrite_filled_audit(tmp_path):
    path = tmp_path / "audit.csv"
    export_audit(REF, path)
    df = pd.read_csv(path, dtype=str).fillna(""); df.loc[1, "ref_suspect"] = "Y"; df.to_csv(path, index=False)
    with pytest.raises(FileExistsError):
        export_audit(REF, path)


def test_load_suspect(tmp_path):
    path = tmp_path / "audit.csv"
    assert load_suspect(path) is None                     # no file
    export_audit(REF, path)
    assert load_suspect(path) is None                     # nothing filled in yet
    df = pd.read_csv(path, dtype=str).fillna(""); df.loc[1, "ref_suspect"] = "Y"; df.loc[0, "ref_suspect"] = "n"
    df.to_csv(path, index=False)
    assert load_suspect(path) == {"i2"}
