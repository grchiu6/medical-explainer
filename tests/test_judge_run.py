"""Tests for literacy_eval/judge/run.py — resume and retry, with a fake generator (no GPU)."""

import json

import pytest

from literacy_eval.judge.jobs import read_jsonl, write_jsonl
from literacy_eval.judge.run import run


@pytest.fixture
def jobs3(tmp_path):
    path = tmp_path / "jobs.jsonl"
    write_jsonl([{"job_id": f"harm|m|low|i{k}", "task": "harm", "prompt": f"P{k}", "prompt_version": "v1"}
                 for k in range(3)], path)
    return path


class FakeGen:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, prompts, max_tokens):
        self.calls.append((list(prompts), max_tokens))
        return [self.reply(p, max_tokens) for p in prompts]


def _ok(p, n):
    return "Feedback: fine [RESULT] 3"


def _run(jobs, out, gen, **kw):
    return run(jobs, out, generate=gen, judge_model="M", judge_revision="abc", **kw)


def test_runs_all_then_resumes_without_rerunning(tmp_path, jobs3):
    out = tmp_path / "out.jsonl"
    assert _run(jobs3, out, FakeGen(_ok)) == 3
    gen = FakeGen(_ok)
    assert _run(jobs3, out, gen) == 0 and gen.calls == []
    assert len(read_jsonl(out)) == 3


def test_unparsable_output_is_retried_once_with_more_tokens(tmp_path, jobs3):
    out = tmp_path / "out.jsonl"
    gen = FakeGen(lambda p, n: "Feedback: cut" if (p == "P1" and n == 512) else "Feedback: ok [RESULT] 3")
    _run(jobs3, out, gen)
    assert [n for _, n in gen.calls] == [512, 1024] and gen.calls[1][0] == ["P1"]
    rows = [r for r in read_jsonl(out) if r["job_id"] == "harm|m|low|i1"]
    assert [(r["ok"], r["max_tokens"]) for r in rows] == [(False, 512), (True, 1024)]


def test_still_unparsable_after_retry_is_not_retried_again(tmp_path, jobs3):
    out = tmp_path / "out.jsonl"
    bad = lambda p, n: "Feedback: never scores" if p == "P2" else "Feedback: [RESULT] 4"
    _run(jobs3, out, FakeGen(bad))
    gen = FakeGen(bad)
    assert _run(jobs3, out, gen) == 0 and gen.calls == []


def test_truncated_last_line_job_is_redone(tmp_path, jobs3):
    out = tmp_path / "out.jsonl"
    good = {"job_id": "harm|m|low|i0", "output": "Feedback: [RESULT] 2", "ok": True, "max_tokens": 512}
    out.write_text(json.dumps(good) + '\n{"job_id": "harm|m|low|i1", "outp', encoding="utf-8")
    gen = FakeGen(_ok)
    _run(jobs3, out, gen)
    assert gen.calls[0][0] == ["P1", "P2"]


def test_match_and_limit_select_jobs(tmp_path, jobs3):
    gen = FakeGen(_ok)
    assert _run(jobs3, tmp_path / "o.jsonl", gen, match="|i2", limit=5) == 1
    assert gen.calls[0][0] == ["P2"]
    gen = FakeGen(_ok)
    assert _run(jobs3, tmp_path / "o2.jsonl", gen, limit=2) == 2


def test_provenance_fields_present(tmp_path, jobs3):
    out = tmp_path / "out.jsonl"
    _run(jobs3, out, FakeGen(_ok))
    row = read_jsonl(out)[0]
    assert row["judge_model"] == "M" and row["judge_revision"] == "abc"
    assert row["prompt_version"] == "v1" and row["run_date"] and row["output"].endswith("3")


def test_resume_twice_after_a_truncated_line_does_not_crash(tmp_path, jobs3):
    out = tmp_path / "out.jsonl"
    good = {"job_id": "harm|m|low|i0", "output": "Feedback: [RESULT] 2", "ok": True, "max_tokens": 512}
    out.write_text(json.dumps(good) + '\n{"job_id": "harm|m|low|i1", "outp', encoding="utf-8")
    _run(jobs3, out, FakeGen(_ok))
    gen = FakeGen(_ok)
    assert _run(jobs3, out, gen) == 0 and gen.calls == []
    assert len(read_jsonl(out)) == 3
