"""Gemini 3.8 Flash wrapper tests — fake client, no network."""

from types import SimpleNamespace

import pytest

import models.gemini_flash as gf
from performance_eval.score import extract_answer


def _resp(text="B", prompt=100, out=1, thoughts=50, finish="STOP"):
    return SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=prompt, candidates_token_count=out,
                                       thoughts_token_count=thoughts),
        candidates=[SimpleNamespace(finish_reason=finish)],
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def fake(monkeypatch):
    def install(*responses):
        client = FakeClient(responses)
        monkeypatch.setattr(gf, "_get_client", lambda: client)
        return client

    monkeypatch.setattr(gf, "USAGE", gf._empty_usage())
    monkeypatch.setattr(gf.time, "sleep", lambda s: None)
    monkeypatch.delenv("GEMINI_BUDGET_USD", raising=False)
    return install


def test_request_uses_temperature_zero_and_low_thinking(fake):
    client = fake(_resp())
    assert gf.get_answer("prompt", gen_kwargs={"do_sample": False, "max_new_tokens": 5}) == "B"
    call = client.calls[0]
    assert call["model"] == gf.MODEL_ID == "gemini-3.8-flash"
    assert call["config"].temperature == 0
    assert call["config"].thinking_config.thinking_level.value == gf.DEFAULT_THINKING_LEVEL


def test_thinking_level_override(fake):
    client = fake(_resp())
    gf.get_answer("p", gen_kwargs={"thinking_level": "HIGH"})
    assert client.calls[0]["config"].thinking_config.thinking_level.value == "HIGH"


def test_usage_accumulates_and_costs(fake):
    fake(_resp(), _resp())
    gf.get_answer("p")
    gf.get_answer("p")
    u = gf.USAGE
    assert u["calls"] == 2 and u["prompt_tokens"] == 200 and u["output_tokens"] == 2
    assert u["thoughts_tokens"] == 100
    assert u["cost_usd"] == pytest.approx(2 * (100 * 0.75 + 51 * 3.75) / 1e6)


def test_usage_resumes_from_file(fake, tmp_path):
    fake(_resp())
    gf.get_answer("p")
    path = tmp_path / "u.json"
    gf.save_usage(path)
    saved = dict(gf.USAGE)
    gf.USAGE.update(gf._empty_usage())
    gf.load_usage(path)
    assert gf.USAGE == saved


def test_load_usage_missing_file_keeps_zero(fake, tmp_path):
    gf.load_usage(tmp_path / "nope.json")
    assert gf.USAGE == gf._empty_usage()


def test_budget_guard_blocks_call(fake, monkeypatch):
    client = fake(_resp())
    monkeypatch.setenv("GEMINI_BUDGET_USD", "0.000001")
    gf.USAGE["cost_usd"] = 1.0
    with pytest.raises(gf.BudgetExceeded):
        gf.get_answer("p")
    assert client.calls == []


def test_default_budget_is_two_dollars(fake):
    client = fake(_resp())
    gf.USAGE["cost_usd"] = 2.0
    with pytest.raises(gf.BudgetExceeded):
        gf.get_answer("p")
    assert client.calls == []


def test_blocked_response_returns_empty(fake):
    fake(_resp(text=None, out=0, thoughts=0, finish="SAFETY"))
    assert gf.get_answer("p") == ""
    assert gf.USAGE["blocked"] == 1


def test_transient_error_retried(fake):
    client = fake(RuntimeError("503"), _resp(text="C"))
    assert gf.get_answer("p") == "C"
    assert len(client.calls) == 2


def test_markdown_letter_scored():
    assert extract_answer("**B**", "first_letter") == "B"
    assert extract_answer("Answer: B", "first_letter") == "B"


def test_run_gemini_records_model_id_and_usage_sidecar(tmp_path, fakes, monkeypatch):
    import json

    from performance_eval import run as run_mod

    saved = []
    mod = SimpleNamespace(MODEL_ID="gemini-3.8-flash",
                          load_usage=lambda p: saved.append(("load", p)),
                          save_usage=lambda p: saved.append(("save", p)))
    monkeypatch.setattr(run_mod, "_load_model_module", lambda name: mod)
    run_mod.main(["--model", "gemini_flash", "--condition", "c0", "--limit", "2",
                  "--output-dir", str(tmp_path)])
    cfg = json.loads((tmp_path / "gemini_flash_c0_raw.csv.config.json").read_text(encoding="utf-8"))
    assert cfg["model_id"] == "gemini-3.8-flash"
    usage = tmp_path / "gemini_flash_c0_raw.csv.usage.json"
    assert saved[0] == ("load", usage)
    assert saved.count(("save", usage)) == 2  # once per answered item


def test_run_local_model_has_no_model_id(tmp_path, fakes):
    import json

    from performance_eval import run as run_mod

    run_mod.main(["--model", "himed", "--condition", "c0", "--limit", "1", "--output-dir", str(tmp_path)])
    cfg = json.loads((tmp_path / "himed_c0_raw.csv.config.json").read_text(encoding="utf-8"))
    assert cfg["model_id"] is None


from tests.test_performance_run import fakes  # noqa: E402,F401


class _ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


def test_client_error_not_retried(fake):
    client = fake(_ApiError(402), _resp())
    with pytest.raises(_ApiError):
        gf.get_answer("p")
    assert len(client.calls) == 1


def test_rate_limit_is_retried(fake):
    client = fake(_ApiError(429), _resp(text="D"))
    assert gf.get_answer("p") == "D"
    assert len(client.calls) == 2
