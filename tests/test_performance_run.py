import csv

import pytest

from performance_eval import run as run_mod
from performance_eval.run import build_gen_kwargs, run_condition, validate_args

GREEDY = {"do_sample": False, "temperature": None, "top_p": None}


def test_run_condition_writes_one_row_per_item(tmp_path):
    items = [{"id": "m1", "question_with_options": "Q1?\nA. x\nB. y", "answer_letter": "A", "n_options": 2}]
    fake_get_answer = lambda prompt, gen_kwargs=None: "Answer: A"
    out = tmp_path / "out.csv"
    run_condition(items, [], get_answer_fn=fake_get_answer, use_cot=False, use_fewshot=False,
                  gen_kwargs={"do_sample": False}, output_path=out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["item_id"] == "m1"
    assert rows[0]["reference_answer"] == "A"
    assert rows[0]["model_answer"] == "Answer: A"


def test_run_condition_resumes_by_item_id_skipping_done_rows(tmp_path):
    out = tmp_path / "out.csv"
    out.write_text("item_id,question_with_options,reference_answer,model_answer\nm1,Q1,A,done\n", encoding="utf-8")
    items = [
        {"id": "m1", "question_with_options": "Q1", "answer_letter": "A", "n_options": 2},
        {"id": "m2", "question_with_options": "Q2", "answer_letter": "B", "n_options": 2},
    ]
    calls = []
    fake_get_answer = lambda prompt, gen_kwargs=None: calls.append(prompt) or "Answer: B"
    run_condition(items, [], get_answer_fn=fake_get_answer, use_cot=False, use_fewshot=False,
                  gen_kwargs={}, output_path=out)
    assert len(calls) == 1
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert {r["item_id"] for r in rows} == {"m1", "m2"}


def test_run_condition_passes_fewshot_pool_into_prompt_when_enabled(tmp_path, monkeypatch):
    captured = {}

    def fake_build_prompt(question_with_options, n_options, use_cot=False, exemplars=None):
        captured["exemplars"] = exemplars
        return "PROMPT"

    monkeypatch.setattr("performance_eval.run.build_prompt", fake_build_prompt)
    items = [{"id": "m1", "question_with_options": "Q1", "answer_letter": "A", "n_options": 2}]
    pool = [{"question_with_options": "ex?", "answer_letter": "A"}]
    out = tmp_path / "out.csv"
    run_condition(items, pool, get_answer_fn=lambda p, gen_kwargs=None: "Answer: A",
                  use_cot=False, use_fewshot=True, gen_kwargs={}, output_path=out)
    assert captured["exemplars"] == pool


def test_build_gen_kwargs_greedy_default():
    assert build_gen_kwargs(False) == GREEDY


def test_build_gen_kwargs_sampling_default_is_empty():
    assert build_gen_kwargs(True) == {}


def test_build_gen_kwargs_overrides():
    assert build_gen_kwargs(True, max_new_tokens=512) == {"max_new_tokens": 512}
    assert build_gen_kwargs(True, temperature=0.7) == {"temperature": 0.7}
    assert build_gen_kwargs(True, top_p=0.9) == {"top_p": 0.9}
    assert build_gen_kwargs(True, repetition_penalty=1.1) == {"repetition_penalty": 1.1}


def test_build_gen_kwargs_overrides_on_top_of_greedy():
    kw = build_gen_kwargs(False, max_new_tokens=256, repetition_penalty=1.2)
    assert kw == {**GREEDY, "max_new_tokens": 256, "repetition_penalty": 1.2}


def test_validate_args_rejects_sampling_knobs_without_sample_decoding():
    for knob in ("temperature", "top_p"):
        with pytest.raises(ValueError):
            validate_args(sample_decoding=False, **{knob: 0.5})
    validate_args(sample_decoding=True, temperature=0.5, top_p=0.9)
    validate_args(sample_decoding=False, repetition_penalty=1.1)


def _fake_items(n=12):
    return [
        {"id": f"i{k}", "question": f"q{k}", "options": {c: c.lower() for c in "ABCD"},"question_with_options": f"Q{k}\nA. a\nB. b\nC. c\nD. d",
         "answer_letter": "ABCD"[k % 4], "n_options": 4}
        for k in range(n)
    ]


@pytest.fixture
def fakes(monkeypatch):
    rec = {"kwargs": [], "prompts": []}

    def fake_get_answer(prompt, gen_kwargs=None):
        rec["prompts"].append(prompt)
        rec["kwargs"].append(gen_kwargs)
        return "Answer: A"

    monkeypatch.setattr(run_mod, "load_milu_hindi_medicine", lambda hf_token=None: _fake_items())
    monkeypatch.setattr(run_mod, "_load_model_fn", lambda name: fake_get_answer)
    monkeypatch.setattr(run_mod, "load_dotenv", lambda *a, **k: None)
    return rec


def test_main_end_to_end_greedy_fewshot(tmp_path, fakes):
    run_mod.main(["--model", "himed", "--condition", "c2", "--use-fewshot", "--fewshot-n", "6",
                  "--output-dir", str(tmp_path)])
    out = tmp_path / "himed_c2_raw.csv"
    assert out.exists()
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 6  # 12 items - 6 pool
    assert all(kw == GREEDY for kw in fakes["kwargs"])
    assert all(p.count("\nAnswer: ") >= 3 and p.split("Here are example questions:")[1].count("Question: ") == 4
               for p in fakes["prompts"])  # 3 exemplars + target question


def test_main_limit_and_shuffle(tmp_path, fakes):
    run_mod.main(["--model", "nanda", "--condition", "x", "--limit", "2", "--shuffle-options",
                  "--fewshot-n", "4", "--output-dir", str(tmp_path)])
    rows = list(csv.DictReader(open(tmp_path / "nanda_x_raw.csv", encoding="utf-8")))
    assert len(rows) == 2


def test_main_rejects_temperature_without_sample_decoding(tmp_path, fakes):
    with pytest.raises(SystemExit):
        run_mod.main(["--model", "himed", "--condition", "c", "--temperature", "0.5",
                      "--output-dir", str(tmp_path)])
