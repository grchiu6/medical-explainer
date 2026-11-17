import csv

import pytest

from performance_eval import run as run_mod
from performance_eval.run import build_gen_kwargs, run_condition, validate_args

GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "repetition_penalty": 1.0}


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

    def fake_build_prompt(question_with_options, n_options, use_cot=False, exemplars=None, passages=None):
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
    monkeypatch.setattr(run_mod, "_set_seed", lambda seed: rec.setdefault("seeds", []).append(seed))
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


def _real_items(n=12):
    """Items with distinct option texts and a qwo consistent with data_milu."""
    from performance_eval.data_milu import _format_qwo

    items = []
    for k in range(n):
        opts = {c: f"opt{k}{c}" for c in "ABCD"}
        items.append({"id": f"r{k}", "question": f"stem{k}", "options": opts,
                      "question_with_options": _format_qwo(f"stem{k}", opts),
                      "answer_letter": "ABCD"[k % 4], "n_options": 4})
    return items


def _option_text(qwo, letter):
    for line in qwo.split("\n"):
        if line.startswith(f"{letter}. "):
            return line[3:]
    raise AssertionError("letter not found")


def test_main_shuffle_keeps_reference_consistent_and_actually_shuffles(tmp_path, fakes, monkeypatch):
    items = _real_items()
    monkeypatch.setattr(run_mod, "load_milu_hindi_medicine", lambda hf_token=None: items)
    run_mod.main(["--model", "nanda", "--condition", "x", "--shuffle-options",
                  "--fewshot-n", "4", "--output-dir", str(tmp_path)])
    rows = list(csv.DictReader(open(tmp_path / "nanda_x_raw.csv", encoding="utf-8")))
    by_id = {it["id"]: it for it in items}
    assert len(rows) == 8
    changed = 0
    for r in rows:
        orig = by_id[r["item_id"]]
        original_text = orig["options"][orig["answer_letter"]]
        assert _option_text(r["question_with_options"], r["reference_answer"]) == original_text
        if r["question_with_options"] != orig["question_with_options"]:
            changed += 1
    assert changed >= 1


def test_main_limit_slices_after_split(tmp_path, fakes, monkeypatch):
    from performance_eval.data_milu import split_test_fewshot

    items = _real_items()
    monkeypatch.setattr(run_mod, "load_milu_hindi_medicine", lambda hf_token=None: items)
    test, pool = split_test_fewshot(items, fewshot_n=6, seed=42)
    run_mod.main(["--model", "himed", "--condition", "l", "--limit", "2", "--use-fewshot",
                  "--fewshot-n", "6", "--output-dir", str(tmp_path)])
    rows = list(csv.DictReader(open(tmp_path / "himed_l_raw.csv", encoding="utf-8")))
    assert [r["item_id"] for r in rows] == [t["id"] for t in test[:2]]
    pool_stems = {it["question_with_options"] for it in pool}
    for p in fakes["prompts"]:
        block = p.split("Here are example questions:")[1].split("You are answering")[0]
        exemplars = [b.split("\nAnswer: ")[0].removeprefix("Question: ").strip()
                     for b in block.strip().split("\n\n") if b.startswith("Question: ")]
        assert len(exemplars) == 3
        assert all(e in pool_stems for e in exemplars)


def test_run_condition_recovers_from_empty_existing_file(tmp_path):
    out = tmp_path / "out.csv"
    out.write_text("", encoding="utf-8")
    items = [{"id": f"m{k}", "question_with_options": f"Q{k}", "answer_letter": "A", "n_options": 2}
             for k in range(2)]
    fn = lambda prompt, gen_kwargs=None: "Answer: A"
    run_condition(items, [], get_answer_fn=fn, use_cot=False, use_fewshot=False, gen_kwargs={}, output_path=out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["item_id"] for r in rows] == ["m0", "m1"]
    calls = []
    run_condition(items, [], get_answer_fn=lambda p, gen_kwargs=None: calls.append(p) or "x",
                  use_cot=False, use_fewshot=False, gen_kwargs={}, output_path=out)
    assert calls == []
    assert len(list(csv.DictReader(open(out, encoding="utf-8")))) == 2


def test_main_rejects_temperature_without_sample_decoding(tmp_path, fakes):
    with pytest.raises(SystemExit):
        run_mod.main(["--model", "himed", "--condition", "c", "--temperature", "0.5",
                      "--output-dir", str(tmp_path)])


# ---- I4 / R22: greedy neutralizes repetition penalty ----

def test_build_gen_kwargs_greedy_sets_repetition_penalty_1_and_override_wins():
    assert build_gen_kwargs(False)["repetition_penalty"] == 1.0
    assert build_gen_kwargs(False, repetition_penalty=1.3)["repetition_penalty"] == 1.3
    assert "repetition_penalty" not in build_gen_kwargs(True)


# ---- I1 / R19: --gen-seed is separate from the data --seed ----

def test_gen_seed_feeds_set_seed_only(tmp_path, fakes):
    run_mod.main(["--model", "himed", "--condition", "a", "--output-dir", str(tmp_path)])
    run_mod.main(["--model", "himed", "--condition", "b", "--gen-seed", "7", "--seed", "42",
                  "--output-dir", str(tmp_path)])
    assert fakes["seeds"] == [42, 7]


def test_gen_seed_leaves_test_items_and_exemplars_unchanged(tmp_path, fakes, monkeypatch):
    items = _real_items()
    monkeypatch.setattr(run_mod, "load_milu_hindi_medicine", lambda hf_token=None: items)
    common = ["--model", "himed", "--use-fewshot", "--shuffle-options", "--fewshot-n", "6",
              "--output-dir", str(tmp_path)]
    run_mod.main(common + ["--condition", "g1", "--gen-seed", "1"])
    prompts_1 = list(fakes["prompts"])
    fakes["prompts"].clear()
    run_mod.main(common + ["--condition", "g2", "--gen-seed", "2"])
    assert fakes["prompts"] == prompts_1
    rows_1 = list(csv.DictReader(open(tmp_path / "himed_g1_raw.csv", encoding="utf-8")))
    rows_2 = list(csv.DictReader(open(tmp_path / "himed_g2_raw.csv", encoding="utf-8")))
    strip = lambda rows: [(r["item_id"], r["question_with_options"], r["reference_answer"]) for r in rows]
    assert strip(rows_1) == strip(rows_2)


# ---- I3 / R21: config sidecar guards resume ----

def test_main_writes_config_sidecar(tmp_path, fakes):
    import json

    run_mod.main(["--model", "himed", "--condition", "c", "--use-fewshot", "--fewshot-n", "6",
                  "--max-new-tokens", "64", "--output-dir", str(tmp_path)])
    cfg = json.loads((tmp_path / "himed_c_raw.csv.config.json").read_text(encoding="utf-8"))
    assert cfg["model"] == "himed" and cfg["condition"] == "c"
    assert cfg["use_cot"] is False and cfg["use_fewshot"] is True and cfg["fewshot_n"] == 6
    assert cfg["gen_kwargs"] == {**GREEDY, "max_new_tokens": 64}
    assert cfg["seed"] == 42 and cfg["gen_seed"] == 42
    assert cfg["shuffle_options"] is False and cfg["limit"] is None
    assert len(cfg["exemplar_ids"]) == 3


def test_main_sidecar_exemplar_ids_empty_without_fewshot(tmp_path, fakes):
    import json

    run_mod.main(["--model", "himed", "--condition", "c", "--output-dir", str(tmp_path)])
    cfg = json.loads((tmp_path / "himed_c_raw.csv.config.json").read_text(encoding="utf-8"))
    assert cfg["exemplar_ids"] == []


def test_main_resume_with_same_config_proceeds(tmp_path, fakes):
    args = ["--model", "himed", "--condition", "c", "--output-dir", str(tmp_path)]
    run_mod.main(args + ["--limit", "2"])
    run_mod.main(args)  # limit is exempt; everything else matches
    rows = list(csv.DictReader(open(tmp_path / "himed_c_raw.csv", encoding="utf-8")))
    assert len(rows) == 2  # 12 items - 10 pool


def test_main_resume_with_different_max_new_tokens_refuses(tmp_path, fakes):
    args = ["--model", "himed", "--condition", "c", "--output-dir", str(tmp_path)]
    run_mod.main(args + ["--max-new-tokens", "64", "--limit", "1"])
    n_calls = len(fakes["prompts"])
    with pytest.raises(SystemExit, match="gen_kwargs") as exc:
        run_mod.main(args + ["--max-new-tokens", "128"])
    assert "new --condition label" in str(exc.value)
    assert len(fakes["prompts"]) == n_calls


def test_main_resume_csv_with_rows_but_no_sidecar_refuses(tmp_path, fakes):
    (tmp_path / "himed_c_raw.csv").write_text(
        "item_id,question_with_options,reference_answer,model_answer\ni0,Q,A,A\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="config.json"):
        run_mod.main(["--model", "himed", "--condition", "c", "--output-dir", str(tmp_path)])
    assert fakes["prompts"] == []


# ---- passages (RAG) ----
import hashlib
import json

from performance_eval.run import check_or_write_config, load_passages

_C0_SIDECAR = json.loads("""{
  "model": "himed", "condition": "c0_baseline", "use_cot": false, "use_fewshot": false,
  "fewshot_n": 10,
  "gen_kwargs": {"do_sample": false, "temperature": null, "top_p": null, "repetition_penalty": 1.0},
  "seed": 42, "gen_seed": 42, "shuffle_options": false, "limit": null, "exemplar_ids": []
}""")


def _write_passages(tmp_path, ids, k=2, meta=True, extra_lines=()):
    p = tmp_path / "passages_rag_k5.jsonl"
    lines = [json.dumps({"item_id": i, "condition": "rag_k5", "passages": [
        {"rank": r + 1, "passage_id": f"s:{i}{r}", "source": "s", "title": f"T{r}", "content": f"c{r}", "score": 1.0}
        for r in range(k)]}) for i in ids] + list(extra_lines)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if meta:
        p.with_suffix(".meta.json").write_text(json.dumps({"condition": "rag_k5", "k": k, "seed": 42,
            "retriever": "bge-m3", "index_meta": {}, "query_format": "q"}), encoding="utf-8")
    return p


def test_load_passages_returns_rank_ordered_dict(tmp_path):
    p = _write_passages(tmp_path, ["a", "b"])
    got = load_passages(p, ["a", "b"])
    assert set(got) == {"a", "b"} and [x["rank"] for x in got["a"]] == [1, 2]


def test_load_passages_refuses_missing_extra_or_wrong_k(tmp_path):
    p = _write_passages(tmp_path, ["a", "b"])
    with pytest.raises(SystemExit):
        load_passages(p, ["a", "b", "c"])  # missing item
    with pytest.raises(SystemExit):
        load_passages(p, ["a"])  # extra id in file
    p2 = _write_passages(tmp_path, ["a"], k=2)
    meta = p2.with_suffix(".meta.json")
    meta.write_text(json.dumps({"k": 5}), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_passages(p2, ["a"])  # wrong k


def test_load_passages_refuses_duplicates_and_missing_meta(tmp_path):
    dup = _write_passages(tmp_path, ["a"])
    dup.write_text(dup.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(SystemExit):
        load_passages(dup, ["a"])
    (tmp_path / "sub").mkdir()
    nometa = _write_passages(tmp_path / "sub", ["a"], meta=False)
    with pytest.raises(SystemExit):
        load_passages(nometa, ["a"])


def test_old_sidecar_without_passage_keys_still_resumes(tmp_path):
    out = tmp_path / "himed_c0_baseline_raw.csv"
    out.write_text("item_id,question_with_options,reference_answer,model_answer\nm1,Q,A,A\n", encoding="utf-8")
    run_mod.config_sidecar_path(out).write_text(json.dumps(_C0_SIDECAR), encoding="utf-8")
    new = dict(_C0_SIDECAR, passages_file=None, passages_sha256=None, passages_k=None)
    check_or_write_config(out, new)  # no raise


def test_resume_refused_when_passages_file_changed(tmp_path):
    out = tmp_path / "o.csv"
    out.write_text("item_id,question_with_options,reference_answer,model_answer\nm1,Q,A,A\n", encoding="utf-8")
    base = dict(_C0_SIDECAR, passages_file="p.jsonl", passages_sha256="aaa", passages_k=5)
    run_mod.config_sidecar_path(out).write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(SystemExit):
        check_or_write_config(out, dict(base, passages_sha256="bbb"))


def test_run_condition_puts_each_items_passages_in_its_prompt(tmp_path):
    items = [{"id": f"m{i}", "question_with_options": f"Q{i}?", "answer_letter": "A", "n_options": 2} for i in (1, 2)]
    passages = {"m1": [{"title": "TitleOne", "content": "body1"}], "m2": [{"title": "TitleTwo", "content": "body2"}]}
    calls = []
    run_condition(items, [], lambda prompt, gen_kwargs=None: calls.append(prompt) or "A", False, False,
                  {}, tmp_path / "o.csv", passages=passages)
    assert "TitleOne" in calls[0] and "TitleTwo" not in calls[0]
    assert "TitleTwo" in calls[1] and calls[1].startswith("Reference passages")


def test_load_passages_allow_extra_returns_only_requested_ids(tmp_path):
    p = _write_passages(tmp_path, ["a", "b", "c"])
    got = load_passages(p, ["a", "b"], allow_extra=True)
    assert set(got) == {"a", "b"}


def test_load_passages_allow_extra_still_refuses_missing_requested_id(tmp_path):
    p = _write_passages(tmp_path, ["a", "b"])
    with pytest.raises(SystemExit):
        load_passages(p, ["a", "z"], allow_extra=True)
