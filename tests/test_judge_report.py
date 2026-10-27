"""Tests for literacy_eval/judge/report.py — validation metrics and LLM tables on synthetic scores."""

import numpy as np
import pandas as pd
import pytest

from literacy_eval.judge import report as R


def _llm(task, model, level, item, score, error=""):
    return {"job_id": f"{task}|{model}|{level}|{item}", "task": task, "model": model, "level": level,
            "item_id": item, "score": score, "judge_error": error, "feedback": "fb", "set": ""}


def _scores(rows):
    df = pd.DataFrame(rows)
    for c in ("set", "case", "label", "rating", "health", "judge_model", "judge_revision", "prompt_version"):
        if c not in df:
            df[c] = ""
    df["score"] = df["score"].astype("Int64")
    return df.fillna({"set": ""})


def test_controlled_band_logic():
    rows = [{"job_id": f"reference_agreement|controlled:{case}|i{k}", "task": "reference_agreement",
             "set": "controlled", "case": case, "score": s, "judge_error": ""}
            for k, (case, s) in enumerate([("positive", 5), ("positive", 4), ("swapped", 3), ("swapped", 1)])]
    t = R.controlled_table(_scores(rows)).set_index(["task", "case"])
    assert t.loc[("reference_agreement", "positive"), "pct_in_band"] == 50.0
    assert t.loc[("reference_agreement", "swapped"), "pct_in_band"] == 50.0
    assert t.loc[("reference_agreement", "positive"), "pass"] == "❌"


def test_split_keeps_paraphrase_and_validation_out_of_llm_rows():
    rows = [_llm("complexity", "himed", "low", "i1", 2),
            {**_llm("complexity", "himed", "low", "i1", 3), "job_id": "para:complexity|himed|low|i1", "set": "paraphrase"},
            {"job_id": "complexity|corpus:val|0", "task": "complexity", "set": "corpus:val", "label": "grade5",
             "score": 1, "judge_error": ""}]
    llm = R.llm_rows(_scores(rows))
    assert llm["job_id"].tolist() == ["complexity|himed|low|i1"]


def test_errors_are_excluded_and_counted():
    rows = [_llm("harm", "himed", "low", "i1", 5), _llm("harm", "himed", "low", "i2", None, "no [RESULT]"),
            _llm("harm", "himed", "low", "i3", None, "missing")]
    s = _scores(rows)
    summary = R.run_summary(s).set_index("task")
    assert summary.loc["harm", "ok"] == 1 and summary.loc["harm", "error"] == 1 and summary.loc["harm", "missing"] == 1
    assert len(R.llm_rows(s)) == 1


def test_match_rate_uses_cutpoints_and_target_levels():
    rows = [_llm("complexity", "himed", "low", "i1", 2), _llm("complexity", "himed", "low", "i2", 3),
            _llm("complexity", "himed", "high", "i1", 5)]
    t = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=(2.5, 3.5), suspect=None).set_index(["model", "level"])
    assert t.loc[("himed", "low"), "complexity_match_pct"] == 50.0      # 2 -> 0 (match), 3 -> 1 (miss)
    assert t.loc[("himed", "high"), "complexity_match_pct"] == 100.0


def test_agreement_columns_and_audit_pending():
    rows = [_llm("reference_agreement", "himed", "low", f"i{k}", s) for k, s in enumerate([5, 4, 1, 3] * 3)]
    t = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect=None).set_index(["model", "level"])
    assert t.loc[("himed", "low"), "agree_pct"] == 50.0 and t.loc[("himed", "low"), "contradict_pct"] == 25.0
    assert t.loc[("himed", "low"), "agree_pct_excl_suspect"] == "audit pending"
    t2 = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect={"i2"}).set_index(["model", "level"])
    assert t2.loc[("himed", "low"), "agree_pct_excl_suspect"] == pytest.approx(54.5, abs=0.1)   # 6 of 11


def test_paired_tests_report_n_too_small():
    rows = [_llm("complexity", "himed", lv, f"i{k}", s) for k in range(5)
            for lv, s in (("low", 1), ("mid", 3), ("high", 5))]
    t = R.paired_tests(R.llm_rows(_scores(rows))).set_index("model")
    assert t.loc["himed", "page_p"] == "n too small"


def test_paired_tests_detect_trend_with_enough_items():
    rng = np.random.default_rng(0)
    rows = [_llm("complexity", "himed", lv, f"i{k}", int(np.clip(base + rng.integers(0, 2), 1, 5)))
            for k in range(20) for lv, base in (("low", 1), ("mid", 2), ("high", 4))]
    t = R.paired_tests(R.llm_rows(_scores(rows))).set_index("model")
    assert t.loc["himed", "page_p"] == "<0.001" and t.loc["himed", "n_all_levels"] == 20


def test_corpus_metrics_from_val_cutpoints():
    rows = []
    for split in ("val", "test"):
        for k, (label, s) in enumerate([("grade5", 1), ("grade5", 2), ("grade10", 3), ("professional", 5)]):
            rows.append({"job_id": f"complexity|corpus:{split}|{k}", "task": "complexity", "set": f"corpus:{split}",
                         "label": label, "score": s, "judge_error": ""})
    s = _scores(rows)
    cuts = R.fit_on_val(s)
    m = R.corpus_metrics(s, cuts)
    assert m["accuracy"] == 1.0 and m["kappa"] == pytest.approx(1.0)


def test_harm_table_lists_low_scores_only():
    rows = [_llm("harm", "nanda", "low", "i1", 1), _llm("harm", "nanda", "low", "i2", 4)]
    t = R.harm_table(R.llm_rows(_scores(rows)))
    assert t["item_id"].tolist() == ["i1"] and "feedback" in t.columns


def test_paraphrase_agreement():
    rows = [_llm("harm", "nanda", "low", "i1", 5), _llm("harm", "nanda", "low", "i2", 5),
            {**_llm("harm", "nanda", "low", "i1", 4), "job_id": "para:harm|nanda|low|i1", "set": "paraphrase"},
            {**_llm("harm", "nanda", "low", "i2", 2), "job_id": "para:harm|nanda|low|i2", "set": "paraphrase"}]
    t = R.paraphrase_table(_scores(rows)).set_index("task")
    assert t.loc["harm", "pct_within_1"] == 50.0 and t.loc["harm", "pct_identical"] == 0.0


def test_build_report_runs_end_to_end_without_bert_or_clean():
    rows = [_llm("complexity", "himed", "low", "i1", 2), _llm("harm", "himed", "low", "i1", 5)]
    text, tables = R.build_report(_scores(rows), clean=None, bert_dir=None, suspect=None)
    assert "## A." in text and "## C." in text and "judge_llm" in tables


def test_small_cells_report_n_too_small_for_rates():
    rows = [_llm("reference_agreement", "airavata", "low", f"i{k}", 5) for k in range(7)] + \
           [_llm("consistency", "airavata", "low", f"i{k}", 4) for k in range(7)]
    t = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect=set()).set_index(["model", "level"])
    for col in ("agree_pct", "contradict_pct", "agree_pct_excl_suspect", "consistency_pct_ge4", "consistency_95ci"):
        assert t.loc[("airavata", "low"), col] == "n too small"


def test_topic_table_uses_pooled_health_set():
    rows = [{"job_id": f"complexity|corpus:health|train:{k}", "task": "complexity", "set": "corpus:health",
             "label": "grade5", "score": 1, "judge_error": ""} for k in range(3)]
    t = R.topic_table(_scores(rows)).set_index("texts")
    assert t.loc["NCERT Class 5, health-related (all splits)", "n"] == 3
