"""Tests for literacy_eval/judge/report.py — validation metrics and LLM tables on synthetic scores."""

import numpy as np
import pandas as pd
import pytest

from literacy_eval.judge import report as R


def _llm(task, model, level, item, score, error="", failure_type="ok"):
    return {"job_id": f"{task}|{model}|{level}|{item}", "task": task, "model": model, "level": level,
            "item_id": item, "score": score, "judge_error": error, "feedback": "fb", "set": "",
            "failure_type": failure_type}


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
    rows = [_llm("term_explanation", "himed", "low", "i1", 5), _llm("term_explanation", "himed", "low", "i2", None, "no [RESULT]"),
            _llm("term_explanation", "himed", "low", "i3", None, "missing")]
    s = _scores(rows)
    summary = R.run_summary(s).set_index("task")
    assert summary.loc["term_explanation", "ok"] == 1 and summary.loc["term_explanation", "error"] == 1 and summary.loc["term_explanation", "missing"] == 1
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
    # audit pending: primary metrics use all items, and the robustness column says so
    assert t.loc[("himed", "low"), "agree_pct"] == 50.0 and t.loc[("himed", "low"), "contradict_pct"] == 25.0
    assert t.loc[("himed", "low"), "agree_pct_incl_invalid"] == "audit pending"
    # invalid item i2 (scored 1) leaves the PRIMARY numbers; the all-items figure moves to the robustness column
    t2 = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect={"i2"}).set_index(["model", "level"])
    assert t2.loc[("himed", "low"), "agree_pct"] == pytest.approx(54.5, abs=0.1)        # 6 of 11
    assert t2.loc[("himed", "low"), "contradict_pct"] == pytest.approx(18.2, abs=0.1)   # 2 of 11
    assert t2.loc[("himed", "low"), "agree_pct_incl_invalid"] == 50.0
    assert t2.loc[("himed", "low"), "n"] == 11


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



def test_paraphrase_agreement():
    rows = [_llm("term_explanation", "nanda", "low", "i1", 5), _llm("term_explanation", "nanda", "low", "i2", 5),
            {**_llm("term_explanation", "nanda", "low", "i1", 4), "job_id": "para:term_explanation|nanda|low|i1", "set": "paraphrase"},
            {**_llm("term_explanation", "nanda", "low", "i2", 2), "job_id": "para:term_explanation|nanda|low|i2", "set": "paraphrase"}]
    t = R.paraphrase_table(_scores(rows)).set_index("task")
    assert t.loc["term_explanation", "pct_within_1"] == 50.0 and t.loc["term_explanation", "pct_identical"] == 0.0


def test_build_report_runs_end_to_end_without_bert_or_clean():
    rows = [_llm("complexity", "himed", "low", "i1", 2), _llm("term_explanation", "himed", "low", "i1", 5)]
    text, tables = R.build_report(_scores(rows), clean=None, bert_dir=None, suspect=None)
    assert "## A." in text and "## C." in text and "judge_llm" in tables


def test_small_cells_report_n_too_small_for_rates():
    rows = [_llm("reference_agreement", "airavata", "low", f"i{k}", 5) for k in range(7)] + \
           [_llm("consistency", "airavata", "low", f"i{k}", 4) for k in range(7)]
    t = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect=set()).set_index(["model", "level"])
    for col in ("agree_pct", "contradict_pct", "agree_pct_incl_invalid", "consistency_pct_ge4", "consistency_95ci"):
        assert t.loc[("airavata", "low"), col] == "n too small"


def test_topic_table_uses_pooled_health_set():
    rows = [{"job_id": f"complexity|corpus:health|train:{k}", "task": "complexity", "set": "corpus:health",
             "label": "grade5", "score": 1, "judge_error": ""} for k in range(3)]
    t = R.topic_table(_scores(rows)).set_index("texts")
    assert t.loc["NCERT Class 5, health-related (all splits)", "n"] == 3


def test_build_report_on_a_partial_run_without_corpus_columns():
    # pilot: only LLM + controlled jobs, so collect's CSV has no label/rating/health/n_words columns
    rows = [_llm("complexity", "himed", "low", "i1", 2),
            {"job_id": "reference_agreement|controlled:positive|i1", "task": "reference_agreement",
             "set": "controlled", "case": "positive", "score": 5, "judge_error": ""}]
    df = pd.DataFrame(rows)
    df["score"] = df["score"].astype("Int64")
    df["set"] = df["set"].fillna("")
    text, _ = R.build_report(df, clean=None, bert_dir=None, suspect=None)
    assert "### B6. Controlled cases" in text


def test_report_has_no_harm_section_or_column():
    rows = [_llm("reference_agreement", "himed", "low", f"i{k}", 5) for k in range(10)]
    text, tables = R.build_report(_scores(rows), clean=None, bert_dir=None, suspect=None)
    assert "harm" not in text.lower() and "judge_harm" not in tables
    assert not any("harm" in c for c in tables["judge_llm"].columns)


def test_invalid_items_leave_every_primary_metric():
    rows = [_llm("complexity", "himed", "low", f"i{k}", 1) for k in range(10)] + \
           [_llm("complexity", "himed", "low", "bad", 5)]
    t = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect={"bad"}).set_index(["model", "level"])
    assert t.loc[("himed", "low"), "complexity_mean"] == 1.0 and t.loc[("himed", "low"), "n"] == 10
    paired = R.paired_tests(R.exclude_invalid(R.llm_rows(_scores(rows)), {"bad"}))
    assert "bad" not in set(R.exclude_invalid(R.llm_rows(_scores(rows)), {"bad"})["item_id"])


def test_generation_failures_are_counted_per_cell():
    rows = [_llm("complexity", "himed", "low", "i1", 2)]
    clean = pd.DataFrame([{"model": "himed", "level": "low", "item_id": "i1", "answer": "उत्तर"},
                          {"model": "himed", "level": "low", "item_id": "i2", "answer": ""},
                          {"model": "himed", "level": "low", "item_id": "i3", "answer": None}])
    t = R.llm_main_table(R.llm_rows(_scores(rows)), cuts=None, suspect=None, clean=clean).set_index(["model", "level"])
    assert t.loc[("himed", "low"), "n_generation_failures"] == 2


def test_scores_break_down_by_failure_type():
    rows = [_llm("reference_agreement", "apollo", "low", "i1", 1, failure_type="degenerate"),
            _llm("reference_agreement", "apollo", "low", "i2", 3, failure_type="degenerate"),
            _llm("reference_agreement", "apollo", "low", "i3", 5)]
    t = R.by_failure_type(R.llm_rows(_scores(rows))).set_index(["task", "failure_type"])
    assert t.loc[("reference_agreement", "degenerate"), "n"] == 2
    assert t.loc[("reference_agreement", "degenerate"), "mean_score"] == 2.0
    assert t.loc[("reference_agreement", "ok (rule-usable)"), "mean_score"] == 5.0


def test_report_states_what_is_in_and_out_of_the_evaluation():
    rows = [_llm("complexity", "himed", "low", "i1", 2, failure_type="echo")]
    clean = pd.DataFrame([{"model": "himed", "level": "low", "item_id": "i1", "answer": "x"}])
    text, tables = R.build_report(_scores(rows), clean=clean, bert_dir=None, suspect={"i9"})
    assert "Excluded dataset items" in text and "i9" in text
    assert "Generation failures" in text and "judge_by_failure_type" in tables
