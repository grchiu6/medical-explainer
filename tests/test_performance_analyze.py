import csv

import pytest

from performance_eval.analyze import (
    accuracy_summary,
    build_ladder_rows,
    compare_conditions,
    main,
    n_options_mode,
    write_results_md,
)


def test_accuracy_summary_counts_correct_and_none_rate():
    rows = [
        {"item_id": "a", "is_correct": True, "parsed_letter": "A"},
        {"item_id": "b", "is_correct": False, "parsed_letter": None},
        {"item_id": "c", "is_correct": False, "parsed_letter": "B"},
    ]
    s = accuracy_summary(rows)
    assert s["n"] == 3
    assert s["n_correct"] == 1
    assert abs(s["accuracy"] - 1 / 3) < 1e-9
    assert abs(s["none_rate"] - 1 / 3) < 1e-9
    assert s["ci_low"] <= s["accuracy"] <= s["ci_high"]


def test_compare_conditions_aligns_by_item_id_and_counts_flips():
    prior = [
        {"item_id": "a", "is_correct": True},
        {"item_id": "b", "is_correct": False},
        {"item_id": "c", "is_correct": False},
    ]
    new = [
        {"item_id": "c", "is_correct": False},  # order differs on purpose
        {"item_id": "b", "is_correct": True},
        {"item_id": "a", "is_correct": True},
    ]
    c = compare_conditions(prior, new)
    assert c["n_paired"] == 3
    assert c["only_new_correct"] == 1
    assert c["only_prior_correct"] == 0
    assert c["accuracy_delta"] > 0
    assert 0.0 <= c["mcnemar_p"] <= 1.0


def test_compare_conditions_ignores_items_missing_from_either_side():
    prior = [{"item_id": "a", "is_correct": True}, {"item_id": "b", "is_correct": False}]
    new = [{"item_id": "a", "is_correct": True}]
    c = compare_conditions(prior, new)
    assert c["n_paired"] == 1


def test_write_results_md_includes_chance_baseline_and_holm_column(tmp_path):
    rows = [{"model": "himed", "condition": "c0_baseline", "n": 100, "accuracy": 0.3,
             "ci_low": 0.21, "ci_high": 0.40, "none_rate": 0.05, "n_options_mode": 4,
             "accuracy_delta": None, "mcnemar_p": None, "mcnemar_p_holm": None}]
    out = tmp_path / "results.md"
    write_results_md(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "chance" in text.lower()
    assert "25.0%" in text
    assert "30.0%" in text and "21.0%" in text and "40.0%" in text
    assert "—" in text
    assert "row above" in text


def test_build_ladder_rows_holm_corrects_across_the_ladder():
    ladder = [
        {"condition": "c0_baseline", "summary": {"n": 100, "n_correct": 30, "accuracy": 0.30,
          "ci_low": 0.22, "ci_high": 0.40, "none_rate": 0.05}, "comparison": None},
        {"condition": "c2_cot", "summary": {"n": 100, "n_correct": 38, "accuracy": 0.38,
          "ci_low": 0.29, "ci_high": 0.48, "none_rate": 0.02},
          "comparison": {"n_paired": 100, "accuracy_delta": 0.08, "mcnemar_p": 0.03,
                          "only_prior_correct": 2, "only_new_correct": 10}},
        {"condition": "c3_fewshot", "summary": {"n": 100, "n_correct": 40, "accuracy": 0.40,
          "ci_low": 0.31, "ci_high": 0.50, "none_rate": 0.01},
          "comparison": {"n_paired": 100, "accuracy_delta": 0.02, "mcnemar_p": 0.04,
                          "only_prior_correct": 3, "only_new_correct": 5}},
    ]
    rows = build_ladder_rows("himed", ladder, n_options_mode=4)
    assert rows[0]["mcnemar_p"] is None and rows[0]["mcnemar_p_holm"] is None
    assert rows[1]["mcnemar_p_holm"] == rows[1]["mcnemar_p"] * 2
    assert rows[2]["mcnemar_p_holm"] >= rows[2]["mcnemar_p"]
    assert all(r["model"] == "himed" for r in rows)
    assert [r["condition"] for r in rows] == ["c0_baseline", "c2_cot", "c3_fewshot"]


# ---- CLI ----

Q = "प्रश्न?\nA. एक\nB. दो\nC. तीन\nD. चार"
REFS = ["A", "B", "C", "D"]


def _write_raw(path, answers):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "question_with_options",
                                          "reference_answer", "model_answer"])
        w.writeheader()
        for i, (ref, ans) in enumerate(zip(REFS, answers)):
            w.writerow({"item_id": f"q{i}", "question_with_options": Q,
                        "reference_answer": ref, "model_answer": ans})


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_n_options_mode_counts_most_common_option_count():
    rows = [{"question_with_options": Q}, {"question_with_options": Q},
            {"question_with_options": "q\nA. x\nB. y"}]
    assert n_options_mode(rows) == 4


def test_main_end_to_end_scores_and_writes_results(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    _write_raw(raw / "m1_c0_baseline_raw.csv", ["A", "B", "A", "D"])  # 3/4 correct
    _write_raw(raw / "m1_c2_cot_raw.csv",
               ["तर्क...\nAnswer: A", "Answer: B", "Answer: C", "Answer: D"])  # 4/4

    main(["--model", "m1", "--raw-dir", str(raw), "--out-dir", str(out),
          "--ladder", "c0_baseline:first_letter", "c2_cot:after_marker"])

    s0 = _read(raw / "m1_c0_baseline_scored.csv")
    s1 = _read(raw / "m1_c2_cot_scored.csv")
    assert [r["is_correct"] for r in s0] == ["1", "1", "0", "1"]
    assert [r["is_correct"] for r in s1] == ["1", "1", "1", "1"]
    assert [r["parsed_letter"] for r in s1] == ["A", "B", "C", "D"]
    assert "model_answer" in s0[0]

    md = (out / "results_m1.md").read_text(encoding="utf-8")
    assert "c0_baseline" in md and "c2_cot" in md
    assert "chance" in md.lower() and "25.0%" in md
    print(md)


def test_main_missing_raw_csv_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
              "--ladder", "c0_baseline:first_letter"])


def test_main_bad_mode_exits(tmp_path):
    _write_raw(tmp_path / "m1_c0_raw.csv", ["A", "B", "C", "D"])
    with pytest.raises(SystemExit):
        main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
              "--ladder", "c0:bogus"])
