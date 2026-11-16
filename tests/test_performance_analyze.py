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


def test_write_results_md_includes_chance_baseline_and_bonferroni_column(tmp_path):
    rows = [{"model": "himed", "condition": "c0_baseline", "n": 100, "accuracy": 0.3,
             "ci_low": 0.21, "ci_high": 0.40, "none_rate": 0.05, "no_marker_rate": None,
             "n_options_mode": 4, "accuracy_delta": None, "n_paired": None,
             "mcnemar_p": None, "mcnemar_p_adj": None, "kept": None, "family_size": 3}]
    out = tmp_path / "results.md"
    write_results_md(rows, out, majority_baseline=("C", 0.288))
    text = out.read_text(encoding="utf-8")
    assert "chance" in text.lower()
    assert "25.0%" in text
    assert "30.0%" in text and "21.0%" in text and "40.0%" in text
    assert "—" in text
    assert "row above" in text
    assert "newly correct" in text
    assert "p (Bonferroni, m=3)" in text
    assert "kept?" in text and "n_paired" in text
    assert "pp" in text
    assert "always answering C would score 28.8%" in text


def test_build_ladder_rows_bonferroni_and_kept():
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
    rows = build_ladder_rows("himed", ladder, n_options_mode=4, family_size=3)
    assert rows[0]["mcnemar_p"] is None and rows[0]["mcnemar_p_adj"] is None
    assert rows[1]["mcnemar_p_adj"] == pytest.approx(0.09)  # 0.03 x 3
    assert rows[2]["mcnemar_p_adj"] == pytest.approx(0.12)  # 0.04 x 3
    assert all(r["family_size"] == 3 for r in rows)
    assert rows[0]["kept"] is None
    assert all(r["model"] == "himed" for r in rows)
    assert rows[0]["n_paired"] is None
    assert rows[0]["only_prior_correct"] is None and rows[0]["only_new_correct"] is None
    assert (rows[1]["n_paired"], rows[1]["only_prior_correct"], rows[1]["only_new_correct"]) == (100, 2, 10)
    assert (rows[2]["only_prior_correct"], rows[2]["only_new_correct"]) == (3, 5)
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
    row0 = next(l for l in md.splitlines() if "| c0_baseline |" in l)
    row1 = next(l for l in md.splitlines() if "| c2_cot |" in l)
    cells0 = [c.strip() for c in row0.strip("|").split("|")]
    cells1 = [c.strip() for c in row1.strip("|").split("|")]
    # columns: model, condition, n, n_paired, acc, none_rate, no_marker_rate, delta, flips,
    #          p, p_adj, kept?, chance
    assert cells0[3] == "—" and cells1[3] == "4"
    assert cells0[6] == "—" and cells1[6] == "0.0%"  # no_marker_rate: after_marker only
    assert cells0[7] == "—" and cells0[8] == "—" and cells0[9] == "—" and cells0[11] == "—"
    assert cells1[7] == "+25.0 pp"
    assert cells1[8] == "+1 / −0"
    assert cells1[9] == "1.0000" and cells1[10] == "1.0000"
    assert cells1[11] == "no"
    assert "always answering A would score 25.0%" in md  # all refs tie; first-seen wins


def test_main_missing_raw_csv_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
              "--ladder", "c0_baseline:first_letter"])


def test_main_bad_mode_exits(tmp_path):
    _write_raw(tmp_path / "m1_c0_raw.csv", ["A", "B", "C", "D"])
    with pytest.raises(SystemExit):
        main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
              "--ladder", "c0:bogus"])


# ---- I6 / R24: no_marker_rate ----

def test_accuracy_summary_no_marker_rate_for_after_marker_only():
    rows = [
        {"item_id": "a", "is_correct": True, "parsed_letter": "A", "marker_found": True},
        {"item_id": "b", "is_correct": False, "parsed_letter": "B", "marker_found": False},
        {"item_id": "c", "is_correct": False, "parsed_letter": None, "marker_found": False},
        {"item_id": "d", "is_correct": True, "parsed_letter": "D", "marker_found": True},
    ]
    assert accuracy_summary(rows, mode="after_marker")["no_marker_rate"] == 0.5
    assert accuracy_summary(rows, mode="first_letter")["no_marker_rate"] is None
    assert accuracy_summary(rows)["no_marker_rate"] is None


def test_scored_csv_includes_marker_found_as_1_0(tmp_path):
    from performance_eval.analyze import score_raw_csv

    raw = tmp_path / "m1_c2_raw.csv"
    _write_raw(raw, ["Answer: A", "विकल्प B", "Answer: C", "x"])
    score_raw_csv(raw, "after_marker", tmp_path / "scored.csv")
    assert [r["marker_found"] for r in _read(tmp_path / "scored.csv")] == ["1", "0", "1", "0"]


# ---- I5 / R23: Bonferroni and kept? ----

def _cmp_entry(delta, p):
    return {"condition": "cx", "summary": {"n": 10, "n_correct": 5, "accuracy": 0.5, "ci_low": 0.2,
            "ci_high": 0.8, "none_rate": 0.0},
            "comparison": {"n_paired": 10, "accuracy_delta": delta, "mcnemar_p": p,
                           "only_prior_correct": 0, "only_new_correct": 1}}


def test_bonferroni_caps_at_one_and_uses_family_size():
    first = {"condition": "c0", "summary": {"n": 10, "n_correct": 5, "accuracy": 0.5, "ci_low": 0.2,
             "ci_high": 0.8, "none_rate": 0.0}, "comparison": None}
    rows = build_ladder_rows("m", [first, _cmp_entry(0.1, 0.4)], n_options_mode=4, family_size=3)
    assert rows[1]["mcnemar_p_adj"] == 1.0
    rows = build_ladder_rows("m", [first, _cmp_entry(0.1, 0.01)], n_options_mode=4, family_size=5)
    assert rows[1]["mcnemar_p_adj"] == pytest.approx(0.05)


@pytest.mark.parametrize("delta,p,expected", [
    (0.08, 0.01, "yes"),   # 0.03 < 0.05, delta > 0
    (0.0, 0.001, "no"),    # delta not > 0
    (-0.05, 0.001, "no"),  # delta negative
    (0.08, 0.02, "no"),    # 0.06 >= 0.05 after Bonferroni
])
def test_kept_logic(delta, p, expected):
    first = {"condition": "c0", "summary": {"n": 10, "n_correct": 5, "accuracy": 0.5, "ci_low": 0.2,
             "ci_high": 0.8, "none_rate": 0.0}, "comparison": None}
    rows = build_ladder_rows("m", [first, _cmp_entry(delta, p)], n_options_mode=4, family_size=3)
    assert rows[1]["kept"] == expected


def test_main_family_size_flag_sets_header(tmp_path):
    _write_raw(tmp_path / "m1_c0_raw.csv", ["A", "B", "C", "D"])
    _write_raw(tmp_path / "m1_c2_raw.csv", ["Answer: A", "Answer: B", "Answer: C", "Answer: D"])
    main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
          "--family-size", "5", "--ladder", "c0:first_letter", "c2:after_marker"])
    assert "p (Bonferroni, m=5)" in (tmp_path / "results_m1.md").read_text(encoding="utf-8")


# ---- I2 / R20: identical id sets, no duplicates, n warning ----

def _write_raw_ids(path, ids, answers=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "question_with_options",
                                          "reference_answer", "model_answer"])
        w.writeheader()
        for i, item_id in enumerate(ids):
            w.writerow({"item_id": item_id, "question_with_options": Q,
                        "reference_answer": "A", "model_answer": (answers or ["A"] * len(ids))[i]})


def test_main_mismatched_id_sets_exit(tmp_path):
    _write_raw_ids(tmp_path / "m1_c0_raw.csv", ["q0", "q1", "q2"])
    _write_raw_ids(tmp_path / "m1_c2_raw.csv", ["q0", "q1"])
    with pytest.raises(SystemExit) as exc:
        main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
              "--ladder", "c0:first_letter", "c2:after_marker"])
    msg = str(exc.value)
    assert "c0" in msg and "c2" in msg and "3" in msg and "2" in msg


def test_main_duplicate_ids_exit(tmp_path):
    _write_raw_ids(tmp_path / "m1_c0_raw.csv", ["q0", "q1", "q1"])
    with pytest.raises(SystemExit, match="duplicate"):
        main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
              "--ladder", "c0:first_letter"])


def test_main_warns_when_n_differs_from_expected(tmp_path, capsys):
    from performance_eval.analyze import EXPECTED_N

    assert EXPECTED_N == 489
    _write_raw(tmp_path / "m1_c0_raw.csv", ["A", "B", "C", "D"])
    main(["--model", "m1", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path),
          "--ladder", "c0:first_letter"])
    captured = capsys.readouterr()
    assert "489" in captured.out + captured.err
