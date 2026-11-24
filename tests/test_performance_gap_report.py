import csv

import pytest

from performance_eval import gap_report as gr


def _cell(n, n_correct, first_correct=0):
    """n rows; items first_correct .. first_correct+n_correct-1 are correct (so cells nest)."""
    return [{"item_id": f"i{k:03d}", "is_correct": first_correct <= k < first_correct + n_correct,
             "parsed_letter": "A"} for k in range(n)]


def _cells(n=200, hi=120, en=170, q_hi_o_en=160, q_en_o_hi=122, cot_hi=125, cot_en=126):
    return {"hi": _cell(n, hi), "en": _cell(n, en), "q_hi_o_en": _cell(n, q_hi_o_en),
            "q_en_o_hi": _cell(n, q_en_o_hi), "cot_hi": _cell(n, cot_hi), "cot_en": _cell(n, cot_en)}


def test_latin_share():
    assert gr.reasoning_latin_share("## Thinking\nThe liver stores glycogen.\n## Response\nB.") == 1.0
    assert gr.reasoning_latin_share("यकृत ग्लाइकोजन\nAnswer: B") == 0.0
    assert gr.reasoning_latin_share("abc अआ\nAnswer: A") == pytest.approx(0.6)
    assert gr.reasoning_latin_share("Answer: B") is None
    assert gr.reasoning_latin_share("") is None


def test_latin_share_uses_last_answer_marker():
    assert gr.reasoning_latin_share("abc Answer: A then यकृत\nAnswer: B") < 1.0


def test_recovered_share():
    cells = {"hi": _cell(100, 60), "en": _cell(100, 80), "q_hi_o_en": _cell(100, 75)}
    comps = gr.run_comparisons(cells)
    assert comps[0]["recovered"] == pytest.approx(0.75)
    assert comps[1]["skipped"] and comps[2]["skipped"]


def test_rule_options_fire():
    cells = _cells()
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=0.95)
    assert "Hindi option terms" in msg
    assert "Hindi question stem" not in msg and "reasoning language" not in msg


def test_rule_none_fires():
    cells = _cells(q_hi_o_en=121, q_en_o_hi=122)
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=0.95)
    assert "only the fully English item recovers it" in msg


def test_rule_cot_fires_when_compliant():
    cells = _cells(q_hi_o_en=121, cot_hi=100, cot_en=160)
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=0.9)
    assert "reasoning language" in msg and "only the fully English" not in msg


def test_cot_noncompliant_inconclusive():
    cells = _cells(q_hi_o_en=121, cot_hi=100, cot_en=160)
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=0.5)
    assert "inconclusive" in msg and "reasoning language" not in msg
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=None)
    assert "inconclusive" in msg and "reasoning language" not in msg


def test_r_below_half_does_not_fire():
    # significant, positive, but recovers only 40% of a 50-item gap
    cells = _cells(hi=100, en=150, q_hi_o_en=120, q_en_o_hi=100, cot_hi=100, cot_en=100)
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=0.9)
    assert "Hindi option terms" not in msg


def test_no_gap():
    cells = _cells(hi=120, en=124)
    msg = gr.localize(cells, gr.run_comparisons(cells), compliance=0.9)
    assert "no gap to localize" in msg and "Hindi option terms" not in msg


def test_missing_cell_not_run():
    cells = _cells()
    del cells["cot_en"]
    comps = gr.run_comparisons(cells)
    assert comps[2]["skipped"] and not comps[0]["skipped"]
    report = gr.build_report(cells, comps, gr.localize(cells, comps, None), [], compliance_lines=[])
    row = [l for l in report.splitlines() if l.startswith("| 3 |")][0]
    assert "not run" in row


def test_missing_control_no_division():
    cells = {"q_hi_o_en": _cell(10, 5)}
    comps = gr.run_comparisons(cells)
    assert all(c["skipped"] for c in comps)
    msg = gr.localize(cells, comps, None)
    assert "incomplete" in msg.lower()


def test_cli_end_to_end(tmp_path):
    def raw(name, answers):
        p = tmp_path / f"himed_{name}_raw.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "question_with_options", "reference_answer", "model_answer"])
            w.writeheader()
            for k, a in enumerate(answers):
                w.writerow({"item_id": f"i{k}", "question_with_options": "Q\nA. a\nB. b",
                            "reference_answer": "A", "model_answer": a})
        return p
    n = 30
    hi = raw("c0_baseline", ["A"] * 10 + ["B"] * 20)
    en = raw("c0_en_google", ["A"] * 28 + ["B"] * 2)
    mix = raw("c0_mix_qhi_oen_google", ["A"] * 27 + ["B"] * 3)
    cot_hi = raw("c2_cot", ["## Thinking\nयकृत\n## Response\nA."] * 10 + ["## Thinking\nयकृत\n## Response\nB."] * 20)
    cot_en = raw("c2_cot_en", ["The liver.\nAnswer: A"] * 11 + ["The liver.\nAnswer: B"] * 19)
    ex = tmp_path / "ex.csv"
    ex.write_text("item_id,reason\ni29,test\n", encoding="utf-8")
    out = tmp_path / "gap.md"
    gr.main(["--cell", f"hi:first_letter:{hi}", "--cell", f"en:first_letter:{en}",
             "--cell", f"q_hi_o_en:first_letter:{mix}", "--cell", f"cot_hi:after_marker:{cot_hi}",
             "--cell", f"cot_en:after_marker:{cot_en}", "--exclude-file", str(ex), "--en-tag", "google",
             "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "Hindi option terms" in text
    assert "c2_cot_en: 100.0%" in text and "c2_cot: 0.0%" in text
    assert "| 2 |" in text and "not run" in text
    assert f"{n - 1} after exclusions" in text


def test_cli_rejects_unknown_cell_and_mode(tmp_path):
    with pytest.raises(SystemExit):
        gr.main(["--cell", f"xx:first_letter:{tmp_path / 'a.csv'}", "--out", str(tmp_path / "o.md")])
    with pytest.raises(SystemExit):
        gr.main(["--cell", f"hi:bogus:{tmp_path / 'a.csv'}", "--out", str(tmp_path / "o.md")])


def test_word_share_counts_words_not_letters():
    # 2 English words (many letters) + 3 Hindi words: letter share > 0.5 but word share = 0.4
    text = "## Thinking\nhypothyroidism thyroxine यकृत में रोग\n## Response\nB."
    assert gr.reasoning_latin_share(text) > 0.5
    assert gr.reasoning_latin_word_share(text) == pytest.approx(0.4)
    assert gr.reasoning_latin_word_share("Answer: B") is None


def test_report_shows_word_level_compliance(tmp_path):
    msgs = ["hypothyroidism thyroxine यकृत में रोग\nAnswer: A"] * 3
    assert gr.compliance_rate(msgs) == 1.0
    assert gr.compliance_rate(msgs, share_fn=gr.reasoning_latin_word_share) == 0.0
