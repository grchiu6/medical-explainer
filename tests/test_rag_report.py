import csv

import pytest

from performance_eval.rag import report


def _rows(correct_ids, all_ids):
    return [{"item_id": i, "is_correct": i in correct_ids, "parsed_letter": "A", "marker_found": True}
            for i in all_ids]


IDS = [f"q{i}" for i in range(40)]


def _scored(c0, rand, rag):
    return {"c0_baseline": _rows(set(c0), IDS), "rag_rand_k5": _rows(set(rand), IDS),
            "rag_k5": _rows(set(rag), IDS)}


def test_relevance_effect():
    # c0 and rand share the first 10; rag gets those plus 20 more.
    base = IDS[:10]
    s = _scored(base, base, IDS[:30])
    comps = report.rag_comparisons(s)
    assert [(c["new"], c["prior"], c["tested"]) for c in comps] == [
        ("rag_k5", "c0_baseline", True), ("rag_k5", "rag_rand_k5", True),
        ("rag_rand_k5", "c0_baseline", False)]
    for c in comps[:2]:
        assert c["significant"]
        assert c["p_adj"] == min(1, c["mcnemar_p"] * 2)
    assert report.verdict(comps) == "relevance effect"


def test_extra_context_effect():
    c0 = IDS[:10]
    both = IDS[:30]
    comps = report.rag_comparisons(_scored(c0, both, both))
    assert comps[0]["significant"] and not comps[1]["significant"]
    assert report.verdict(comps) == "extra-context effect"


def test_no_effect_and_descriptive_row():
    comps = report.rag_comparisons(_scored(IDS[:10], IDS[:10], IDS[:10]))
    assert report.verdict(comps) == "no effect"
    assert comps[2]["mcnemar_p"] is None and comps[2]["p_adj"] is None
    assert comps[2]["significant"] is None


def _summ(s):
    from performance_eval.analyze import accuracy_summary
    return {k: accuracy_summary(v) for k, v in s.items()}


def test_types_table_only_when_given(tmp_path):
    s = _scored(IDS[:10], IDS[:10], IDS[:30])
    comps = report.rag_comparisons(s)
    v = report.verdict(comps)
    p1 = tmp_path / "a.md"
    report.write_report(p1, _summ(s), comps, v, None, None, scored=s)
    assert "Per-type" not in p1.read_text()
    p2 = tmp_path / "b.md"
    types = {i: "anatomy" for i in IDS[:39]}
    report.write_report(p2, _summ(s), comps, v, None, types, scored=s)
    text = p2.read_text()
    assert "Per-type" in text and "untagged" in text and "anatomy" in text
    assert "family m = 2" in text and "descriptive only" in text


def test_default_out_name():
    assert report.DEFAULT_OUT.name == "results_himed_rag.md"


def test_cli_refuses_results_himed(tmp_path):
    with pytest.raises(SystemExit):
        report.main(["--raw-dir", str(tmp_path), "--out", str(tmp_path / "results_himed.md")])


def test_cli_end_to_end(tmp_path):
    for cond, correct in [("c0_baseline", 1), ("rag_rand_k5", 1), ("rag_k5", 4)]:
        with open(tmp_path / f"himed_{cond}_raw.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "question_with_options", "reference_answer", "model_answer"])
            for i in range(4):
                w.writerow([f"q{i}", "Q?\nA. x\nB. y\nC. z\nD. w", "A", "A" if i < correct else "B"])
    types = tmp_path / "types.csv"
    types.write_text("item_id,tag\nq0,anatomy\n")
    out = tmp_path / "out" / "results_himed_rag.md"
    report.main(["--raw-dir", str(tmp_path), "--types", str(types), "--out", str(out)])
    text = out.read_text()
    assert "rag_k5" in text and "25.0%" in text and "untagged" in text
    assert (tmp_path / "himed_rag_k5_scored.csv").exists()
