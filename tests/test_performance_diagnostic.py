import pytest

from performance_eval.diagnostic import check_same_items, decide, run_comparisons


def _rows(n_correct, n=100, prefix="q"):
    return [{"item_id": f"{prefix}{i}", "is_correct": i < n_correct, "parsed_letter": "A",
             "marker_found": False} for i in range(n)]


def _cells(gem_hi, himed_hi, nanda_hi, himed_en, nanda_en, gem_en):
    return {("gemini_flash", "hi"): _rows(gem_hi), ("himed", "hi"): _rows(himed_hi),
            ("nanda", "hi"): _rows(nanda_hi), ("himed", "en"): _rows(himed_en),
            ("nanda", "en"): _rows(nanda_en), ("gemini_flash", "en"): _rows(gem_en)}


def test_cells_must_share_items():
    with pytest.raises(SystemExit):
        check_same_items({("himed", "hi"): _rows(5, n=10), ("nanda", "hi"): _rows(5, n=9)})


def test_five_preregistered_comparisons_in_order():
    comps = run_comparisons(_cells(80, 60, 65, 60, 65, 80))
    assert [c["label"] for c in comps] == [
        "gemini_flash/hi vs himed/hi", "gemini_flash/hi vs nanda/hi",
        "himed/en vs himed/hi", "nanda/en vs nanda/hi", "gemini_flash/en vs gemini_flash/hi"]
    assert comps[0]["accuracy_delta"] == pytest.approx(0.20)


def test_bonferroni_m5():
    comps = run_comparisons(_cells(80, 60, 65, 60, 65, 80))
    c = comps[0]
    assert c["p_adj"] == pytest.approx(min(1.0, c["mcnemar_p"] * 5))


def test_bonferroni_p002_not_significant():
    from performance_eval.diagnostic import _significant

    assert not _significant({"accuracy_delta": 0.1, "mcnemar_p": 0.02}, family_size=5)


def test_verdict_open_models_bottleneck():
    cells = _cells(85, 60, 65, 61, 66, 85)
    verdict = decide(cells, run_comparisons(cells))
    assert "open Indic models" in verdict
    assert "Hindi is the bottleneck" not in verdict


def test_verdict_language_bottleneck():
    cells = _cells(66, 60, 65, 80, 85, 70)
    assert "Hindi is the bottleneck" in decide(cells, run_comparisons(cells))


def test_verdict_questions():
    cells = _cells(65, 65, 65, 65, 65, 65)
    assert "questions" in decide(cells, run_comparisons(cells))


def test_missing_cell_skips_comparison_and_says_incomplete():
    cells = _cells(85, 60, 65, 61, 66, 85)
    del cells[("gemini_flash", "en")]
    comps = run_comparisons(cells)
    assert comps[4]["skipped"] is True
    assert "incomplete" in decide(cells, comps)
