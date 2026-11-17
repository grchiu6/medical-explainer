import csv

import pytest

from performance_eval.rag.spotcheck import sample_spotcheck_ids, summarize, write_sheet

HEADER = ["item_id", "question_with_options", "passage_1", "passage_2", "passage_3",
          "passage_4", "passage_5", "relevance"]


def _items(n):
    return [{"id": f"it{i:02d}", "question_with_options": f"Q{i}\nA. a\nB. b",
             "answer_letter": "D"} for i in range(n)]


def _passages(items):
    return {it["id"]: [{"rank": r, "passage_id": f"s:{r}", "source": "s",
                        "title": f"T{r}", "content": f"body {r}", "score": 0.5}
                       for r in range(1, 6)] for it in items}


def test_sampling_deterministic_and_order_independent():
    ids = [f"id{i}" for i in range(100)]
    a = sample_spotcheck_ids(ids, n=20, seed=42)
    assert len(a) == 20 and len(set(a)) == 20
    assert a == sample_spotcheck_ids(ids, n=20, seed=42)
    assert a == sample_spotcheck_ids(list(reversed(ids)), n=20, seed=42)


def test_sheet_columns_and_no_gold(tmp_path):
    items = _items(3)
    p = tmp_path / "sheet.csv"
    write_sheet(p, items, _passages(items))
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == HEADER
    assert [r[0] for r in rows[1:]] == ["it00", "it01", "it02"]
    assert rows[1][2] == "T1\nbody 1"
    assert rows[1][-1] == ""
    assert "answer_letter" not in rows[0]
    assert "D" not in rows[0]


def _labelled(tmp_path, labels):
    items = _items(len(labels))
    p = tmp_path / "s.csv"
    write_sheet(p, items, _passages(items))
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    for r, lab in zip(rows[1:], labels):
        r[-1] = lab
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f).writerows(rows)
    return p


def test_gate_pass(tmp_path):
    p = _labelled(tmp_path, ["relevant"] * 10 + ["partial"] * 10)
    assert summarize(p) == {"relevant": 10, "partial": 10, "none": 0, "passes_gate": True}


def test_gate_fail(tmp_path):
    p = _labelled(tmp_path, ["Relevant "] * 9 + ["partial"] * 11)
    r = summarize(p)
    assert r["relevant"] == 9 and r["passes_gate"] is False


def test_blank_and_unknown_raise(tmp_path):
    with pytest.raises(ValueError, match="it01"):
        summarize(_labelled(tmp_path, ["relevant", ""]), expected_n=2)
    with pytest.raises(ValueError, match="it00"):
        summarize(_labelled(tmp_path, ["yes", "none"]), expected_n=2)


def _sheet_with(tmp_path, n, dup=False):
    p = tmp_path / "s.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i in range(n):
            iid = "it00" if dup and i == 1 else f"it{i:02d}"
            w.writerow([iid, "Q", "", "", "", "", "", "relevant"])
    return p


def test_summarize_rejects_truncated_sheet(tmp_path):
    with pytest.raises(ValueError, match="20"):
        summarize(_sheet_with(tmp_path, 19))
    assert summarize(_sheet_with(tmp_path, 19), expected_n=19)["relevant"] == 19


def test_summarize_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="unique|duplicate"):
        summarize(_sheet_with(tmp_path, 20, dup=True))


def test_sheet_truncates_passages_like_the_prompt(tmp_path):
    from performance_eval.prompts import PASSAGE_MAX_CHARS

    items = _items(1)
    ps = {"it00": [{"rank": 1, "title": "T", "content": "x" * (PASSAGE_MAX_CHARS + 500)}]}
    p = tmp_path / "t.csv"
    write_sheet(p, items, ps)
    with open(p, encoding="utf-8-sig", newline="") as f:
        cell = list(csv.reader(f))[1][2]
    assert cell == "T\n" + "x" * PASSAGE_MAX_CHARS
