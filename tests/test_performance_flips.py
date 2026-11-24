import csv

import pytest

from performance_eval import flips


def _row(i, correct, pred="A", gold="A", qwo="Q"):
    return {"item_id": i, "is_correct": correct, "parsed_letter": pred, "reference_answer": gold,
            "question_with_options": qwo}


def test_flip_rows_directions():
    hi = [_row("a", True, qwo="HA"), _row("b", False, pred="B", qwo="HB"), _row("c", True)]
    en = [_row("a", False, pred="C", qwo="EA"), _row("b", True, qwo="EB"), _row("c", True)]
    rows = flips.flip_rows(hi, en)
    assert [(r["item_id"], r["direction"]) for r in rows] == [("a", "hi_only_correct"), ("b", "en_only_correct")]
    a = rows[0]
    assert a["hi_pred"] == "A" and a["en_pred"] == "C" and a["gold"] == "A"
    assert a["hindi_item"] == "HA" and a["english_item"] == "EA"
    assert a["category"] == "" and a["note"] == "" and a["same_flip_alt"] == ""
    assert list(a) == flips.COLUMNS


def test_flip_rows_rejects_different_items():
    with pytest.raises(ValueError):
        flips.flip_rows([_row("a", True)], [_row("b", True)])


def test_same_flip_alt():
    hi = [_row("a", True), _row("b", False)]
    en = [_row("a", False), _row("b", True)]
    alt = [_row("a", False), _row("b", False)]
    rows = flips.flip_rows(hi, en, alt)
    assert {r["item_id"]: r["same_flip_alt"] for r in rows} == {"a": True, "b": False}


def test_none_prediction_rendered_empty():
    rows = flips.flip_rows([_row("a", False, pred=None)], [_row("a", True)])
    assert rows[0]["hi_pred"] == ""


def test_summarize_rejects_unknown_category():
    rows = [{"item_id": "a", "direction": "en_only_correct", "category": "terminology", "same_flip_alt": "True"},
            {"item_id": "b", "direction": "en_only_correct", "category": "vibes", "same_flip_alt": "True"},
            {"item_id": "c", "direction": "hi_only_correct", "category": "", "same_flip_alt": "True"}]
    with pytest.raises(ValueError, match="b") as e:
        flips.summarize(rows)
    assert "c" in str(e.value)


def test_summarize_counts():
    rows = [{"item_id": "a", "direction": "en_only_correct", "category": "terminology", "same_flip_alt": "True"},
            {"item_id": "b", "direction": "en_only_correct", "category": "terminology", "same_flip_alt": "False"},
            {"item_id": "c", "direction": "hi_only_correct", "category": "item_noise", "same_flip_alt": "True"}]
    md = flips.summarize(rows)
    assert md.splitlines()[0] == "Labels: one LLM rater (Claude), no human check."
    assert "| terminology | 2 | 0 | 2 |" in md
    assert "| item_noise | 0 | 1 | 1 |" in md
    assert "| total | 2 | 1 | 3 |" in md
    assert "2 of 3" in md  # same flip under the alternative translation


def test_cli_export_and_summarize(tmp_path):
    def scored(name, rows):
        p = tmp_path / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["item_id", "question_with_options", "reference_answer",
                                              "model_answer", "parsed_letter", "marker_found", "is_correct"])
            w.writeheader()
            for r in rows:
                w.writerow({"item_id": r[0], "question_with_options": r[1], "reference_answer": "A",
                            "model_answer": r[2], "parsed_letter": r[2], "marker_found": 0,
                            "is_correct": int(r[2] == "A")})
        return p
    hi = scored("hi.csv", [("a", "हिंदी", "A"), ("b", "हिंदी", "B"), ("x", "हिंदी", "B")])
    en = scored("en.csv", [("a", "Eng", "B"), ("b", "Eng", "A"), ("x", "Eng", "A")])
    ex = tmp_path / "ex.csv"
    ex.write_text("item_id,reason\nx,test\n", encoding="utf-8")
    out = tmp_path / "flips.csv"
    flips.main(["export", "--hi", str(hi), "--en", str(en), "--alt-en", str(en), "--exclude-file", str(ex),
                "--out", str(out)])
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [r["item_id"] for r in rows] == ["a", "b"]
    assert rows[0]["same_flip_alt"] == "True"
    for r in rows:
        r["category"] = "other"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flips.COLUMNS)
        w.writeheader()
        w.writerows(rows)
    md = tmp_path / "s.md"
    flips.main(["summarize", "--labeled", str(out), "--out", str(md)])
    assert "| other | 1 | 1 | 2 |" in md.read_text(encoding="utf-8")
