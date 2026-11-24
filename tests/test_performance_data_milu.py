# Real schema of ai4bharat/MILU, config "Hindi", split "test" (inspected 2026-11-15):
#   14831 rows in test (812 in validation). Columns (all str unless noted):
#     question, option1, option2, option3, option4, target (e.g. "option2"),
#     is_translated (bool), language ("Hindi"), domain ("Health & Medicine"),
#     subject ("Health and Medicine")
#   No id column. Always exactly 4 options. `target` is "option1".."option4".
#   subject == "Health and Medicine" -> 499 rows (matches MILU paper Table 9).
#   Target distribution in subset: option1 116, option2 117, option3 145, option4 121.
#   Example row:
#     {'question': "'एक ओर अजगरहि लखि एक ओर मृगराय ।'\nबिकल बटोही बीच ही परयो मूरछा खाय ।। में कौन सा रस है?",
#      'option1': 'वीर रस', 'option2': 'भयानक रस', 'option3': 'अदुभत रस', 'option4': 'वीभत्स रस',
#      'target': 'option2', 'is_translated': False, 'language': 'Hindi',
#      'domain': 'Health & Medicine', 'subject': 'Health and Medicine'}
#   Question stems may contain newlines; one option in the subset contains a newline.
import pytest

from performance_eval import data_milu
from performance_eval.data_milu import (
    load_milu_hindi_medicine,
    pick_exemplars,
    shuffle_item_options,
    split_test_fewshot,
)


def _row(q="Q?", opts=("a", "b", "c", "d"), target="option2", subject="Health and Medicine"):
    r = {"question": q, "target": target, "is_translated": False,
         "language": "Hindi", "domain": "Health & Medicine", "subject": subject}
    for i, o in enumerate(opts, 1):
        r[f"option{i}"] = o
    return r


def _patch(monkeypatch, rows):
    calls = {}

    def fake(name, config, split=None, token=None):
        calls.update(name=name, config=config, split=split)
        return rows

    monkeypatch.setattr(data_milu, "load_dataset", fake)
    return calls


def test_filters_subject(monkeypatch):
    calls = _patch(monkeypatch, [_row("keep"), _row("drop", subject="Physics")])
    items = load_milu_hindi_medicine()
    assert [i["question"] for i in items] == ["keep"]
    assert calls == {"name": "ai4bharat/MILU", "config": "Hindi", "split": "test"}


def test_answer_letter_matches_correct_option(monkeypatch):
    _patch(monkeypatch, [_row(opts=("w", "x", "y", "z"), target="option3")])
    (it,) = load_milu_hindi_medicine()
    assert it["answer_letter"] == "C"
    assert it["options"][it["answer_letter"]] == "y"


def test_n_options_matches_present_options(monkeypatch):
    r = _row()
    del r["option4"]
    _patch(monkeypatch, [r, _row()])
    a, b = load_milu_hindi_medicine()
    assert (a["n_options"], len(a["options"])) == (3, 3)
    assert (b["n_options"], len(b["options"])) == (4, 4)


def test_duplicate_ids_raise(monkeypatch):
    _patch(monkeypatch, [_row(), _row()])
    with pytest.raises(ValueError, match="uplicate"):
        load_milu_hindi_medicine()


def test_id_is_stable_and_content_based(monkeypatch):
    _patch(monkeypatch, [_row("one"), _row("two")])
    a = load_milu_hindi_medicine()
    _patch(monkeypatch, [_row("two"), _row("one")])
    b = load_milu_hindi_medicine()
    ids_a = {i["question"]: i["id"] for i in a}
    ids_b = {i["question"]: i["id"] for i in b}
    assert ids_a == ids_b
    assert all(v.startswith("milu_hi_") and len(v) == len("milu_hi_") + 12 for v in ids_a.values())


def test_question_with_options_format_and_newline_collapse(monkeypatch):
    _patch(monkeypatch, [_row("stem\nline2", opts=("a", "b\n C. x", "c", "d"))])
    (it,) = load_milu_hindi_medicine()
    assert it["question_with_options"] == "stem\nline2\nA. a\nB. b C. x\nC. c\nD. d"
    for letter, text in it["options"].items():
        assert f"{letter}. {text}" in it["question_with_options"]
    assert "\n" not in it["options"]["B"]


def _items(n):
    return [{"id": f"i{k}", "question": "q", "options": {}, "question_with_options": "q",
             "answer_letter": "ABCD"[k % 4], "n_options": 4} for k in range(n)]


def test_split_is_disjoint():
    items = _items(20)
    test, pool = split_test_fewshot(items, fewshot_n=10, seed=42)
    assert {i["id"] for i in test}.isdisjoint({i["id"] for i in pool})
    assert len(pool) == 10 and len(test) == 10


def test_split_is_deterministic():
    items = _items(20)
    a = split_test_fewshot(items, 10, 42)
    b = split_test_fewshot(items, 10, 42)
    assert [i["id"] for i in a[0]] == [i["id"] for i in b[0]]
    assert [i["id"] for i in a[1]] == [i["id"] for i in b[1]]


def test_pick_exemplars_deterministic_and_from_pool():
    pool = _items(10)
    a = pick_exemplars(pool, 3, 42)
    assert a == pick_exemplars(pool, 3, 42)
    assert len(a) == 3 and all(x in pool for x in a)


def test_pick_exemplars_distinct_letters_when_possible():
    pool = _items(10)
    assert len({x["answer_letter"] for x in pick_exemplars(pool, 3, 7)}) == 3


def test_pick_exemplars_fills_when_few_letters():
    pool = _items(8)
    for x in pool:
        x["answer_letter"] = "A" if x["id"] < "i4" else "B"
    got = pick_exemplars(pool, 3, 1)
    assert len(got) == 3 and len({x["id"] for x in got}) == 3


def _full_item():
    return {"id": "milu_hi_abc", "question": "Q", "options": {"A": "p", "B": "q", "C": "r", "D": "s"},
            "question_with_options": "Q\nA. p\nB. q\nC. r\nD. s", "answer_letter": "C", "n_options": 4}


def test_shuffle_keeps_correct_text_and_id():
    it = _full_item()
    for seed in range(10):
        out = shuffle_item_options(it, seed)
        assert out["id"] == it["id"] and out["n_options"] == 4
        assert out["options"][out["answer_letter"]] == "r"
        assert sorted(out["options"].values()) == ["p", "q", "r", "s"]
        assert list(out["options"]) == ["A", "B", "C", "D"]
        for l, t in out["options"].items():
            assert f"{l}. {t}" in out["question_with_options"]
        assert out["question_with_options"].startswith("Q\n")


def test_shuffle_deterministic_and_non_mutating():
    it = _full_item()
    import copy
    snap = copy.deepcopy(it)
    assert shuffle_item_options(it, 5) == shuffle_item_options(it, 5)
    assert it == snap
    assert any(shuffle_item_options(it, s)["options"] != it["options"] for s in range(10))


# --- Phase A: mixed-language items -----------------------------------------------------------

def _mixed_fixture(tmp_path):
    import json

    hi = {"id": "i1", "question": "कौन सा विटामिन?",
          "options": {"A": "विटामिन A", "B": "विटामिन B", "C": "विटामिन C", "D": "विटामिन D"},
          "answer_letter": "C", "n_options": 4}
    hi["question_with_options"] = data_milu._format_qwo(hi["question"], hi["options"])
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"item_id": "i1", "question": "Which vitamin?",
                             "options": {"A": "Vitamin  A", "B": "Vitamin B", "C": "Vitamin\nC", "D": "Vitamin D"},
                             "flags": []}, ensure_ascii=False) + "\n", encoding="utf-8")
    return p, hi


def test_mixed_hi_hi_is_identity(tmp_path):
    p, hi = _mixed_fixture(tmp_path)
    [it] = data_milu.load_mixed_items(p, [hi], "hi", "hi")
    assert it["question_with_options"] == hi["question_with_options"]


def test_mixed_hi_question_en_options(tmp_path):
    p, hi = _mixed_fixture(tmp_path)
    [it] = data_milu.load_mixed_items(p, [hi], "hi", "en")
    lines = it["question_with_options"].split("\n")
    assert lines[0] == "कौन सा विटामिन?"
    assert lines[1:] == ["A. Vitamin A", "B. Vitamin B", "C. Vitamin C", "D. Vitamin D"]
    assert it["id"] == "i1" and it["answer_letter"] == "C" and it["n_options"] == 4


def test_mixed_en_question_hi_options(tmp_path):
    p, hi = _mixed_fixture(tmp_path)
    [it] = data_milu.load_mixed_items(p, [hi], "en", "hi")
    lines = it["question_with_options"].split("\n")
    assert lines[0] == "Which vitamin?"
    assert lines[1:] == ["A. विटामिन A", "B. विटामिन B", "C. विटामिन C", "D. विटामिन D"]
    assert it["id"] == "i1" and it["answer_letter"] == "C"


def test_mixed_validates_like_english_loader(tmp_path):
    p, hi = _mixed_fixture(tmp_path)
    with pytest.raises(ValueError, match="i9"):
        data_milu.load_mixed_items(p, [{**hi, "id": "i9"}], "hi", "en")
    with pytest.raises(ValueError):
        data_milu.load_mixed_items(p, [hi], "hi", "fr")


def test_load_english_items_unchanged(tmp_path):
    p, hi = _mixed_fixture(tmp_path)
    assert data_milu.load_english_items(p, [hi]) == data_milu.load_mixed_items(p, [hi], "en", "en")
