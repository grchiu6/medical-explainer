import json

import pytest

from performance_eval.data_milu import load_english_items
from performance_eval.translate import (
    agreement,
    build_translate_prompt,
    check_translation,
    chrf,
    parse_translation,
    translate_google,
    write_bad_translations,
)

HI = {"id": "i1", "question": "कौन सा विटामिन?\n1. कथन एक\n2. कथन दो",
      "options": {"A": "विटामिन A", "B": "विटामिन B", "C": "विटामिन C", "D": "विटामिन D"},
      "answer_letter": "C", "n_options": 4}
EN_OK = {"question": "Which vitamin?\n1. Statement one\n2. Statement two",
         "options": {"A": "Vitamin A", "B": "Vitamin B", "C": "Vitamin C", "D": "Vitamin D"}}


def test_clean_translation_has_no_flags():
    assert check_translation(HI, EN_OK) == []


def test_option_count_mismatch_flag():
    tr = {**EN_OK, "options": {"A": "a", "B": "b", "C": "c"}}
    assert "option_count_mismatch" in check_translation(HI, tr)


def test_empty_option_flag():
    tr = {**EN_OK, "options": {**EN_OK["options"], "B": "  "}}
    assert "empty_option" in check_translation(HI, tr)


def test_devanagari_left_flag():
    tr = {**EN_OK, "question": "कौन सा विटामिन? which\n1. x\n2. y"}
    assert "devanagari_left" in check_translation(HI, tr)


def test_numbering_lost_flag():
    tr = {**EN_OK, "question": "Which vitamin? Statement one and statement two"}
    assert "numbering_lost" in check_translation(HI, tr)


def test_parse_translation_bad_json_flag():
    tr, flags = parse_translation("not json at all", 4)
    assert tr is None and flags == ["bad_json"]


def test_parse_translation_accepts_fenced_json():
    text = "```json\n" + json.dumps(EN_OK) + "\n```"
    tr, flags = parse_translation(text, 4)
    assert tr == EN_OK and flags == []


def test_translate_prompt_has_question_not_answer():
    p = build_translate_prompt(HI)
    assert "कौन सा विटामिन?" in p and "answer_letter" not in p
    assert build_translate_prompt({**HI, "answer_letter": "A"}) == p


def test_translate_google_sends_question_and_options_in_order():
    sent = {}

    def fake_post(url, params, data, timeout):
        sent.update(url=url, params=params, data=data)
        out = [{"translatedText": t} for t in [EN_OK["question"], *EN_OK["options"].values()]]
        return type("R", (), {"status_code": 200, "raise_for_status": lambda s: None,
                              "json": lambda s: {"data": {"translations": out}}})()

    [rec] = translate_google([HI], "KEY", post=fake_post)
    assert sent["data"]["q"] == [HI["question"], *HI["options"].values()]
    assert sent["data"]["source"] == "hi" and sent["data"]["target"] == "en"
    assert sent["data"]["format"] == "text" and sent["params"] == {"key": "KEY"}
    assert rec["item_id"] == "i1" and rec["options"] == EN_OK["options"] and rec["flags"] == []


def test_chrf_identical_is_one_and_disjoint_is_zero():
    assert chrf("vitamin c", "vitamin c") == pytest.approx(1.0)
    assert chrf("aaaa", "zzzz") == 0.0


def test_agreement_per_item():
    a = {"i1": {"question": "Which vitamin", "options": {"A": "x"}}}
    b = {"i1": {"question": "Which vitamin", "options": {"A": "x"}}}
    assert agreement(a, b)["i1"] == pytest.approx(1.0)


def _write_jsonl(path, recs):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def test_load_english_items_keeps_id_and_gold(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [{"item_id": "i1", **EN_OK, "flags": []}])
    [it] = load_english_items(p, [HI])
    assert it["id"] == "i1" and it["answer_letter"] == "C" and it["n_options"] == 4
    assert it["question_with_options"].startswith("Which vitamin?")
    assert it["question_with_options"].endswith("D. Vitamin D")


def test_load_english_items_missing_translation_raises(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [])
    with pytest.raises(ValueError, match="i1"):
        load_english_items(p, [HI])


def test_load_english_items_drop_ids(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(p, [])
    assert load_english_items(p, [HI], drop_ids={"i1"}) == []


def test_write_bad_translations_lists_structural_flags_only(tmp_path):
    recs = [{"item_id": "a", "flags": ["option_count_mismatch"]},
            {"item_id": "b", "flags": ["devanagari_left"]},
            {"item_id": "c", "flags": []}]
    out = tmp_path / "bad.csv"
    write_bad_translations(recs, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["item_id,reason", "a,option_count_mismatch"]
