"""Tests for literacy_eval/topics_himed_trad.py — the HiMed-Trad reference set.

Parsing/filter tests use small hand-written strings (in the three option
formats found in the real data). Sampling/finalize tests use a mock JSON
in tmp_path. One test uses the real dataset if present, to lock in the
60-item balanced sample.
"""

import csv
import json
from collections import Counter

import pytest

from literacy_eval import topics_himed_trad as T


# ---------- parse_question ----------

@pytest.mark.parametrize("question", [
    "कफ क्या है?\nA. एक\nB. दो\nC. तीन\nD. चार\nE. पांच",          # newline
    "कफ क्या है? A. एक B. दो C. तीन D. चार E. पांच",                # inline
    "कफ क्या है?<br>A. एक<br>B. दो<br>C. तीन<br>D. चार<br>E. पांच",  # <br>
])
def test_parse_question_handles_all_option_formats(question):
    stem, options = T.parse_question(question)
    assert stem == "कफ क्या है?"
    assert options == {"A": "एक", "B": "दो", "C": "तीन", "D": "चार", "E": "पांच"}


def test_parse_question_returns_none_without_options():
    assert T.parse_question("राजयक्ष्मा में क्या होता है? ऊर्जा, वजन, प्यास।") is None


def test_parse_question_ignores_stray_marker_in_stem():
    q = "विटामिन A. की कमी से क्या होता है?\nA. एक\nB. दो\nC. तीन\nD. चार\nE. पांच"
    stem, options = T.parse_question(q)
    assert stem.startswith("विटामिन")
    assert options["A"] == "एक"


# ---------- normalize_answer ----------

@pytest.mark.parametrize("raw, expected", [
    ("C", "C"), (" b ", "B"), ("e", "E"), ("Е", "E"),  # last is Cyrillic
    ("जब तीनों दोषों का योग होता है", None), ("F", None),
])
def test_normalize_answer(raw, expected):
    assert T.normalize_answer(raw) == expected


# ---------- is_answerable ----------

@pytest.mark.parametrize("stem, expected", [
    ("कफ का मुख्य कारण क्या है?", True),
    ("औषधि को अमृत के समान माना जाता है।", False),        # not a question
    ("कौन सा लक्षण कफ का नहीं है?", False),               # negation
    ("उपरोक्त में से कौन सही है?", False),                 # refers to options
    ("User: मेरा मित्र... Assistant: हाँ?", False),         # chat transcript
    ("", False),
])
def test_is_answerable(stem, expected):
    assert T.is_answerable(stem) is expected


# ---------- draft_topic / draft_key_facts ----------

def test_draft_topic_strips_leadins_and_adds_system():
    topic = T.draft_topic("पारंपरिक चिकित्सा में निम्नलिखित में से कौन सा लक्षण प्रमुख है?", "Unani")
    assert topic == "यूनानी चिकित्सा के अनुसार: कौन सा लक्षण प्रमुख है?"


@pytest.mark.parametrize("cot, expected", [
    # "सही विकल्प है X" restatement
    ("कफ ठंडा होता है। यह भारी होता है। सही विकल्प है ठंडा।",
     ["कफ ठंडा होता है", "यह भारी होता है"]),
    # reverse word order "विकल्प X सही है"
    ("कफ ठंडा होता है। इसलिए विकल्प C सही है।", ["कफ ठंडा होता है"]),
    # meta "इस आइटम ..."
    ("कफ ठंडा होता है। इस आइटम के संदर्भ में मुख्य उत्तर ठंडा है।", ["कफ ठंडा होता है"]),
])
def test_draft_key_facts_drops_answer_restatement_and_meta(cot, expected):
    assert T.draft_key_facts(cot) == expected


def test_draft_key_facts_caps_at_max():
    cot = "एक। दो। तीन। चार। पांच।"
    assert len(T.draft_key_facts(cot)) == T.MAX_KEY_FACTS


# ---------- sampling ----------

def _mock_item(n, system, subject, answer="A"):
    return {
        "question_number": f"{n:011d}",
        "system": system,
        "subject": subject,
        "type": "MCQ",
        "question": f"प्रश्न {n} क्या है?\nA. क{n}\nB. ख\nC. ग\nD. घ\nE. ङ",
        "answer": answer,
        "cot": f"तथ्य {n} एक। तथ्य {n} दो।",
        "source": "HiMed-trad",
    }


@pytest.fixture
def mock_data(tmp_path):
    """5 items per cell, except Yoga/prognosis which has only 1."""
    items, n = [], 0
    for system in T.SYSTEMS:
        for subject in T.SUBJECTS:
            count = 1 if (system, subject) == ("Yoga", "prognosis") else 5
            for _ in range(count):
                items.append(_mock_item(n, system, subject))
                n += 1
    items.append(_mock_item(n, "Sowa-Rigpa", "treatment"))  # excluded system
    path = tmp_path / "trad.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_candidates_excludes_sowa_rigpa(mock_data):
    candidates = T.load_candidates(mock_data)
    assert all(c["system"] in T.SYSTEMS for c in candidates)
    assert candidates[0]["correct_option_text"].startswith("क")


def test_stratified_sample_is_balanced_and_fills_thin_cells(mock_data):
    sample = T.stratified_sample(T.load_candidates(mock_data))
    primaries = [s for s in sample if s["role"] == "primary"]
    assert len(primaries) == len(T.SYSTEMS) * len(T.SUBJECTS) * T.PER_CELL
    # Yoga/prognosis has 1 item; the shortfall is filled from other Yoga subjects.
    assert Counter(p["system"] for p in primaries)["Yoga"] == len(T.SUBJECTS) * T.PER_CELL
    assert len({s["item_id"] for s in sample}) == len(sample)  # no duplicates


def test_stratified_sample_is_deterministic(mock_data):
    candidates = T.load_candidates(mock_data)
    a = [s["item_id"] for s in T.stratified_sample(candidates)]
    b = [s["item_id"] for s in T.stratified_sample(list(reversed(candidates)))]
    assert a == b


# ---------- review CSV → finalize ----------

def _read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=T.REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_write_review_csv_refuses_to_overwrite(mock_data, tmp_path):
    items = T.stratified_sample(T.load_candidates(mock_data))
    path = tmp_path / "review.csv"
    T.write_review_csv(items, path)
    with pytest.raises(FileExistsError):
        T.write_review_csv(items, path)


def test_finalize_requires_review(mock_data, tmp_path):
    review = tmp_path / "review.csv"
    T.write_review_csv(T.stratified_sample(T.load_candidates(mock_data)), review)
    with pytest.raises(ValueError, match="not marked reviewed"):
        T.finalize(review, tmp_path / "ref.json")


def test_finalize_promotes_reserve_and_uses_final_edits(mock_data, tmp_path):
    review = tmp_path / "review.csv"
    T.write_review_csv(T.stratified_sample(T.load_candidates(mock_data)), review)
    rows = _read_rows(review)
    for row in rows:
        row["reviewed"] = "y"

    first = next(r for r in rows if r["role"] == "primary")
    reserve = next(r for r in rows if r["role"] == "reserve"
                   and (r["system"], r["subject"]) == (first["system"], first["subject"]))
    first["exclude"] = "y"
    reserve["topic_hi_final"] = "संपादित प्रश्न?"
    reserve["key_fact_1"] = "संपादित तथ्य"
    _write_rows(review, rows)

    ref = T.finalize(review, tmp_path / "ref.json")
    ids = {r["id"] for r in ref}
    assert first["item_id"] not in ids
    promoted = next(r for r in ref if r["id"] == reserve["item_id"])
    assert promoted["topic_hi"] == "संपादित प्रश्न?"
    assert promoted["key_facts"][0] == "संपादित तथ्य"
    assert len(ref) == len(T.SYSTEMS) * len(T.SUBJECTS) * T.PER_CELL
    assert T.load_himed_trad_topics(tmp_path / "ref.json") == ref


@pytest.mark.parametrize("bad_fact", ["इसलिए विकल्प B सही है", "इस आइटम के संदर्भ में मुख्य उत्तर X है"])
def test_finalize_rejects_meta_or_option_facts(mock_data, tmp_path, bad_fact):
    review = tmp_path / "review.csv"
    T.write_review_csv(T.stratified_sample(T.load_candidates(mock_data)), review)
    rows = _read_rows(review)
    for row in rows:
        row["reviewed"] = "y"
    rows[0]["key_fact_1"] = bad_fact  # overrides the clean draft
    _write_rows(review, rows)
    with pytest.raises(ValueError, match="option / is meta"):
        T.finalize(review, tmp_path / "ref.json")


def test_replacement_draw_fills_excluded_cells(mock_data, tmp_path):
    review = tmp_path / "review.csv"
    candidates = T.load_candidates(mock_data)
    T.write_review_csv(T.stratified_sample(candidates), review)
    rows = _read_rows(review)
    # Exclude an entire cell (both primaries + reserve) that has spares in the pool.
    target = ("Ayurveda", "etiology")  # mock has 5 items here, only 3 sampled
    for row in rows:
        if (row["system"], row["subject"]) == target:
            row["exclude"] = "y"
    _write_rows(review, rows)

    new_rows = T.replacement_draw(review, candidates=candidates)
    assert new_rows, "expected replacements for the emptied cell"
    assert all((r["system"], r["subject"]) == target for r in new_rows)
    # New rows are unreviewed and don't duplicate existing item_ids.
    old_ids = {r["item_id"] for r in rows}
    assert all(r["item_id"] not in old_ids and r["reviewed"] == "" for r in new_rows)
    # Deterministic: same call on a fresh copy yields the same ids.
    review2 = tmp_path / "review2.csv"
    _write_rows(review2, rows)
    again = T.replacement_draw(review2, candidates=candidates)
    assert [r["item_id"] for r in new_rows] == [r["item_id"] for r in again]


def test_finalize_rejects_answer_leak(mock_data, tmp_path):
    review = tmp_path / "review.csv"
    T.write_review_csv(T.stratified_sample(T.load_candidates(mock_data)), review)
    rows = _read_rows(review)
    for row in rows:
        row["reviewed"] = "y"
    leaky = next(r for r in rows if r["role"] == "primary")
    leaky["topic_hi_final"] = f"क्या {leaky['correct_option_text']} सही है?"
    _write_rows(review, rows)
    with pytest.raises(ValueError, match="contains the correct answer"):
        T.finalize(review, tmp_path / "ref.json")


# ---------- real data (skipped if the dataset isn't present) ----------

@pytest.mark.skipif(not T._DATA_PATH.exists(), reason="HiMed-Trad data not present")
def test_real_sample_is_60_balanced_items():
    sample = T.stratified_sample(T.load_candidates())
    primaries = [s for s in sample if s["role"] == "primary"]
    assert len(primaries) == 60
    assert set(Counter(p["system"] for p in primaries).values()) == {10}
