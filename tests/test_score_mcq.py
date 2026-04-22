"""Tests for the MCQ scoring pipeline (score_mcq.py).

Two layers of tests:
1. The letter-extraction helper — catches edge cases in raw model output
2. The end-to-end score_mcq_model() — reads a raw CSV, writes a scored CSV,
   returns aggregate accuracy

Why test both layers? The extraction logic is where 99% of bugs live
(weird model outputs), so we test it exhaustively in isolation. The
end-to-end test just verifies the CSV plumbing.
"""

import csv
from pathlib import Path

import pytest

from score_mcq import extract_letter, score_mcq_model


def _write_raw_csv(path: Path, rows: list[dict]) -> None:
    """Helper to create a fake raw CSV (same shape as evaluate.py produces)."""
    fieldnames = ["question", "reference_answer", "model_answer"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- extract_letter tests ---


def test_extract_letter_clean_single_letter():
    """'A' should extract to 'A' — the ideal case."""
    assert extract_letter("A") == "A"
    assert extract_letter("B") == "B"
    assert extract_letter("E") == "E"


def test_extract_letter_lowercase_is_normalized():
    """'a' should normalize to 'A' — models sometimes lowercase."""
    assert extract_letter("a") == "A"
    assert extract_letter("b") == "B"


def test_extract_letter_with_trailing_punctuation():
    """'A.' or 'A)' or 'A।' (Hindi danda) should still extract 'A'."""
    assert extract_letter("A.") == "A"
    assert extract_letter("A)") == "A"
    assert extract_letter("A।") == "A"
    assert extract_letter("A\n") == "A"


def test_extract_letter_from_full_sentence():
    """When the model writes a sentence, grab the first standalone A-E.

    E.g. 'उत्तर: B है' or 'The answer is C' should give 'B' and 'C'.
    """
    assert extract_letter("उत्तर: B है") == "B"
    assert extract_letter("The answer is C.") == "C"
    assert extract_letter("I think D is correct.") == "D"


def test_extract_letter_returns_none_for_empty():
    """Empty or whitespace-only output should return None (not crash)."""
    assert extract_letter("") is None
    assert extract_letter("   ") is None
    assert extract_letter("\n\n") is None


def test_extract_letter_returns_none_for_no_match():
    """If no A-E letter is present, return None."""
    assert extract_letter("मुझे नहीं पता") is None
    assert extract_letter("F") is None  # outside A-E range
    assert extract_letter("123") is None


def test_extract_letter_ignores_letters_inside_words():
    """'Apollo' should NOT match 'A' — we want standalone letter tokens.

    Why: if a model writes 'Apollo is the answer', we don't want that
    to be scored as 'A'. We need word-boundary matching.
    """
    # 'Apollo' contains 'A' but not as a standalone token
    # However if the ONLY A-E in the string is 'Apollo', we should
    # treat this as no valid letter found
    assert extract_letter("Apollo") is None
    # But a real letter elsewhere should still be found
    assert extract_letter("Apollo says B") == "B"


# --- score_mcq_model tests ---


def test_score_mcq_model_perfect_accuracy(tmp_path):
    """All correct answers → accuracy 1.0."""
    raw_path = tmp_path / "test_raw.csv"
    scored_path = tmp_path / "test_scored.csv"

    _write_raw_csv(raw_path, [
        {"question": "Q1?\nA. ...\nB. ...", "reference_answer": "A", "model_answer": "A"},
        {"question": "Q2?\nA. ...\nB. ...", "reference_answer": "B", "model_answer": "B"},
        {"question": "Q3?\nA. ...\nB. ...", "reference_answer": "C", "model_answer": "उत्तर C है"},
    ])

    result = score_mcq_model(raw_path, scored_path)

    assert scored_path.exists()
    assert result["accuracy"] == 1.0, f"Expected 1.0, got {result['accuracy']}"
    assert result["n_correct"] == 3
    assert result["n_total"] == 3


def test_score_mcq_model_zero_accuracy(tmp_path):
    """All wrong answers → accuracy 0.0."""
    raw_path = tmp_path / "test_raw.csv"
    scored_path = tmp_path / "test_scored.csv"

    _write_raw_csv(raw_path, [
        {"question": "Q1?", "reference_answer": "A", "model_answer": "B"},
        {"question": "Q2?", "reference_answer": "B", "model_answer": "C"},
    ])

    result = score_mcq_model(raw_path, scored_path)

    assert result["accuracy"] == 0.0
    assert result["n_correct"] == 0
    assert result["n_total"] == 2


def test_score_mcq_model_partial_accuracy(tmp_path):
    """Mixed correct/incorrect → correct fraction."""
    raw_path = tmp_path / "test_raw.csv"
    scored_path = tmp_path / "test_scored.csv"

    _write_raw_csv(raw_path, [
        {"question": "Q1?", "reference_answer": "A", "model_answer": "A"},  # correct
        {"question": "Q2?", "reference_answer": "B", "model_answer": "C"},  # wrong
        {"question": "Q3?", "reference_answer": "C", "model_answer": "C"},  # correct
        {"question": "Q4?", "reference_answer": "D", "model_answer": ""},   # unparseable → wrong
    ])

    result = score_mcq_model(raw_path, scored_path)

    # 2 correct out of 4 = 0.5
    assert result["accuracy"] == 0.5
    assert result["n_correct"] == 2
    assert result["n_total"] == 4


def test_score_mcq_model_scored_csv_has_parsed_letter_and_is_correct(tmp_path):
    """The scored CSV should contain parsed_letter and is_correct columns.

    Why: For debugging / error analysis, you need to see WHICH questions
    the model got wrong and WHAT letter it actually output. Just reporting
    aggregate accuracy hides the per-question story.
    """
    raw_path = tmp_path / "test_raw.csv"
    scored_path = tmp_path / "test_scored.csv"

    _write_raw_csv(raw_path, [
        {"question": "Q1?", "reference_answer": "A", "model_answer": "उत्तर A है"},
        {"question": "Q2?", "reference_answer": "B", "model_answer": "C"},
    ])

    score_mcq_model(raw_path, scored_path)

    with open(scored_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["parsed_letter"] == "A"
    assert rows[0]["is_correct"] == "1"
    assert rows[1]["parsed_letter"] == "C"
    assert rows[1]["is_correct"] == "0"
