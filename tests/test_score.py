"""Tests for the scoring pipeline (score.py).

These tests verify that BLEU and ROUGE-L scoring works correctly
for Hindi text. We test two extremes:
1. Perfect match (model answer = reference answer) → high scores
2. No match (completely different text) → low scores

Why test extremes? They're easy to reason about. If a perfect match
doesn't give ~1.0 and a total mismatch doesn't give ~0.0, something
is fundamentally broken in the scoring logic.
"""

import csv
from pathlib import Path

from score import score_model


def _write_raw_csv(path: Path, rows: list[dict]) -> None:
    """Helper function to create a fake raw output CSV for testing.

    This simulates what evaluate.py would produce — a CSV with
    question, context, reference_answer, and model_answer columns.
    """
    fieldnames = ["question", "context", "reference_answer", "model_answer"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_score_model_perfect_match(tmp_path):
    """When model answer = reference answer, scores should be ~1.0.

    This tests the 'best case' — the model gave the exact right answer.
    BLEU and ROUGE-L should both be very high (above 0.9).
    """
    raw_path = tmp_path / "test_raw.csv"
    scored_path = tmp_path / "test_scored.csv"

    _write_raw_csv(raw_path, [
        {
            "question": "मधुमेह क्या है?",
            "context": "मधुमेह एक रोग है।",
            "reference_answer": "मधुमेह एक चयापचय रोग है जो शर्करा से होता है",
            "model_answer": "मधुमेह एक चयापचय रोग है जो शर्करा से होता है",  # Exact same text!
        }
    ])

    result = score_model(raw_path, scored_path)

    assert scored_path.exists(), "Scored CSV should be created"
    assert result["mean_rouge_l"] > 0.9, f"ROUGE-L should be >0.9 for perfect match, got {result['mean_rouge_l']}"
    assert result["mean_bleu"] > 0.9, f"BLEU should be >0.9 for perfect match, got {result['mean_bleu']}"


def test_score_model_no_match(tmp_path):
    """When model answer is completely different, scores should be low.

    This tests the 'worst case' — the model gave a totally wrong answer.
    BLEU and ROUGE-L should both be low (below 0.5).
    """
    raw_path = tmp_path / "test_raw.csv"
    scored_path = tmp_path / "test_scored.csv"

    _write_raw_csv(raw_path, [
        {
            "question": "प्रश्न?",
            "context": "संदर्भ",
            "reference_answer": "मधुमेह एक चयापचय रोग है",
            "model_answer": "कृपया चिकित्सक से परामर्श करें",
        }
    ])

    result = score_model(raw_path, scored_path)

    assert result["mean_rouge_l"] < 0.5, f"ROUGE-L should be <0.5 for mismatch, got {result['mean_rouge_l']}"
    assert result["mean_bleu"] < 0.5, f"BLEU should be <0.5 for mismatch, got {result['mean_bleu']}"
