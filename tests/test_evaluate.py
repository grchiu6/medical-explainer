"""Tests for the evaluation runner (evaluate.py).

These tests use a FAKE model function that takes a prompt and returns
a fixed answer instantly. This is called "mocking" — we replace the
real get_answer() with a fake one so tests don't hit real APIs or
load gigabyte-sized models.

Why mock? We want to test the RUNNER's logic:
- Does it save results to CSV correctly?
- Does it resume from where it left off?
- Does it route the right prompt builder for each dataset?

We DON'T want to test the models themselves here — that's what
test_gemini.py / test_airavata.py / test_apollo.py cover.
"""

import csv
import json
from pathlib import Path
from unittest.mock import patch

from evaluate import run_evaluation


def _fake_get_answer(prompt: str) -> str:
    """A fake model that returns a fixed answer regardless of prompt.

    Same interface as the real get_answer() functions after our refactor:
    takes a pre-built prompt string, returns a string.
    """
    return "यह एक परीक्षण उत्तर है।"


def test_run_evaluation_creates_output_csv_for_indicqa(tmp_path):
    """IndicQA path: runner should create a CSV with the correct columns."""
    fake_data = [
        {"question": "प्रश्न 1?", "context": "संदर्भ 1", "reference_answer": "उत्तर 1"},
        {"question": "प्रश्न 2?", "context": "संदर्भ 2", "reference_answer": "उत्तर 2"},
    ]
    output_file = tmp_path / "test_raw.csv"

    # patch() swaps out load_indicqa for one that returns our fake_data.
    # This means the test doesn't depend on the real JSON being downloaded.
    with patch("evaluate.load_indicqa", return_value=fake_data):
        run_evaluation(
            model_name="test",
            dataset_name="indicqa",
            get_answer_fn=_fake_get_answer,
            sample_size=None,
            output_path=output_file,
        )

    assert output_file.exists()
    with open(output_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    expected_columns = {"question", "context", "reference_answer", "model_answer"}
    assert set(reader.fieldnames) == expected_columns
    assert rows[0]["model_answer"] == "यह एक परीक्षण उत्तर है।"
    # Context should be preserved for IndicQA
    assert rows[0]["context"] == "संदर्भ 1"


def test_run_evaluation_creates_output_csv_for_himed(tmp_path):
    """HiMed MCQ path: runner should load HiMed, write rows with empty context.

    Why empty context: HiMed questions don't have a separate context field —
    the options are embedded in the question itself. We keep the 'context'
    column for schema consistency across datasets, just leave it blank.
    """
    # Write a mock HiMed file into tmp_path/himed/ and point loader at it
    himed_dir = tmp_path / "himed"
    himed_dir.mkdir()
    mock_himed = [
        {
            "question_number": "1",
            "subject": "etiology",
            "type": "MCQ",
            "question": "Q1?\nA. x\nB. y\nC. z\nD. w\nE. v",
            "answer": "A",
            "cot": "...",
            "system": "Ayurveda",
            "source": "HiMed-trad",
        },
        {
            "question_number": "2",
            "subject": "treatment",
            "type": "MCQ",
            "question": "Q2?\nA. x\nB. y\nC. z\nD. w\nE. v",
            "answer": "C",
            "cot": "...",
            "system": "Ayurveda",
            "source": "HiMed-trad",
        },
    ]
    with open(himed_dir / "HiMed-Trad_Bench.json", "w", encoding="utf-8") as f:
        json.dump(mock_himed, f, ensure_ascii=False)

    output_file = tmp_path / "test_himed_raw.csv"

    # Patch the HIMED_DIR constant so the loader finds our mock file
    with patch("data.himed_loader.HIMED_DIR", himed_dir):
        run_evaluation(
            model_name="test",
            dataset_name="himed_trad",
            get_answer_fn=_fake_get_answer,
            sample_size=None,
            output_path=output_file,
        )

    with open(output_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    # For HiMed, the reference_answer is a single letter, context is empty
    assert rows[0]["reference_answer"] == "A"
    assert rows[1]["reference_answer"] == "C"
    assert rows[0]["context"] == ""
    # The question text should still contain the MCQ options
    assert "A. x" in rows[0]["question"]


def test_run_evaluation_resumes_from_existing(tmp_path):
    """If output file already has rows, skip those questions.

    This tests the 'resume' feature — if the script crashes at question 500,
    you can re-run it and it picks up from 501 instead of starting over.
    """
    fake_data = [
        {"question": "प्रश्न 1?", "context": "संदर्भ 1", "reference_answer": "उत्तर 1"},
        {"question": "प्रश्न 2?", "context": "संदर्भ 2", "reference_answer": "उत्तर 2"},
        {"question": "प्रश्न 3?", "context": "संदर्भ 3", "reference_answer": "उत्तर 3"},
    ]
    output_file = tmp_path / "test_raw.csv"

    # Pre-populate with 1 row (simulating a previous partial run)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "context", "reference_answer", "model_answer"]
        )
        writer.writeheader()
        writer.writerow({
            "question": "प्रश्न 1?",
            "context": "संदर्भ 1",
            "reference_answer": "उत्तर 1",
            "model_answer": "पिछला उत्तर",
        })

    # Count how many times the fake model gets called
    call_count = 0

    def counting_get_answer(prompt):
        nonlocal call_count
        call_count += 1
        return "नया उत्तर"

    with patch("evaluate.load_indicqa", return_value=fake_data):
        run_evaluation(
            model_name="test",
            dataset_name="indicqa",
            get_answer_fn=counting_get_answer,
            sample_size=None,
            output_path=output_file,
        )

    # Should only have called the model for the 2 NEW questions
    assert call_count == 2, f"Expected 2 calls, got {call_count}"

    with open(output_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    # First row should keep its original answer (not re-run)
    assert rows[0]["model_answer"] == "पिछला उत्तर"


def test_run_evaluation_rejects_unknown_dataset(tmp_path):
    """Calling with an unknown dataset name should raise ValueError."""
    import pytest
    with pytest.raises(ValueError, match="Unknown dataset"):
        run_evaluation(
            model_name="test",
            dataset_name="this_dataset_does_not_exist",
            get_answer_fn=_fake_get_answer,
            sample_size=None,
            output_path=tmp_path / "should_not_be_created.csv",
        )
