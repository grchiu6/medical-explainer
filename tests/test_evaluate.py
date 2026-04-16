"""Tests for the evaluation runner (evaluate.py).

These tests use a FAKE model function instead of calling real APIs or
loading real models. This is called "mocking" — we replace the real
get_answer() with a fake one that returns a fixed answer instantly.

Why mock? Because we want to test the RUNNER'S logic:
- Does it save results to CSV correctly?
- Does it resume from where it left off?
- Does it handle the data format properly?

We DON'T want to test the models themselves here — that's what
test_gemini.py, test_airavata.py, etc. are for.
"""

import csv
from pathlib import Path
from unittest.mock import patch

from evaluate import run_evaluation


def _fake_get_answer(question: str, context: str) -> str:
    """A fake model that instantly returns a fixed answer.

    This replaces the real get_answer() during testing.
    It takes the same arguments (question, context) and returns
    a string — same interface as the real thing.
    """
    return "यह एक परीक्षण उत्तर है।"


def test_run_evaluation_creates_output_csv(tmp_path):
    """run_evaluation should create a CSV with the correct columns.

    tmp_path is a pytest 'fixture' — it automatically creates a
    temporary directory for each test that gets cleaned up afterward.
    This ensures tests don't leave files lying around and don't
    interfere with each other.
    """
    # Create some fake data (same format as load_indicqa returns)
    fake_data = [
        {"question": "प्रश्न 1?", "context": "संदर्भ 1", "reference_answer": "उत्तर 1"},
        {"question": "प्रश्न 2?", "context": "संदर्भ 2", "reference_answer": "उत्तर 2"},
    ]
    output_file = tmp_path / "test_raw.csv"

    # patch() temporarily replaces load_indicqa with a function that
    # returns our fake data instead of reading the real JSON file.
    # This means the test doesn't depend on the dataset being downloaded.
    with patch("evaluate.load_indicqa", return_value=fake_data):
        run_evaluation(
            model_name="test",
            get_answer_fn=_fake_get_answer,
            sample_size=None,
            output_path=output_file,
        )

    # Verify the CSV was created with correct structure
    assert output_file.exists(), "Output CSV should exist"

    with open(output_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
    expected_columns = {"question", "context", "reference_answer", "model_answer"}
    assert set(reader.fieldnames) == expected_columns
    assert rows[0]["model_answer"] == "यह एक परीक्षण उत्तर है।"


def test_run_evaluation_resumes_from_existing(tmp_path):
    """If output file already has rows, skip those questions.

    This tests the 'resume' feature — if the script crashes at
    question 500, you can re-run it and it picks up from 501
    instead of starting over. This is critical for long runs
    (1,052 questions on a slow API = could take hours).
    """
    fake_data = [
        {"question": "प्रश्न 1?", "context": "संदर्भ 1", "reference_answer": "उत्तर 1"},
        {"question": "प्रश्न 2?", "context": "संदर्भ 2", "reference_answer": "उत्तर 2"},
        {"question": "प्रश्न 3?", "context": "संदर्भ 3", "reference_answer": "उत्तर 3"},
    ]
    output_file = tmp_path / "test_raw.csv"

    # Pre-populate the CSV with 1 row (simulating a previous partial run)
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

    def counting_get_answer(question, context):
        nonlocal call_count
        call_count += 1
        return "नया उत्तर"

    with patch("evaluate.load_indicqa", return_value=fake_data):
        run_evaluation(
            model_name="test",
            get_answer_fn=counting_get_answer,
            sample_size=None,
            output_path=output_file,
        )

    # Should only have called the model for the 2 NEW questions
    assert call_count == 2, f"Expected 2 calls, got {call_count}"

    # Total rows should be 3 (1 old + 2 new)
    with open(output_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    # First row should keep its original answer (not re-run)
    assert rows[0]["model_answer"] == "पिछला उत्तर"
