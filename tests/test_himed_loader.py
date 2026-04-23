"""Tests for data/himed_loader.py — written first (TDD).

HiMed is a Hindi medical MCQ (multiple-choice) dataset from Kaggle.
Each question is a single dict with fields like:
    {
        "question_number": "00200033010",
        "subject": "etiology",
        "type": "MCQ",
        "question": "<Hindi text>\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nE. ...",
        "answer": "A",
        "cot": "<Hindi reasoning>",
        "system": "Ayurveda",
        "source": "HiMed-trad"
    }

Unlike IndicQA (extractive QA — find the answer in a passage), HiMed is
MCQ — the question contains the options, and the answer is a single letter.

These tests use a MOCK json file in tmp_path so they don't depend on
the actual HiMed download being present.
"""

import json
from pathlib import Path

import pytest

from data.himed_loader import load_himed


# Mock HiMed data — same shape as the real JSON but only 3 entries
# so tests run fast and we can reason about exact counts.
MOCK_HIMED_DATA = [
    {
        "question_number": "00200033010",
        "subject": "etiology",
        "type": "MCQ",
        "question": "जीवाणु के कौन से तत्व से निर्मित होने की बात कही गई है?\nA. पञ्चमहाभूत\nB. विशुद्ध रासायनिक यौगिक\nC. मिथकीय पदार्थ\nD. नाभिकीय कण\nE. बिजली",
        "answer": "A",
        "cot": "पारंपरिक चिकित्सा में, जीवाणुओं का जन्म...",
        "system": "Ayurveda",
        "source": "HiMed-trad",
    },
    {
        "question_number": "00200033011",
        "subject": "treatment",
        "type": "MCQ",
        "question": "उपचार का सबसे अच्छा तरीका क्या है?\nA. विकल्प 1\nB. विकल्प 2\nC. विकल्प 3\nD. विकल्प 4\nE. विकल्प 5",
        "answer": "C",
        "cot": "क्योंकि...",
        "system": "Ayurveda",
        "source": "HiMed-trad",
    },
    {
        "question_number": "00200033012",
        "subject": "diagnosis",
        "type": "MCQ",
        "question": "निदान कैसे किया जाता है?\nA. विकल्प 1\nB. विकल्प 2\nC. विकल्प 3\nD. विकल्प 4\nE. विकल्प 5",
        "answer": "B",
        "cot": "निदान के लिए...",
        "system": "Ayurveda",
        "source": "HiMed-trad",
    },
]

EXPECTED_KEYS = {"question", "answer", "subject", "system", "source"}


def _write_mock_himed(path: Path, data: list[dict]) -> None:
    """Helper to write mock HiMed JSON into a tmp file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def test_load_himed_returns_list_of_dicts(tmp_path):
    """Loading mock data returns a list of dicts with expected keys.

    Why this test: every downstream component (evaluate.py, score_mcq.py)
    assumes this exact dict shape. If the loader changes shape silently,
    everything breaks downstream. This test pins the contract.
    """
    mock_file = tmp_path / "HiMed-Trad_Bench.json"
    _write_mock_himed(mock_file, MOCK_HIMED_DATA)

    items = load_himed(file_path=mock_file)

    assert isinstance(items, list), "Result should be a list"
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"

    for item in items:
        assert isinstance(item, dict)
        assert EXPECTED_KEYS == set(item.keys()), (
            f"Expected keys {EXPECTED_KEYS}, got {set(item.keys())}"
        )


def test_load_himed_preserves_question_with_options(tmp_path):
    """The 'question' field must include the A-E options (don't strip them).

    Why this matters: For MCQ, the model needs to SEE the options to pick
    one. If the loader stripped them, the model would have no choices.
    """
    mock_file = tmp_path / "HiMed-Trad_Bench.json"
    _write_mock_himed(mock_file, MOCK_HIMED_DATA)

    items = load_himed(file_path=mock_file)

    first = items[0]
    assert "A. पञ्चमहाभूत" in first["question"], "Option A should be in question"
    assert "B. विशुद्ध रासायनिक यौगिक" in first["question"], "Option B should be in question"
    assert "E. बिजली" in first["question"], "Option E should be in question"


def test_load_himed_answer_is_single_letter(tmp_path):
    """Reference answer should be a single uppercase letter A-E.

    Why: MCQ scoring relies on this. If 'answer' is something weird
    like 'A. पञ्चमहाभूत', our letter-comparison would fail.
    """
    mock_file = tmp_path / "HiMed-Trad_Bench.json"
    _write_mock_himed(mock_file, MOCK_HIMED_DATA)

    items = load_himed(file_path=mock_file)

    for item in items:
        assert item["answer"] in {"A", "B", "C", "D", "E"}, (
            f"Answer should be a single letter A-E, got '{item['answer']}'"
        )


def test_load_himed_sample_size(tmp_path):
    """sample_size=2 should return exactly 2 items, deterministically."""
    mock_file = tmp_path / "HiMed-Trad_Bench.json"
    _write_mock_himed(mock_file, MOCK_HIMED_DATA)

    items_a = load_himed(file_path=mock_file, sample_size=2)
    items_b = load_himed(file_path=mock_file, sample_size=2)

    assert len(items_a) == 2
    assert len(items_b) == 2
    # Same seed → same sample → reproducible
    assert items_a == items_b, "Sampling should be reproducible (same seed)"


def test_load_himed_variant_dispatch(tmp_path, monkeypatch):
    """variant='trad' loads HiMed-Trad_Bench.json from data/himed/.

    Why: We want users to call load_himed(variant='trad') without
    specifying paths — the loader should know where to look.
    """
    # Set up a fake himed dir with a mock Trad file
    fake_himed_dir = tmp_path / "himed"
    fake_himed_dir.mkdir()
    _write_mock_himed(fake_himed_dir / "HiMed-Trad_Bench.json", MOCK_HIMED_DATA)

    # Point the loader at our fake dir via monkeypatch
    import data.himed_loader as himed_loader
    monkeypatch.setattr(himed_loader, "HIMED_DIR", fake_himed_dir)

    items = load_himed(variant="trad")
    assert len(items) == 3


def test_load_himed_missing_file_raises(tmp_path):
    """Loading a nonexistent file raises FileNotFoundError with helpful message."""
    nonexistent = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_himed(file_path=nonexistent)


def test_load_himed_west_schema(tmp_path):
    """West variant uses a separate 'options' dict instead of embedding them.

    The loader should detect this and re-embed the options into the question
    text so the model sees the choices when prompted.
    """
    mock_west_data = [{
        "question_number": 1,
        "question": "What does this gene do?",
        "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "answer": "D",
        "source": "GPQA-med",
    }]
    mock_file = tmp_path / "HiMed-West_Bench.json"
    _write_mock_himed(mock_file, mock_west_data)

    items = load_himed(file_path=mock_file)

    assert len(items) == 1
    q = items[0]["question"]
    assert "What does this gene do?" in q, "Question stem should be present"
    assert "A. alpha" in q, "Option A should be appended to question"
    assert "D. delta" in q, "Option D should be appended to question"
    assert items[0]["answer"] == "D"
