"""Tests for results_table.py — the script that aggregates scored CSVs.

We're testing in layers:
1. parse_scored_filename — pure string parsing, no I/O
2. compute_metrics_for_file — reads ONE scored CSV, returns metrics
3. build_results_table — finds all scored CSVs, builds the full table

Each layer is tested independently so when something breaks, the test name
tells you exactly which layer is wrong.
"""

import csv
from pathlib import Path

import pytest

from results_table import (
    build_results_table,
    compute_metrics_for_file,
    parse_scored_filename,
)


def _write_indicqa_scored_csv(path: Path, rows: list[dict]) -> None:
    """Helper: create a fake IndicQA scored CSV (matches score.py's output)."""
    fieldnames = ["question", "reference_answer", "model_answer", "rouge_l", "bleu"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_himed_scored_csv(path: Path, rows: list[dict]) -> None:
    """Helper: create a fake HiMed scored CSV (matches score_mcq.py's output)."""
    fieldnames = ["question", "reference_answer", "model_answer", "parsed_letter", "is_correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- parse_scored_filename tests ---

def test_parse_scored_filename_simple_dataset():
    """A dataset name with no underscores should parse cleanly."""
    model, dataset = parse_scored_filename("gemini_indicqa_scored.csv")
    assert model == "gemini"
    assert dataset == "indicqa"


def test_parse_scored_filename_multipart_dataset():
    """A dataset name like 'himed_trad' (contains underscore) should stay intact.

    This is the tricky case — naive splitting would give ('gemini', 'himed')
    and lose the '_trad' part.
    """
    model, dataset = parse_scored_filename("gemini_himed_trad_scored.csv")
    assert model == "gemini"
    assert dataset == "himed_trad"


def test_parse_scored_filename_works_for_all_models():
    """Make sure the parse works regardless of which model name is used."""
    assert parse_scored_filename("airavata_himed_west_scored.csv") == ("airavata", "himed_west")
    assert parse_scored_filename("apollo_indicqa_scored.csv") == ("apollo", "indicqa")


# --- compute_metrics_for_file tests ---

def test_compute_metrics_for_indicqa(tmp_path):
    """For IndicQA, returns BLEU and ROUGE-L means across all rows."""
    path = tmp_path / "test_scored.csv"
    _write_indicqa_scored_csv(path, [
        {"question": "Q1", "reference_answer": "R1", "model_answer": "M1",
         "rouge_l": "0.4", "bleu": "0.3"},
        {"question": "Q2", "reference_answer": "R2", "model_answer": "M2",
         "rouge_l": "0.6", "bleu": "0.5"},
    ])

    metrics = compute_metrics_for_file(path, dataset="indicqa")

    # Check the dict has the right keys (display names, not column names)
    assert set(metrics.keys()) == {"BLEU", "ROUGE-L"}
    # pytest.approx handles tiny float rounding errors (see explanation below)
    assert metrics["BLEU"] == pytest.approx(0.4)     # mean of 0.3, 0.5
    assert metrics["ROUGE-L"] == pytest.approx(0.5)  # mean of 0.4, 0.6


def test_compute_metrics_for_himed(tmp_path):
    """For HiMed (MCQ), accuracy = mean of is_correct column."""
    path = tmp_path / "test_scored.csv"
    _write_himed_scored_csv(path, [
        {"question": "Q1", "reference_answer": "A", "model_answer": "A",
         "parsed_letter": "A", "is_correct": "1"},
        {"question": "Q2", "reference_answer": "B", "model_answer": "C",
         "parsed_letter": "C", "is_correct": "0"},
        {"question": "Q3", "reference_answer": "C", "model_answer": "C",
         "parsed_letter": "C", "is_correct": "1"},
        {"question": "Q4", "reference_answer": "D", "model_answer": "D",
         "parsed_letter": "D", "is_correct": "1"},
    ])

    metrics = compute_metrics_for_file(path, dataset="himed_trad")

    assert set(metrics.keys()) == {"Accuracy"}
    assert metrics["Accuracy"] == pytest.approx(0.75)  # 3 correct / 4 total


def test_compute_metrics_unknown_dataset_raises(tmp_path):
    """An unknown dataset name should raise ValueError, not silently misbehave."""
    path = tmp_path / "test_scored.csv"
    _write_indicqa_scored_csv(path, [])

    with pytest.raises(ValueError, match="Unknown dataset"):
        compute_metrics_for_file(path, dataset="not_a_real_dataset")


# --- build_results_table tests ---

def test_build_results_table_basic(tmp_path):
    """Building a table from 2 scored CSVs returns one row per metric."""
    _write_indicqa_scored_csv(
        tmp_path / "gemini_indicqa_scored.csv",
        [{"question": "Q", "reference_answer": "R", "model_answer": "M",
          "rouge_l": "0.5", "bleu": "0.4"}],
    )
    _write_himed_scored_csv(
        tmp_path / "gemini_himed_trad_scored.csv",
        [{"question": "Q", "reference_answer": "A", "model_answer": "A",
          "parsed_letter": "A", "is_correct": "1"}],
    )

    df = build_results_table(tmp_path)

    # 3 rows total: BLEU + ROUGE-L from indicqa, Accuracy from himed
    assert len(df) == 3
    assert set(df["metric"]) == {"BLEU", "ROUGE-L", "Accuracy"}
    assert set(df["model"]) == {"gemini"}
    # Schema check: these 4 columns must exist
    assert set(df.columns) == {"model", "dataset", "metric", "value"}


def test_build_results_table_warns_on_missing_files(tmp_path, capsys):
    """When expected (model, dataset) combos have no scored CSV, print warnings."""
    # Only write 1 of the expected files
    _write_indicqa_scored_csv(
        tmp_path / "gemini_indicqa_scored.csv",
        [{"question": "Q", "reference_answer": "R", "model_answer": "M",
          "rouge_l": "0.5", "bleu": "0.4"}],
    )

    build_results_table(tmp_path)

    captured = capsys.readouterr()
    # Should mention "missing" and at least one specific missing combo
    assert "missing" in captured.out.lower()
    assert "airavata_indicqa" in captured.out
