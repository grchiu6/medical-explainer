"""Tests for data/loader.py — written first (TDD)."""

import pytest

from data.loader import load_indicqa


EXPECTED_KEYS = {"question", "context", "reference_answer"}
FULL_ANSWERABLE_COUNT = 1052


def test_load_indicqa_returns_list_of_dicts():
    """Loading 3 items returns a list of dicts with required non-empty string fields."""
    items = load_indicqa(sample_size=3)

    assert isinstance(items, list), "Result should be a list"
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"

    for item in items:
        assert isinstance(item, dict), "Each item should be a dict"
        assert EXPECTED_KEYS == set(item.keys()), (
            f"Expected keys {EXPECTED_KEYS}, got {set(item.keys())}"
        )
        for key in EXPECTED_KEYS:
            assert isinstance(item[key], str), f"Field '{key}' should be a string"
            assert item[key].strip(), f"Field '{key}' should be non-empty"


def test_load_indicqa_sample_size():
    """Sampling 5 vs 10 items returns correct lengths."""
    five = load_indicqa(sample_size=5)
    ten = load_indicqa(sample_size=10)

    assert len(five) == 5, f"Expected 5 items, got {len(five)}"
    assert len(ten) == 10, f"Expected 10 items, got {len(ten)}"


def test_load_indicqa_filters_unanswerable():
    """All loaded items must have a non-empty reference_answer (unanswerable filtered out)."""
    items = load_indicqa()

    for item in items:
        assert item["reference_answer"].strip(), (
            f"Found item with empty reference_answer: question='{item['question'][:60]}'"
        )


def test_load_indicqa_full_load():
    """Full load returns approximately 1052 answerable items."""
    items = load_indicqa()

    assert len(items) == FULL_ANSWERABLE_COUNT, (
        f"Expected {FULL_ANSWERABLE_COUNT} answerable items, got {len(items)}"
    )
