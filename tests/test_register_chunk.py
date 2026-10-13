"""Tests for literacy_eval/register/chunk.py — passages, de-duplication, splits, balance."""

import random

import pytest

from literacy_eval.register import chunk as C

S10 = "यह एक दस शब्दों वाला सरल वाक्य है परीक्षण करता।"   # exactly 10 words


def test_split_sentences_keeps_final_punctuation():
    assert C.split_sentences("पहला वाक्य है। क्या दूसरा है? तीसरा!") == ["पहला वाक्य है।", "क्या दूसरा है?", "तीसरा!"]


def test_split_sentences_ignores_empty_pieces():
    assert C.split_sentences("।। ? ") == []


def test_length_targets_are_clipped():
    assert C.length_targets([3, 50, 900]) == [C.MIN_WORDS, 50, C.MAX_WORDS]
    with pytest.raises(ValueError):
        C.length_targets([])


def test_pack_passages_follows_targets_and_keeps_whole_sentences():
    sentences = [S10] * 12
    passages = C.pack_passages(sentences, targets=[30], rng=random.Random(0))
    assert [C.n_words(p) for p in passages] == [30, 30, 30, 30]
    assert all(p.endswith("।") for p in passages)


def test_pack_passages_drops_short_remainder_and_overlong_sentences():
    overlong = " ".join(["शब्द"] * (C.MAX_WORDS + 1)) + "।"
    passages = C.pack_passages([S10, S10, S10, overlong, S10], targets=[30], rng=random.Random(0))
    assert [C.n_words(p) for p in passages] == [30]       # 4th S10 alone (10 words) < MIN_WORDS


def test_dedupe_sentences_across_calls_ignores_punctuation_but_keeps_short_repeats():
    seen: set[str] = set()
    assert C.dedupe_sentences([S10, "यह सही है।"], seen) == [S10, "यह सही है।"]
    assert C.dedupe_sentences([S10.replace("।", "?"), "यह सही है।"], seen) == ["यह सही है।"]


def test_split_groups_never_puts_a_group_in_two_splits_and_fills_test_and_val():
    sizes = {f"g{i}": 10 for i in range(20)}
    assignment = C.split_groups(sizes, random.Random(1))
    assert set(assignment) == set(sizes)
    counts = {s: sum(sizes[g] for g, a in assignment.items() if a == s) for s in ("train", "val", "test")}
    assert counts["test"] == 30 and counts["val"] == 30 and counts["train"] == 140


def test_split_groups_with_three_groups_gives_one_each():
    assignment = C.split_groups({"a": 5, "b": 5, "c": 5}, random.Random(0))
    assert sorted(assignment.values()) == ["test", "train", "val"]


def test_balance_downsamples_to_smallest_class_round_robin_across_groups():
    big = [{"group": f"issue{i % 5}", "id": i} for i in range(100)]
    small = [{"group": "ch1", "id": i} for i in range(10)]
    out = C.balance({"professional": big, "grade5": small}, random.Random(0))
    assert len(out["professional"]) == len(out["grade5"]) == 10
    assert {p["group"] for p in out["professional"]} == {f"issue{i}" for i in range(5)}   # spread over all issues
