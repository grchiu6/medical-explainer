"""Stage 4 of the register corpus: sentences -> passages -> de-duplicate -> split -> balance.

WHAT THIS FILE DOES (all functions are pure, so they can be unit-tested):
  split_sentences   normalized text -> sentences, KEEPING the final ।/?/!
                    (analysis/features.py:split_sentences drops them; a
                    passage should read like real text)
  length_targets    the passage-length distribution, taken from the usable
                    LLM outputs and clipped to [MIN_WORDS, MAX_WORDS]
  pack_passages     pack whole sentences of ONE group (chapter/issue) into
                    passages whose word counts follow length_targets
  dedupe_sentences  drop sentences already seen anywhere in the corpus
  split_groups      assign whole groups to train/val/test (~70/15/15)
  balance           down-sample every class to the smallest, round-robin
                    across groups

WHY MATCH LENGTHS: if professional passages were longer than 5th-grade ones,
"long = professional" would be a free shortcut. Drawing every class's
passage lengths from the SAME distribution removes it. Sentence length, a
real register signal, is left alone.

WHY SPLIT BY GROUP: passages from one chapter share vocabulary and topic.
With one chapter on both sides of the split, test accuracy would be inflated.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict

MIN_WORDS = 20      # shorter passages carry too little signal
MAX_WORDS = 200     # longer LLM outputs are scored in windows of this size
SPLIT_SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}

_SENTENCE = re.compile(r"[^।?!]+[।?!]?")


def split_sentences(text: str) -> list[str]:
    """Split normalized text into sentences, keeping each sentence's final punctuation."""
    return [s.strip() for s in _SENTENCE.findall(text or "") if s.strip(" ।?!")]


def n_words(text: str) -> int:
    return len(text.split())


def length_targets(output_lengths: list[int]) -> list[int]:
    """The passage-length distribution: LLM-output word counts clipped to [MIN_WORDS, MAX_WORDS]."""
    clipped = [min(max(n, MIN_WORDS), MAX_WORDS) for n in output_lengths if n > 0]
    if not clipped:
        raise ValueError("no output lengths to build targets from")
    return clipped


def pack_passages(sentences: list[str], targets: list[int], rng: random.Random) -> list[str]:
    """Pack consecutive whole sentences into passages with word counts drawn from `targets`.

    A passage closes once it reaches its target. A sentence that alone exceeds MAX_WORDS is skipped
    (usually OCR run-on text). A trailing remainder shorter than MIN_WORDS is dropped.
    """
    passages, current, size = [], [], 0
    target = rng.choice(targets)
    for sentence in sentences:
        words = n_words(sentence)
        if words > MAX_WORDS:
            continue
        if current and size + words > MAX_WORDS:
            passages.append(" ".join(current))
            current, size, target = [], 0, rng.choice(targets)
        current.append(sentence)
        size += words
        if size >= target:
            passages.append(" ".join(current))
            current, size, target = [], 0, rng.choice(targets)
    if current and size >= MIN_WORDS:
        passages.append(" ".join(current))
    return [p for p in passages if n_words(p) >= MIN_WORDS]


def _key(sentence: str) -> str:
    return hashlib.md5(re.sub(r"[।?!\s]+", " ", sentence).strip().encode()).hexdigest()


def dedupe_sentences(sentences: list[str], seen: set[str]) -> list[str]:
    """Drop sentences whose text was already seen (updates `seen`). Very short sentences are kept:
    repeats like "यह सही है।" are ordinary language, not duplicated documents."""
    kept = []
    for sentence in sentences:
        if n_words(sentence) < 5:
            kept.append(sentence)
            continue
        key = _key(sentence)
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
    return kept


def split_groups(group_sizes: dict[str, int], rng: random.Random) -> dict[str, str]:
    """Assign each group to train/val/test so each split gets ~its share of PASSAGES.

    Groups are shuffled, then val and test are filled first, so they always receive at least one
    group when there are 3 or more groups.
    """
    groups = list(group_sizes)
    rng.shuffle(groups)
    total = sum(group_sizes.values())
    assignment: dict[str, str] = {}
    filled = {name: 0 for name in SPLIT_SHARES}
    for split in ("test", "val"):
        for group in list(groups):
            if filled[split] >= SPLIT_SHARES[split] * total or len(groups) <= 1:
                break
            assignment[group] = split
            filled[split] += group_sizes[group]
            groups.remove(group)
    for group in groups:
        assignment[group] = "train"
    return assignment


def balance(passages_by_class: dict[str, list[dict]], rng: random.Random) -> dict[str, list[dict]]:
    """Down-sample every class to the smallest class's size, taking passages round-robin across
    groups so the kept passages come from as many chapters/issues as possible."""
    n = min(len(p) for p in passages_by_class.values())
    balanced = {}
    for label, passages in passages_by_class.items():
        by_group: dict[str, list[dict]] = defaultdict(list)
        for p in passages:
            by_group[p["group"]].append(p)
        queues = list(by_group.values())
        for q in queues:
            rng.shuffle(q)
        rng.shuffle(queues)
        picked: list[dict] = []
        while len(picked) < n:
            for q in queues:
                if q and len(picked) < n:
                    picked.append(q.pop())
        balanced[label] = picked
    return balanced
