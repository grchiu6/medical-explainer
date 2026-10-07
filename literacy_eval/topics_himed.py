"""HiMed-West question extraction for literacy-level evaluation.

WHAT THIS FILE DOES:
Extracts medical question stems from the HiMed-West_Bench.json dataset
and converts them into open-ended explanation topics that plug directly
into the existing literacy_eval pipeline.

WHY WE CONVERT MCQs TO OPEN-ENDED PROMPTS:
The HiMed dataset contains MCQ questions ("which of the following...").
For literacy-level evaluation, we want the models to EXPLAIN a medical
topic at a target register, not pick a multiple-choice answer. We strip
the MCQ options and rephrase the question stem as an explanation task.

This makes the two evaluation tracks directly comparable:
  - Curated track (topics.py): "मधुमेह क्या है?" → explain at level X
  - HiMed track (this file):   MCQ stem stripped → explain at level X

WHY MMLU-Pro-Health SPECIFICALLY:
HiMed-West_Bench.json contains 1784 questions from 3 sources:
  - GPQA-med:          249 — graduate-level, very niche
  - MMLU-Pro-Biology:  717 — molecular biology, not clinical
  - MMLU-Pro-Health:   818 — clinical/public health, most relevant

MMLU-Pro-Health questions cover topics like polio, viruses, public
health interventions — closer to the curated list's clinical focus.

NOTE ON DATA LOCATION:
The JSON file is read from the project root's data/himed/ directory,
which is relative to this file. On the HPC cluster, the outputs go to
scratch, but the DATA is read from wherever the project is checked out.
The data files are small (~3.5MB) so they live in the repo itself.

REPRODUCIBILITY:
A fixed random seed (SAMPLE_SEED = 42) is used for sampling so the
same 50 questions are always selected. Do not change this after the
initial run to keep results reproducible.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

# Path to the HiMed-West benchmark JSON, relative to the project root.
# Resolved from this file's location (literacy_eval/../data/himed/).
_DATA_PATH = Path(__file__).parent.parent / "data" / "himed" / "HiMed-West_Bench.json"

# Source within West_Bench to use.
# GPQA-med = graduate-level niche; MMLU-Pro-Biology = molecular bio;
# MMLU-Pro-Health = clinical/public health (best fit for this research).
_SOURCE_FILTER = "MMLU-Pro-Health"

# How many questions to sample. 50 matches the scale of the curated
# 30-topic track closely enough for comparison.
SAMPLE_SIZE = 50

# Fixed seed for reproducible sampling. Do not change after the first run.
SAMPLE_SEED = 42


# MCQ lead-in patterns in Hindi that indicate "which of the following"
# style questions. These are stripped/rewritten for open-ended prompts.
_MCQ_LEADINS = [
    r"^निम्नलिखित में से (?:कौन सा|किस|किसके|कौन|कितने|क्या|किसे|किन).*?\?",
    r"^नीचे दिए गए.*?में से.*?\?",
]


def _question_to_explanation_stem(question: str) -> str:
    """Convert a MCQ question stem into an open-ended explanation topic.

    MCQ questions use "which of the following" framing that doesn't work
    for open-ended explanations. This function converts them to a topic
    the model can explain.

    Strategy:
    1. If the question is already open-ended (ends with explanation cues),
       use it as-is.
    2. For "निम्नलिखित में से" style questions, extract the core topic
       phrase and rephrase it as "explain about [topic]".
    3. For all others, prepend a framing instruction so the model explains
       the underlying concept rather than answering the MCQ.

    Args:
        question: The raw MCQ question text from HiMed.

    Returns:
        A rephrased, open-ended explanation topic string in Hindi.
    """
    question = question.strip()

    # Check if it's already framed as an explanation/description question
    explanation_cues = [
        "क्या है", "कैसे होता", "कैसे काम", "बताइए", "समझाइए",
        "क्यों होता", "कैसे फैलता", "क्या होता",
    ]
    if any(cue in question for cue in explanation_cues):
        return question

    # For "निम्नलिखित में से" patterns, try to extract the core topic
    # e.g., "निम्नलिखित में से किसके द्वारा पोलियो का उन्मूलन किया जा सकता है?"
    # → "पोलियो का उन्मूलन कैसे किया जा सकता है?"
    for pattern in _MCQ_LEADINS:
        if re.match(pattern, question):
            # Remove the lead-in phrase up to the first content word
            # Replace "निम्नलिखित में से किस..." with direct topic
            stripped = re.sub(
                r"^निम्नलिखित में से (?:कौन सा|किस|किसके|कौन|कितने|क्या|किसे|किन)\s+",
                "",
                question,
                flags=re.IGNORECASE,
            ).strip()
            if stripped and not stripped.startswith("निम्नलिखित"):
                return stripped
            break

    # Fallback: wrap in an explanation frame so the model explains the
    # concept rather than treating it as a MCQ
    return f"निम्नलिखित विषय के बारे में विस्तार से समझाएं: {question}"


def _make_topic_id(question_number: int | str) -> str:
    """Create a stable topic ID from a HiMed question number.

    Prefixed with 'himed_' to avoid collisions with curated topic IDs.

    Args:
        question_number: The question_number field from HiMed.

    Returns:
        A short string like 'himed_042'.
    """
    return f"himed_{int(question_number):04d}"


def load_himed_topics(
    sample_size: int = SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> list[dict]:
    """Load and return HiMed-West questions as literacy-eval topic dicts.

    Samples `sample_size` questions from MMLU-Pro-Health source,
    converts each MCQ to an open-ended explanation topic, and returns
    them in the same format as curated topics.py.

    Each returned dict has:
        id:               short ID like 'himed_0042'
        topic_hi:         the explanation topic in Hindi (MCQ stem converted)
        topic_en:         "(HiMed) " + original question for reference
        category:         always "health" (MMLU-Pro-Health has no subject field)
        original_question: raw MCQ question text (for accuracy scoring later)
        options:          dict of MCQ options (A, B, C, ...) — for accuracy eval
        correct_answer:   the correct option letter (e.g., "E")
        source:           "MMLU-Pro-Health"

    Args:
        sample_size: How many questions to sample (default 50).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of topic dicts in literacy_eval pipeline format.

    Raises:
        FileNotFoundError: If HiMed-West_Bench.json doesn't exist.
        ValueError: If fewer questions exist than sample_size.
    """
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"HiMed dataset not found at {_DATA_PATH}. "
            "Run data/download.py to fetch the dataset, or check that the "
            "data/ directory is present in the project root."
        )

    with open(_DATA_PATH, encoding="utf-8") as f:
        all_items = json.load(f)

    # Filter to our chosen source
    source_items = [
        item for item in all_items
        if item.get("source") == _SOURCE_FILTER
    ]

    if len(source_items) < sample_size:
        raise ValueError(
            f"Only {len(source_items)} questions found for source "
            f"'{_SOURCE_FILTER}', but sample_size={sample_size}."
        )

    # Sample deterministically
    rng = random.Random(seed)
    sampled = rng.sample(source_items, sample_size)

    # Convert to topic dict format
    topics = []
    for item in sampled:
        q_number = item["question_number"]
        original_q = item["question"]
        options = item.get("options", {})
        correct_ans = item.get("answer", "")

        # Convert MCQ stem to open-ended explanation topic
        topic_hi = _question_to_explanation_stem(original_q)

        topics.append({
            # Core fields (same as curated topics)
            "id": _make_topic_id(q_number),
            "topic_hi": topic_hi,
            "topic_en": f"(HiMed-{_SOURCE_FILTER}) {original_q[:80]}",
            "category": "health",
            # Extra fields for accuracy evaluation
            "original_question": original_q,
            "options": options,
            "correct_answer": correct_ans,
            "source": _SOURCE_FILTER,
        })

    return topics


def preview_topics(n: int = 5) -> None:
    """Print the first N topics for inspection before running the full eval.

    Usage:
        python -c "from literacy_eval.topics_himed import preview_topics; preview_topics()"
    """
    topics = load_himed_topics()
    print(f"Loaded {len(topics)} HiMed topics (source: {_SOURCE_FILTER})\n")
    for i, t in enumerate(topics[:n], 1):
        print(f"[{i}] ID: {t['id']}")
        print(f"     Original: {t['original_question'][:100]}")
        print(f"     Converted: {t['topic_hi']}")
        print(f"     Correct answer: {t['correct_answer']}")
        opts_str = ", ".join(f"{k}: {v[:30]}" for k, v in list(t["options"].items())[:3])
        print(f"     Options (first 3): {opts_str}...")
        print()
