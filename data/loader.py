"""Load IndicQA Hindi dataset into a standardized format for evaluation.

The dataset is in SQuAD format (a nested JSON structure). This module
flattens it into a simple list of dicts that the rest of the pipeline
can work with.

SQuAD format (what the file looks like):
    data -> articles -> paragraphs -> each has a context + list of QAs

Our output format (what the pipeline needs):
    [{"question": "...", "context": "...", "reference_answer": "..."}, ...]
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def _find_data_file() -> Path:
    """Find the IndicQA Hindi JSON file in the data/ directory.

    Returns:
        Path to indicqa_hi.json.

    Raises:
        FileNotFoundError: If the file hasn't been downloaded yet.
    """
    path = Path(__file__).parent / "indicqa" / "indicqa_hi.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Run 'python data/download.py' first to download it."
        )
    return path


def _flatten_squad_format(raw_data: dict) -> list[dict]:
    """Flatten nested SQuAD JSON into a flat list of QA dicts.

    The SQuAD format nests questions inside paragraphs inside articles.
    We flatten all of that into a simple list where each item is one
    question with its context and answer.

    Only includes questions that have actual answers (filters out
    'unanswerable' questions where the answer text is empty).

    Args:
        raw_data: The parsed JSON from indicqa_hi.json.

    Returns:
        List of dicts with keys: question, context, reference_answer.
    """
    items = []

    for article in raw_data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                # Skip unanswerable questions (empty answer text)
                if not qa["answers"] or not qa["answers"][0]["text"].strip():
                    continue

                items.append({
                    "question": qa["question"].strip(),
                    "context": context.strip(),
                    "reference_answer": qa["answers"][0]["text"].strip(),
                })

    return items


def load_indicqa(sample_size: int | None = None) -> list[dict]:
    """Load IndicQA Hindi dataset as a list of standardized dicts.

    Args:
        sample_size: If provided, randomly sample this many items.
                     Uses a fixed seed (42) for reproducibility — meaning
                     you get the same sample every time you run it.
                     If None, return all answerable items.

    Returns:
        List of dicts with keys: "question", "context", "reference_answer".
    """
    path = _find_data_file()

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    items = _flatten_squad_format(raw_data)

    if sample_size is not None:
        random.seed(42)
        items = random.sample(items, min(sample_size, len(items)))

    return items
