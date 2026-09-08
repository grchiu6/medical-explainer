"""Main evaluation runner — ties the whole pipeline together.

THIS IS THE SCRIPT YOU RUN. It connects the data loader, prompt builder,
model, and output saving into one workflow.

Usage:
    python evaluate.py --model gemini --dataset indicqa --sample 5
    python evaluate.py --model gemini --dataset himed_trad --sample 5
    python evaluate.py --model gemini --dataset himed_west
    python evaluate.py --model airavata --dataset himed_trad  # on HPC cluster

SUPPORTED COMBINATIONS:
    --dataset indicqa      → extractive QA  → score with BLEU + ROUGE-L  (score.py)
    --dataset himed_trad   → MCQ            → score with accuracy        (score_mcq.py)
    --dataset himed_west   → MCQ            → score with accuracy        (score_mcq.py)

KEY FEATURES:
- Incremental saving: Each answer is written to disk immediately.
  If the script crashes at question 500, questions 1-499 are safe.
- Resume support: If you re-run the script, it skips questions that
  already have answers in the CSV. No wasted API calls or GPU time.

ARCHITECTURE NOTE — "dispatch tables":
Instead of branching with if/elif on --dataset, we map dataset names
to (loader_fn, prompt_builder_fn) tuples. This pattern scales cleanly:
adding a new dataset = one line in the DATASETS dict, no if/elif to edit.
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path
from typing import Callable

from data.himed_loader import load_himed
from data.loader import load_indicqa
from models.base import build_mcq_prompt, build_prompt

# Where output CSVs go. Created automatically if it doesn't exist.
OUTPUTS_DIR = Path(__file__).parent / "outputs"

# The column names for our output CSV.
# We keep the same schema regardless of dataset — for MCQ, the
# 'context' column is just empty. Single-schema plumbing means the
# raw CSV shape is predictable for any downstream tool.
FIELDNAMES = ["question", "context", "reference_answer", "model_answer"]


def _load_indicqa_items(sample_size: int | None) -> list[dict]:
    """Load IndicQA items in a consistent dict shape for the runner."""
    return load_indicqa(sample_size=sample_size)


def _load_himed_items(variant: str, sample_size: int | None) -> list[dict]:
    """Load HiMed items and add an empty 'context' field for uniformity.

    The runner expects every item to have {question, context, reference_answer}.
    HiMed doesn't have context, so we add an empty string. The reference_answer
    for MCQ is the single letter (A-E).
    """
    raw = load_himed(variant=variant, sample_size=sample_size)
    return [
        {
            "question": item["question"],
            "context": "",
            "reference_answer": item["answer"],
        }
        for item in raw
    ]


def _build_extractive_prompt(item: dict) -> str:
    """Build prompt for extractive QA (IndicQA): question + context."""
    return build_prompt(item["question"], item["context"])


def _build_mcq_prompt(item: dict) -> str:
    """Build prompt for MCQ (HiMed): question with embedded options."""
    return build_mcq_prompt(item["question"])


# Dispatch table: dataset name → (loader, prompt_builder).
# Adding a new dataset = add one entry here. No other code changes.
# Each loader is a zero-arg lambda so callers don't have to know
# which kwargs each underlying loader takes.
DATASETS = {
    "indicqa": {
        "load": lambda sample_size: _load_indicqa_items(sample_size),
        "build_prompt": _build_extractive_prompt,
    },
    "himed_trad": {
        "load": lambda sample_size: _load_himed_items("trad", sample_size),
        "build_prompt": _build_mcq_prompt,
    },
    "himed_west": {
        "load": lambda sample_size: _load_himed_items("west", sample_size),
        "build_prompt": _build_mcq_prompt,
    },
}


def _load_existing_questions(output_path: Path) -> set[str]:
    """Load questions that were already answered in a previous run.

    Uses a set for O(1) "is this question already done?" lookups.

    Args:
        output_path: Path to an existing CSV file from a previous run.

    Returns:
        Set of question strings that are already answered.
    """
    if not output_path.exists():
        return set()

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["question"] for row in reader}


def run_evaluation(
    model_name: str,
    dataset_name: str,
    get_answer_fn: Callable[[str], str],
    sample_size: int | None,
    output_path: Path | None = None,
) -> None:
    """Run a model on a dataset and save results to CSV.

    Args:
        model_name: Name of the model (used in the output filename).
        dataset_name: Dataset key from DATASETS (indicqa, himed_trad, himed_west).
        get_answer_fn: Function that takes a prompt string and returns an
                       answer string. This is the model's get_answer().
        sample_size: Number of questions to evaluate, or None for all.
        output_path: Where to save results. If None, uses
                     outputs/{model_name}_{dataset_name}_raw.csv.
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid: {sorted(DATASETS.keys())}"
        )
    spec = DATASETS[dataset_name]

    if output_path is None:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        output_path = OUTPUTS_DIR / f"{model_name}_{dataset_name}_raw.csv"

    # Step 1: Load dataset using the right loader for this dataset
    data = spec["load"](sample_size)

    # Step 2: Check what's already been answered (resume support)
    already_done = _load_existing_questions(output_path)
    remaining = [item for item in data if item["question"] not in already_done]

    if not remaining:
        print(f"All {len(data)} questions already answered. Nothing to do.")
        return

    print(
        f"Dataset: {dataset_name}, Model: {model_name}\n"
        f"Total questions: {len(data)}, "
        f"already done: {len(already_done)}, "
        f"remaining: {len(remaining)}"
    )

    # Step 3: Open the CSV for appending
    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        # Step 4: Loop through each question
        for i, item in enumerate(remaining, 1):
            print(f"[{i}/{len(remaining)}] Asking {model_name}...")

            # Build the right prompt for this dataset's task
            prompt = spec["build_prompt"](item)

            # Call the model with the pre-built prompt
            answer = get_answer_fn(prompt)

            # Step 5: Save this answer immediately
            writer.writerow({
                "question": item["question"],
                "context": item["context"],
                "reference_answer": item["reference_answer"],
                "model_answer": answer,
            })

            # Force the write to disk so a crash doesn't lose this row
            f.flush()

    print(f"Done. Results saved to {output_path}")


def _get_model_function(model_name: str) -> Callable[[str], str]:
    """Dynamically import the get_answer function for a given model.

    We use importlib to import the module by name at runtime. This
    means adding a new model doesn't require changing evaluate.py —
    just create a new file in models/ and add its name to the choices.

    Args:
        model_name: The model name (matches the filename in models/).

    Returns:
        The get_answer function from that model's module.
    """
    module = importlib.import_module(f"models.{model_name}")
    return module.get_answer


def main():
    """Parse command-line arguments and run the evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model on a Hindi QA/MCQ dataset"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemini", "airavata", "apollo"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Which dataset to run the model on",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of questions to sample (default: all)",
    )
    args = parser.parse_args()

    get_answer_fn = _get_model_function(args.model)
    run_evaluation(
        model_name=args.model,
        dataset_name=args.dataset,
        get_answer_fn=get_answer_fn,
        sample_size=args.sample,
    )


if __name__ == "__main__":
    main()
