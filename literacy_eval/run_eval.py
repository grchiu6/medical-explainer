"""Literacy-level evaluation runner — collect model outputs at varying registers.

THIS IS THE SCRIPT YOU RUN. It ties together topics, prompts, and model
wrappers to collect Hindi medical explanations at different literacy levels.

Usage:
    # Single level, small sample (for testing / smoke test)
    python -m literacy_eval.run_eval --model airavata --level low --sample 1 --topic diabetes

    # All levels for one model, outputs to HPC scratch
    python -m literacy_eval.run_eval --model airavata --level all \
        --output-dir /home/hice1/gchiu6/scratch/literacy_eval_outputs

    # Single topic (for quick testing)
    python -m literacy_eval.run_eval --model airavata --level low --topic diabetes

    # Dry run — print prompts without calling the model
    python -m literacy_eval.run_eval --model airavata --level low --sample 2 --dry-run

ARCHITECTURE:
    - Topics come from literacy_eval/topics.py (curated list of 30)
    - Prompts come from literacy_eval/prompts.py (3 literacy levels)
    - Model wrappers come from models/ (reuses the existing pipeline)
    - Outputs go to --output-dir/{model}_{level}_raw.csv
      (defaults to literacy_eval/outputs/ if not specified)

INCREMENTAL SAVING + RESUME:
    Same pattern as the original evaluate.py — each response is written
    to disk immediately, and re-running skips already-answered topics.
    If the script crashes at topic 15, topics 1-14 are safe.

SUPPORTED MODELS:
    airavata, apollo, nanda, mmed_llama, himed
    (Gemini excluded per research design decision)
"""

from __future__ import annotations

import argparse
import csv
import importlib
from pathlib import Path

from literacy_eval.prompts import VALID_LEVELS, build_explanation_prompt
from literacy_eval.topics import load_topics
from literacy_eval.topics_himed import load_himed_topics
from literacy_eval.topics_himed_trad import load_himed_trad_topics

# Default output directory when --output-dir is not specified.
# On the HPC cluster, always pass --output-dir pointing at scratch.
DEFAULT_OUTPUTS_DIR = Path(__file__).parent / "outputs"

# CSV column names for the output file.
# topic_id:        short identifier (e.g., "diabetes" or "himed_0042")
# topic_hi:        the Hindi question sent to the model
# topic_en:        English translation (for our reference)
# category:        medical category (for analysis grouping)
# literacy_level:  which level was requested (low/mid/high)
# prompt:          the full prompt that was sent to the model
# model_response:  what the model returned
# --- HiMed-specific columns (empty for curated topics) ---
# original_question: raw MCQ question text from HiMed
# options_json:    MCQ options serialized as JSON string
# correct_answer:  correct MCQ option letter (e.g., "E")
# topic_source:    'curated' or 'himed'
FIELDNAMES = [
    "topic_id",
    "topic_hi",
    "topic_en",
    "category",
    "literacy_level",
    "topic_source",
    "prompt",
    "model_response",
    # HiMed accuracy columns — empty string for curated topics
    "original_question",
    "options_json",
    "correct_answer",
]


def _load_existing_topic_ids(output_path: Path) -> set[str]:
    """Load topic IDs that already have responses from a previous run.

    Used for resume support — if we already got a response for "diabetes"
    at the "low" level, we skip it on re-run.

    Args:
        output_path: Path to an existing CSV file from a previous run.

    Returns:
        Set of topic_id strings that already have responses.
    """
    if not output_path.exists():
        return set()

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["topic_id"] for row in reader}


def _get_model_function(model_name: str):
    """Dynamically import the get_answer function from models/{model_name}.py.

    Reuses the existing model wrappers from the original pipeline.
    The literacy eval only changes the PROMPT, not the model plumbing.

    Args:
        model_name: Must match a file in models/ (e.g., "airavata").

    Returns:
        The get_answer(prompt: str) -> str function from that model.
    """
    module = importlib.import_module(f"models.{model_name}")
    return module.get_answer


def run_literacy_eval(
    model_name: str,
    literacy_level: str,
    topic_source: str = "curated",
    sample_size: int | None = None,
    topic_id: str | None = None,
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Run the literacy-level evaluation for one model at one level.

    Args:
        model_name: Which model to use (e.g., "airavata", "apollo").
        literacy_level: Target level ("low", "mid", or "high").
        topic_source: Either 'curated' (30 hand-picked topics from topics.py)
                      or 'himed' (50 MMLU-Pro-Health questions from HiMed
                      converted to open-ended explanation topics).
        sample_size: Number of topics to evaluate, or None for all.
        topic_id: If provided, only run this single topic (for testing).
        dry_run: If True, print prompts but don't call the model.
        output_dir: Directory to write CSVs to. Defaults to
                    DEFAULT_OUTPUTS_DIR. On the HPC cluster, pass the
                    scratch path so large outputs don't fill $HOME.
    """
    # Load topics from the appropriate source
    if topic_source == "himed":
        topics = load_himed_topics()
        if sample_size is not None:
            topics = topics[:sample_size]
    elif topic_source == "himed_trad":
        topics = load_himed_trad_topics()
        if sample_size is not None:
            topics = topics[:sample_size]
    else:
        topics = load_topics(sample_size=sample_size)

    # Filter to a single topic if requested
    if topic_id is not None:
        topics = [t for t in topics if t["id"] == topic_id]
        if not topics:
            raise ValueError(
                f"Unknown topic '{topic_id}'. "
                f"For curated topics check literacy_eval/topics.py; "
                f"for himed topics use the himed_XXXX ID format; "
                f"for himed_trad topics use the trad_XXXX ID format."
            )

    # Set up output path — use scratch dir on cluster, local dir by default
    # Each topic source gets its own suffix so outputs don't overwrite each other.
    outputs_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if topic_source == "himed":
        source_suffix = "_himed"
    elif topic_source == "himed_trad":
        source_suffix = "_himed_trad"
    else:
        source_suffix = ""
    output_path = outputs_dir / f"{model_name}_{literacy_level}{source_suffix}_raw.csv"

    # Check what's already done (resume support)
    already_done = _load_existing_topic_ids(output_path)
    remaining = [t for t in topics if t["id"] not in already_done]

    if not remaining:
        print(f"All {len(topics)} topics already answered for "
              f"{model_name} at {literacy_level} level. Nothing to do.")
        return

    print(
        f"Model: {model_name} | Level: {literacy_level}\n"
        f"Total topics: {len(topics)}, "
        f"already done: {len(already_done)}, "
        f"remaining: {len(remaining)}"
    )

    if dry_run:
        # Just print the prompts — useful for verifying prompt quality
        for i, topic in enumerate(remaining, 1):
            prompt = build_explanation_prompt(topic["topic_hi"], literacy_level)
            print(f"\n{'='*60}")
            print(f"[{i}/{len(remaining)}] Topic: {topic['id']} "
                  f"({topic['topic_en']})")
            print(f"{'='*60}")
            print(prompt)
        print(f"\n(Dry run — no model calls made)")
        return

    # Load the model's get_answer function
    get_answer_fn = _get_model_function(model_name)

    # Open CSV for appending
    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        for i, topic in enumerate(remaining, 1):
            prompt = build_explanation_prompt(topic["topic_hi"], literacy_level)

            print(f"[{i}/{len(remaining)}] Asking {model_name} about "
                  f"'{topic['id']}' at {literacy_level} level...")

            # Call the model
            response = get_answer_fn(prompt)

            # Save immediately
            import json as _json
            writer.writerow({
                "topic_id": topic["id"],
                "topic_hi": topic["topic_hi"],
                "topic_en": topic["topic_en"],
                "category": topic["category"],
                "literacy_level": literacy_level,
                "topic_source": topic_source,
                "prompt": prompt,
                "model_response": response,
                # HiMed accuracy fields — empty for curated topics
                "original_question": topic.get("original_question", ""),
                "options_json": _json.dumps(topic.get("options", {}), ensure_ascii=False),
                "correct_answer": topic.get("correct_answer", ""),
            })

            # Force write to disk
            f.flush()

            print(f"  → Got {len(response)} chars")

    print(f"\nDone. Results saved to {output_path}")


def main():
    """Parse CLI arguments and run the literacy-level evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a model's ability to adapt Hindi medical "
                    "explanations to different literacy levels."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["airavata", "apollo", "nanda", "mmed_llama", "himed"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=VALID_LEVELS + ["all"],
        help="Target literacy level, or 'all' to run all 3 levels",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of topics to evaluate (default: all 30)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Run a single topic by ID (e.g., 'diabetes'). For testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the model (for verification)",
    )
    parser.add_argument(
        "--topic-source",
        choices=["curated", "himed", "himed_trad"],
        default="curated",
        help="Topic source: 'curated' (30 hand-picked topics, default), "
             "'himed' (50 MMLU-Pro-Health questions from HiMed-West_Bench), or "
             "'himed_trad' (60 reviewed HiMed-Trad questions with ground-truth CoT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output CSVs (default: literacy_eval/outputs/). "
             "On the HPC cluster, set this to your scratch path, e.g. "
             "/home/hice1/gchiu6/scratch/literacy_eval_outputs",
    )
    args = parser.parse_args()

    # Determine which levels to run
    if args.level == "all":
        levels = VALID_LEVELS
    else:
        levels = [args.level]

    # Run evaluation for each level
    for level in levels:
        print(f"\n{'#'*60}")
        print(f"# Starting level: {level}")
        print(f"{'#'*60}\n")

        run_literacy_eval(
            model_name=args.model,
            literacy_level=level,
            topic_source=args.topic_source,
            sample_size=args.sample,
            topic_id=args.topic,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
