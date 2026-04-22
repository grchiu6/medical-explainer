"""Score MCQ model outputs using accuracy.

For multiple-choice questions, the right metric is simple accuracy:
    accuracy = (# questions the model got right) / (# total questions)

Unlike free-form text (where we used BLEU/ROUGE-L), MCQ answers are
categorical — one of {A, B, C, D, E}. Text-similarity metrics are
meaningless here. A model that answers "B" when the truth is "A" isn't
"80% right" — it's wrong. Period.

RANDOM BASELINE:
For 5-option MCQ, random guessing yields ~20% accuracy. A useful
reference point: if your model scores below 20%, it's actively being
misled (worse than noise). A well-aligned medical LLM should
reach 50%+ on Hindi medical benchmarks.

EXTRACTION STEP:
Models rarely output exactly "A". They output "A.", "उत्तर A है",
or a full paragraph. extract_letter() grabs the first standalone A-E
letter from the output — this is the standard approach used in MMLU
and similar MCQ benchmarks.

Usage:
    python score_mcq.py --model gemini --dataset himed_trad
"""

import argparse
import csv
import re
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Match a single A-E letter as a "standalone token":
# - (?<![A-Za-z])  = NOT preceded by another letter (so 'Apollo' won't match)
# - [A-Ea-e]       = the letter itself, A-E case-insensitive
# - (?![A-Za-z])   = NOT followed by another letter (same reason)
#
# Lookarounds (?<!...) and (?!...) are zero-width — they assert
# "the character on either side is NOT a letter" without consuming any
# characters. This is how we implement word-boundary matching that works
# even adjacent to Hindi characters, punctuation, or digits.
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])")


def extract_letter(text: str) -> str | None:
    """Extract the first standalone A-E letter from model output.

    Handles real-world messiness:
        "A"            → "A"
        "a"            → "A"    (case-normalized)
        "A."           → "A"
        "उत्तर: B है"   → "B"
        "Apollo says C" → "C"   (skips 'A' inside 'Apollo')
        ""             → None   (nothing to extract)
        "F"            → None   (outside A-E)

    Args:
        text: Raw model output.

    Returns:
        Uppercase letter 'A'-'E', or None if no valid letter found.
    """
    if not text or not text.strip():
        return None

    match = _LETTER_RE.search(text)
    if match is None:
        return None
    return match.group(1).upper()


def score_mcq_model(
    raw_path: Path,
    scored_path: Path | None = None,
) -> dict:
    """Score a model's raw MCQ output CSV using accuracy.

    Reads the raw CSV from evaluate.py, extracts the model's letter
    from each answer, compares to the reference, and writes a scored
    CSV with per-question outcomes.

    Args:
        raw_path: Path to raw output CSV.
        scored_path: Where to save per-question scores. If None,
                     replaces '_raw.csv' with '_scored.csv'.

    Returns:
        Dict with 'accuracy' (float 0.0-1.0), 'n_correct' (int), 'n_total' (int).
    """
    if scored_path is None:
        scored_path = raw_path.parent / raw_path.name.replace("_raw.csv", "_scored.csv")

    with open(raw_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scored_fieldnames = [
        "question", "reference_answer", "model_answer",
        "parsed_letter", "is_correct",
    ]
    n_correct = 0

    with open(scored_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
        writer.writeheader()

        for row in rows:
            ref = row["reference_answer"].strip().upper()
            raw = row["model_answer"]
            parsed = extract_letter(raw)

            is_correct = (parsed == ref)
            if is_correct:
                n_correct += 1

            writer.writerow({
                "question": row["question"],
                "reference_answer": ref,
                "model_answer": raw,
                "parsed_letter": parsed if parsed is not None else "",
                "is_correct": "1" if is_correct else "0",
            })

    n_total = len(rows)
    accuracy = n_correct / n_total if n_total else 0.0

    print(f"Scored {n_total} MCQ questions")
    print(f"  Correct:  {n_correct}/{n_total}")
    print(f"  Accuracy: {accuracy:.4f}  (random baseline for 5-option: 0.20)")
    print(f"  Saved to: {scored_path}")

    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": n_total}


def main():
    parser = argparse.ArgumentParser(description="Score MCQ model outputs")
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemini", "airavata", "apollo"],
        help="Which model's output to score",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["himed_trad", "himed_west"],
        help="Which MCQ dataset was evaluated",
    )
    args = parser.parse_args()

    raw_path = OUTPUTS_DIR / f"{args.model}_{args.dataset}_raw.csv"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run evaluate.py first.")
        return

    score_mcq_model(raw_path)


if __name__ == "__main__":
    main()
