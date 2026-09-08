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

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Match a single A-J letter as a "standalone token":
# - (?<![A-Za-z])  = NOT preceded by another letter (so 'Apollo' won't match)
# - [A-Ja-j]       = the letter itself, A-J case-insensitive
# - (?![A-Za-z])   = NOT followed by another letter (same reason)
#
# A-J covers all datasets:
#   HiMed-Trad: options A-E (5 options)
#   HiMed-West: options A-D up to A-J (3-10 options)
_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])")

# Hindi phonetic spellings of Latin option letters.
# Airavata (and potentially other Hindi-tuned models) writes the letter
# of its chosen option in Devanagari script rather than Latin.
#
# Confirmed from actual model output (seen in scored CSVs):
#   ए  → A  (sounds like "ay")       बी → B  (sounds like "bee")
#   सी → C  (sounds like "see")      डी → D  (sounds like "dee")
#   ई  → E  (sounds like "ee")       एफ → F  (sounds like "ef")
#   जी → G  (sounds like "jee")      एच → H  (sounds like "aitch")
#   आई → I  (sounds like "eye")      जे → J  (sounds like "jay")
#
# We check these BEFORE the regex so a response of just "ए" or "ए."
# is correctly mapped to 'A' rather than returning None.
# The model may also include the full option text after the letter
# (e.g. "ए. वात रोग") — we handle that by checking startswith.
_HINDI_LETTER_MAP: dict[str, str] = {
    # A–E: seen in HiMed-Trad and West (Airavata uses these extensively)
    "ए": "A",  "ए.": "A",
    "बी": "B", "बी.": "B",
    "सी": "C", "सी.": "C",
    "डी": "D", "डी.": "D",
    "ई": "E",  "ई.": "E",
    # F–J: seen in HiMed-West answers (confirmed from scored CSV data)
    "एफ": "F", "एफ.": "F",
    "जी": "G", "जी.": "G",
    "एच": "H", "एच.": "H",
    "आई": "I", "आई.": "I",
    "जे": "J", "जे.": "J",
}


def extract_letter(text: str) -> str | None:
    """Extract the first standalone A-J letter from model output.

    Handles both Latin and Hindi phonetic letter representations:
        "A"            → "A"
        "a"            → "A"    (case-normalized)
        "A."           → "A"
        "ए"            → "A"    (Hindi phonetic for 'A')
        "ए. वात रोग"   → "A"    (Hindi letter + option text)
        "बी"           → "B"    (Hindi phonetic for 'B')
        "डी."          → "D"    (Hindi phonetic for 'D' with period)
        "उत्तर: B है"   → "B"    (Latin letter inside Hindi sentence)
        "Apollo says C" → "C"   (skips 'A' inside 'Apollo')
        ""             → None   (nothing to extract)
        "K"            → None   (outside A-J)

    Args:
        text: Raw model output.

    Returns:
        Uppercase letter 'A'-'J', or None if no valid letter found.
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()

    # Step 1: Check for a Hindi phonetic letter (exact match or as a prefix
    # followed by a space/period — e.g. "ए. वात रोग" starts with "ए.").
    for hindi, latin in _HINDI_LETTER_MAP.items():
        if stripped == hindi or stripped.startswith(hindi + " ") or stripped.startswith(hindi[0] + "."):
            return latin

    # Step 2: Fall back to Latin letter regex.
    match = _LETTER_RE.search(stripped)
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

    # Cyrillic-to-Latin normalization table (same as in himed_loader).
    # Handles reference_answer fields in already-generated raw CSVs that
    # may contain Cyrillic lookalike characters (e.g. U+0415 'Е' vs U+0045 'E').
    _CYRILLIC_TO_LATIN = str.maketrans("АВСЕОРХУКІМВТ", "ABCEOPXYKIMBt")

    scored_fieldnames = [
        "question", "reference_answer", "model_answer",
        "parsed_letter", "is_correct", "skipped",
    ]
    n_correct = 0
    n_skipped = 0

    with open(scored_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
        writer.writeheader()

        for row in rows:
            # Normalize the reference answer: strip, Cyrillic→Latin, uppercase.
            ref = row["reference_answer"].strip().translate(_CYRILLIC_TO_LATIN).upper()
            raw = row["model_answer"]

            # Skip rows where the reference_answer isn't a single valid letter.
            # These are dataset errors (non-MCQ entries) that can't be scored.
            if len(ref) != 1 or ref not in "ABCDEFGHIJ":
                n_skipped += 1
                writer.writerow({
                    "question": row["question"],
                    "reference_answer": ref,
                    "model_answer": raw,
                    "parsed_letter": "",
                    "is_correct": "",   # blank = not scored, not wrong
                    "skipped": "1",
                })
                continue

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
                "skipped": "0",
            })

    if n_skipped:
        print(f"  Skipped {n_skipped} row(s) with invalid reference answers (dataset errors).")

    n_total = len(rows) - n_skipped
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
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Folder containing raw CSVs (and where scored CSVs will be saved). "
             "Default: ./outputs/. Use this to point at a dated subfolder, "
             "e.g. --outputs-dir outputs/9:4:2026",
    )
    # Note: HiMed-West has up to 10 options (A-J). The random baseline
    # is NOT 1/5 = 20% for West — it varies by question (1/3 to 1/10).
    # Weighted average random baseline for West ≈ 13%.
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    raw_path = outputs_dir / f"{args.model}_{args.dataset}_raw.csv"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run evaluate.py first.")
        return

    # scored CSV goes to the same folder, replacing any existing file
    scored_path = outputs_dir / f"{args.model}_{args.dataset}_scored.csv"
    score_mcq_model(raw_path, scored_path=scored_path)


if __name__ == "__main__":
    main()
