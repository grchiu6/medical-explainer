"""Score model outputs using BLEU and ROUGE-L metrics.

This script reads the raw output CSV from evaluate.py, compares
each model answer to the reference answer, and computes two scores:

BLEU: Measures n-gram overlap (how many words/phrases match)
ROUGE-L: Measures longest common subsequence (shared word order)

Both scores range from 0.0 (no match) to 1.0 (perfect match).

HINDI-SPECIFIC TOKENIZATION:
English tokenizers split text by spaces AND handle punctuation,
contractions, etc. For Hindi (Devanagari script), simple whitespace
splitting works well because Hindi words are naturally space-separated.
Using an English tokenizer on Hindi text would produce incorrect
token boundaries.

Usage:
    python score.py --model gemini
    python score.py --model airavata
    python score.py --model apollo
"""

import argparse
import csv
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# NOTE: We deliberately do NOT call nltk.download() here. We tokenize
# Hindi text ourselves (whitespace split) and pass pre-tokenized lists
# to sentence_bleu, so NLTK's punkt models are never used. Downloading
# them was unnecessary and broke on Python 3.14 + brewed Python (the
# downloader requires xml.expat which isn't built into that interpreter).

OUTPUTS_DIR = Path(__file__).parent / "outputs"


def _tokenize_hindi(text: str) -> list[str]:
    """Tokenize Hindi text by splitting on whitespace.

    Tokenization = splitting text into individual words (tokens).

    For Hindi (Devanagari script), whitespace splitting is appropriate
    because words are naturally separated by spaces. This is simpler
    than English tokenization, which needs special handling for
    contractions ("don't" → "do", "n't"), punctuation, etc.

    Example:
        "मधुमेह एक रोग है" → ["मधुमेह", "एक", "रोग", "है"]

    When you'd customize tokenization: When working with a language
    that has special rules. Chinese/Japanese don't use spaces between
    words, so they need character-level or special tokenizers. Hindi
    is straightforward — spaces work fine.

    Args:
        text: Hindi text string.

    Returns:
        List of word tokens.
    """
    return text.strip().split()


def _compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score between reference and model's answer.

    BLEU (Bilingual Evaluation Understudy) was originally designed
    for machine translation but is now used for any text generation task.

    It works by counting how many n-grams (word sequences) from the
    model's output appear in the reference:
    - 1-grams: individual words
    - 2-grams: pairs of consecutive words
    - 3-grams: triplets of consecutive words
    - 4-grams: four consecutive words

    The final score is a weighted combination of all n-gram matches.

    SmoothingFunction: When the model's answer has no matching n-grams
    at a certain level (e.g., no 4-gram matches), the raw BLEU score
    would be 0 regardless of other matches. Smoothing prevents this
    by adding a small value, giving partial credit.

    Args:
        reference: The correct answer.
        hypothesis: The model's answer.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    ref_tokens = _tokenize_hindi(reference)
    hyp_tokens = _tokenize_hindi(hypothesis)

    # Edge case: if either text is empty, score is 0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # sentence_bleu expects the reference as a LIST of lists
    # (because you can have multiple valid references — we have one)
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [ref_tokens],  # reference: list of one reference (which is a list of tokens)
        hyp_tokens,     # hypothesis: list of tokens
        smoothing_function=smoothie,
    )


class _HindiTokenizer:
    """Custom tokenizer for Hindi text used by ROUGE scorer.

    The default rouge_score tokenizer is designed for English and
    produces 0 tokens for Hindi (Devanagari) text, resulting in
    scores of 0.0 for everything. This custom tokenizer splits
    on whitespace, which works correctly for Hindi.

    Why a class and not just a function? The rouge_score library
    expects a tokenizer OBJECT with a .tokenize() METHOD. This is
    a common pattern called the "strategy pattern" — the library
    defines an interface (must have .tokenize()), and you provide
    any object that matches it.
    """

    def tokenize(self, text: str) -> list[str]:
        """Split Hindi text into words on whitespace."""
        return text.strip().split()


# Create one tokenizer instance to reuse (no need to create a new
# one for every scoring call)
_hindi_tokenizer = _HindiTokenizer()


def _compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score between reference and model's answer.

    ROUGE-L measures the Longest Common Subsequence (LCS) between
    two texts. LCS finds the longest sequence of words that appears
    in both texts in the same order (but not necessarily consecutive).

    Example:
        Reference: "मधुमेह एक चयापचय रोग है"
        Hypothesis: "मधुमेह एक रोग है"
        LCS: "मधुमेह एक रोग है" (4 words)

    The F1 score combines:
    - Precision: what fraction of the model's words are in the LCS?
    - Recall: what fraction of the reference's words are in the LCS?
    F1 = harmonic mean of precision and recall (balances both)

    We pass our custom _hindi_tokenizer because the default tokenizer
    doesn't handle Devanagari script (it returns 0 tokens → score of 0).
    use_stemmer=False because stemming is designed for English.

    Args:
        reference: The correct answer.
        hypothesis: The model's answer.

    Returns:
        ROUGE-L F1 score between 0.0 and 1.0.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False, tokenizer=_hindi_tokenizer
    )
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def score_model(
    raw_path: Path,
    scored_path: Path | None = None,
) -> dict:
    """Score a model's raw output CSV with BLEU and ROUGE-L.

    Reads the CSV from evaluate.py, computes per-question scores,
    saves them to a new CSV, and returns aggregate (mean) scores.

    Args:
        raw_path: Path to the raw output CSV (from evaluate.py).
        scored_path: Where to save per-question scores. If None,
                     replaces '_raw.csv' with '_scored.csv'.

    Returns:
        Dict with 'mean_rouge_l' and 'mean_bleu' (both floats).
    """
    if scored_path is None:
        scored_path = raw_path.parent / raw_path.name.replace("_raw.csv", "_scored.csv")

    # Read all the raw results
    with open(raw_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Compute scores for each question
    scored_fieldnames = [
        "question", "reference_answer", "model_answer", "rouge_l", "bleu",
    ]
    rouge_scores = []
    bleu_scores = []

    with open(scored_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
        writer.writeheader()

        for row in rows:
            ref = row["reference_answer"]
            hyp = row["model_answer"]

            rouge_l = _compute_rouge_l(ref, hyp)
            bleu = _compute_bleu(ref, hyp)

            rouge_scores.append(rouge_l)
            bleu_scores.append(bleu)

            writer.writerow({
                "question": row["question"],
                "reference_answer": ref,
                "model_answer": hyp,
                "rouge_l": f"{rouge_l:.4f}",
                "bleu": f"{bleu:.4f}",
            })

    # Compute mean (average) scores across all questions
    mean_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    mean_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    print(f"Scored {len(rows)} questions")
    print(f"  Mean ROUGE-L: {mean_rouge:.4f}")
    print(f"  Mean BLEU:    {mean_bleu:.4f}")
    print(f"  Saved to: {scored_path}")

    return {"mean_rouge_l": mean_rouge, "mean_bleu": mean_bleu}


def main():
    parser = argparse.ArgumentParser(description="Score IndicQA model outputs (BLEU + ROUGE-L)")
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemini", "airavata", "apollo"],
        help="Which model's output to score",
    )
    parser.add_argument(
        "--dataset",
        default="indicqa",
        help="Which dataset's raw CSV to score (default: indicqa). "
             "score.py is for extractive QA; use score_mcq.py for MCQ datasets.",
    )
    args = parser.parse_args()

    raw_path = OUTPUTS_DIR / f"{args.model}_{args.dataset}_raw.csv"
    if not raw_path.exists():
        print(f"Error: {raw_path} not found. Run evaluate.py first.")
        return

    score_model(raw_path)


if __name__ == "__main__":
    main()
