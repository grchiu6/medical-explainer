"""Stage 7: turn the 900 HiMed-Trad LLM responses into classifier input.

WHAT THIS FILE DOES:
  1. Reuses analysis/clean.py:build_clean_table() to extract the user-facing
     answer (HiMed-8B: only the text after "## Response") and to flag unusable
     responses (refusal, echo, degenerate, non-Hindi, too short).
  2. Records each response's share of Latin-script words BEFORE normalizing.
     normalize() removes English, so this is where English use is measured
     (the `low` prompt forbids it). It is reported, not classified.
  3. Applies the SAME normalize() used on the training corpus.
  4. Splits usable answers into windows of whole sentences of at most
     chunk.MAX_WORDS words. Most answers fit in one window; the classifier's
     score for a response is the average over its windows.

Writes data/register_corpus/llm_responses.csv (one row per response, all 900,
with usability and latin_share) and data/register_corpus/llm_windows.jsonl
(one row per window of each usable response).

WHY UNUSABLE RESPONSES ARE NOT SCORED: a register label for a refusal or a
repetition loop means nothing. Usability is reported as its own outcome.

    python -m literacy_eval.register.prepare_outputs
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from literacy_eval.analysis.clean import build_clean_table
from literacy_eval.register.build_corpus import CORPUS_DIR
from literacy_eval.register.chunk import MAX_WORDS, n_words, split_sentences
from literacy_eval.register.normalize import latin_share, normalize

RESPONSES_CSV = CORPUS_DIR / "llm_responses.csv"
WINDOWS_JSONL = CORPUS_DIR / "llm_windows.jsonl"
KEYS = ["model", "level", "item_id"]


def windows(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Greedy, deterministic packing of whole sentences into windows of at most `max_words` words.
    A single sentence longer than max_words becomes its own window (cut to max_words)."""
    out, current, size = [], [], 0
    for sentence in split_sentences(text):
        words = n_words(sentence)
        if words > max_words:
            if current:
                out.append(" ".join(current))
                current, size = [], 0
            out.append(" ".join(sentence.split()[:max_words]))
            continue
        if current and size + words > max_words:
            out.append(" ".join(current))
            current, size = [], 0
        current.append(sentence)
        size += words
    if current:
        out.append(" ".join(current))
    return out


def prepare(clean: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    responses = clean[KEYS + ["failure_type", "usable_rules", "answer"]].copy()
    responses["answer"] = responses["answer"].fillna("")
    responses["latin_share"] = responses["answer"].map(latin_share).round(3)
    responses["normalized"] = responses["answer"].map(normalize)
    responses["n_words"] = responses["normalized"].map(n_words)
    window_rows = []
    for row in responses[responses["usable_rules"]].itertuples():
        for i, text in enumerate(windows(row.normalized)):
            window_rows.append({"model": row.model, "level": row.level, "item_id": row.item_id,
                                "window": i, "text": text, "n_words": n_words(text)})
    responses["n_windows"] = responses[KEYS].apply(tuple, axis=1).map(
        pd.Series([tuple(r[k] for k in KEYS) for r in window_rows]).value_counts()).fillna(0).astype(int)
    return responses.drop(columns=["answer"]), window_rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    responses, window_rows = prepare(build_clean_table())
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    responses.to_csv(RESPONSES_CSV, index=False)
    with WINDOWS_JSONL.open("w", encoding="utf-8") as f:
        for row in window_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(responses.groupby(["model", "level"])["usable_rules"].sum().unstack().to_string())
    print(f"{int(responses['usable_rules'].sum())} usable responses -> {len(window_rows)} windows")
    print(f"wrote {RESPONSES_CSV} and {WINDOWS_JSONL}")


if __name__ == "__main__":
    main()
