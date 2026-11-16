"""Two-mode answer scorer for MCQ performance conditions.

- "first_letter": non-CoT conditions, where the prompt asks for only the letter.
  Delegates to score_mcq.extract_letter.
- "after_marker": CoT conditions, where the model ends with ``Answer: X``.
"""

import re

from score_mcq import extract_letter as _extract_first_letter

_MARKER_RE = re.compile(r"Answer\s*[:：]?\s*([A-Ja-j])(?![A-Za-z])", re.IGNORECASE)
_LAST_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])")
_VALID_REFERENCE_RE = re.compile(r"[A-J]")


def extract_answer(text: str, mode: str) -> str | None:
    if mode == "first_letter":
        return _extract_first_letter(text)
    if mode == "after_marker":
        marker_matches = list(_MARKER_RE.finditer(text))
        if marker_matches:
            # Last match, in case the model restates its answer.
            return marker_matches[-1].group(1).upper()
        letter_matches = list(_LAST_LETTER_RE.finditer(text))
        if letter_matches:
            return letter_matches[-1].group(1).upper()
        return None
    raise ValueError(f"Unknown mode: {mode!r}")


def score_rows(rows: list[dict], mode: str) -> list[dict]:
    """Score each row; unparseable answers count as incorrect, never skipped.

    Each input row needs item_id, reference_answer, model_answer. Output rows
    keep every input key and add parsed_letter (str | None) and is_correct (bool).
    Raises ValueError (naming the item) if a reference is not a single A-J letter.
    """
    scored = []
    for row in rows:
        reference = row["reference_answer"].strip().upper()
        if not _VALID_REFERENCE_RE.fullmatch(reference):
            raise ValueError(
                f"Invalid reference_answer for item {row['item_id']!r}: "
                f"{row['reference_answer']!r} is not a single letter A-J"
            )
        parsed = extract_answer(row["model_answer"], mode=mode)
        scored.append(
            {
                **row,
                "parsed_letter": parsed,
                "is_correct": parsed is not None and parsed == reference,
            }
        )
    return scored
