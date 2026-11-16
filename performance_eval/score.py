"""Two-mode answer scorer for MCQ performance conditions.

- "first_letter": non-CoT conditions, where the prompt asks for only the letter.
  Delegates to score_mcq.extract_letter.
- "after_marker": CoT conditions, where the model ends with ``Answer: X``
  (or the Hindi ``उत्तर: X``).

In both modes HiMed's "## Thinking ... ## Response ..." block is stripped first
and only the Response part is scored; Thinking with no Response (truncated
while thinking) scores as None.
"""

import re

from literacy_eval.analysis.clean import extract_answer as _split_thinking
from score_mcq import extract_letter as _extract_first_letter

_MARKER_RE = re.compile(r"(?:Answer|उत्तर)\s*[:：]?\s*([A-Ja-j])(?![A-Za-z])", re.IGNORECASE)
_LAST_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])")
_VALID_REFERENCE_RE = re.compile(r"[A-J]")
_OPTION_LINE_RE = re.compile(r"^[A-J]\. ", re.MULTILINE)


def _extract(text: str, mode: str) -> tuple[str | None, bool]:
    """Return (letter, marker_found). marker_found is only ever True in after_marker mode."""
    if mode not in ("first_letter", "after_marker"):
        raise ValueError(f"Unknown mode: {mode!r}")
    response, _thinking = _split_thinking(text)
    if not response:
        return None, False  # empty, or truncated while still thinking
    if mode == "first_letter":
        return _extract_first_letter(response), False
    marker_matches = list(_MARKER_RE.finditer(response))
    if marker_matches:
        # Last match, in case the model restates its answer.
        return marker_matches[-1].group(1).upper(), True
    letter_matches = list(_LAST_LETTER_RE.finditer(response))
    if letter_matches:
        return letter_matches[-1].group(1).upper(), False
    return None, False


def extract_answer(text: str, mode: str) -> str | None:
    return _extract(text, mode)[0]


def _n_options(row: dict) -> int | None:
    """Option count from an ``n_options`` key, else from the option lines; None if unknown."""
    if row.get("n_options") not in (None, ""):
        return int(row["n_options"])
    qwo = row.get("question_with_options")
    if qwo:
        return len(_OPTION_LINE_RE.findall(qwo)) or None
    return None


def score_rows(rows: list[dict], mode: str) -> list[dict]:
    """Score each row; unparseable answers count as incorrect, never skipped.

    Each input row needs item_id, reference_answer, model_answer. Output rows
    keep every input key and add parsed_letter (str | None), marker_found (bool)
    and is_correct (bool). A letter beyond the item's option count (from
    n_options or question_with_options) becomes None.
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
        parsed, marker_found = _extract(row["model_answer"], mode=mode)
        k = _n_options(row)
        if parsed is not None and k is not None and ord(parsed) - ord("A") >= k:
            parsed = None
        scored.append(
            {
                **row,
                "parsed_letter": parsed,
                "marker_found": marker_found,
                "is_correct": parsed is not None and parsed == reference,
            }
        )
    return scored
