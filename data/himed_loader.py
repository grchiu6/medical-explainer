"""Load the HiMed Hindi medical MCQ dataset into a standardized format.

HiMed structure (from Kaggle):
    A flat JSON ARRAY (not nested like SQuAD) where each entry is one
    multiple-choice question with the options baked into the question text
    and the answer as a single letter.

Two benchmark variants we support:
    - 'trad' → HiMed-Trad_Bench.json (Traditional medicine: Ayurveda etc.)
    - 'west' → HiMed-West_Bench.json (Western/modern medicine)

Output shape (what the rest of the pipeline consumes):
    [
        {
            "question": "...full text incl. options A-E...",
            "answer":   "A",     # single letter
            "subject":  "etiology",
            "system":   "Ayurveda",
            "source":   "HiMed-trad",
        },
        ...
    ]

We intentionally drop the 'cot' (chain-of-thought) and 'question_number'
fields — they aren't needed for accuracy scoring. If you want to use CoT
later for reasoning evaluation, add it back here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Where HiMed JSON files live. Tests can override this via monkeypatch.
HIMED_DIR = Path(__file__).parent / "himed"

# Map short variant names to actual filenames. Adding a new variant
# (e.g., the West_Exam file) is a one-line change here — no need to
# modify any caller code.
_VARIANT_FILES = {
    "trad": "HiMed-Trad_Bench.json",
    "west": "HiMed-West_Bench.json",
}


def _resolve_path(variant: str | None, file_path: Path | None) -> Path:
    """Figure out which file to load.

    Two ways to call load_himed():
        1. load_himed(variant="trad")          → look up filename in HIMED_DIR
        2. load_himed(file_path=Path("..."))   → use exact path (mainly for tests)

    Args:
        variant: Short name like 'trad' or 'west'.
        file_path: Direct path to a JSON file.

    Returns:
        Path to the JSON file to load.

    Raises:
        ValueError: If neither or both are given, or unknown variant.
        FileNotFoundError: If the resolved path doesn't exist.
    """
    if (variant is None) == (file_path is None):
        raise ValueError("Provide exactly one of variant= or file_path=")

    if file_path is not None:
        path = file_path
    else:
        if variant not in _VARIANT_FILES:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Valid: {sorted(_VARIANT_FILES.keys())}"
            )
        path = HIMED_DIR / _VARIANT_FILES[variant]

    if not path.exists():
        raise FileNotFoundError(
            f"HiMed file not found at {path}. "
            "Download it from Kaggle and place in data/himed/. "
            "Run 'python data/download.py' for instructions."
        )
    return path


def _format_question(raw: dict) -> str:
    """Build the question text with options visible to the model.

    HiMed has TWO schemas:
    - Trad: options are already embedded in the 'question' string
            (e.g. 'Q?\\nA. ...\\nB. ...').
    - West: options live in a separate 'options' dict
            (e.g. {'A': '...', 'B': '...'}). The 'question' is just
            the stem with no options.

    We detect which case we're in and produce a single string that
    always contains both the question and the options. This is what
    the model sees in the MCQ prompt.
    """
    question = raw["question"].strip()
    options = raw.get("options")

    if not options:
        # Trad-style: options already embedded — use as-is.
        return question

    # West-style: append 'A. ...\nB. ...' to the question stem.
    options_text = "\n".join(
        f"{letter}. {text}" for letter, text in sorted(options.items())
    )
    return f"{question}\n{options_text}"

# Cyrillic letters that look identical to Latin letters but are different
# Unicode code points. Found in HiMed-Trad (e.g., Cyrillic 'Е' U+0415
# instead of Latin 'E' U+0045). A model always outputs Latin letters,
# so without this normalization the scorer would mark the answer wrong
# even when the model chose correctly.
_CYRILLIC_TO_LATIN = str.maketrans("АВСЕОРХУКІМВТ", "ABCEOPXYKIMBt")

# Valid answer letters for each variant.
_VALID_TRAD_ANSWERS = set("ABCDE")
_VALID_WEST_ANSWERS = set("ABCDEFGHIJ")


def _normalize_answer(raw_answer: str) -> str:
    """Clean an answer field to a single uppercase Latin letter.

    Steps:
    1. Strip whitespace
    2. Translate Cyrillic lookalike characters → Latin equivalents
    3. Uppercase

    Returns the cleaned string (may still be invalid if the source
    data has a non-letter answer — callers use _is_valid_entry to
    filter those out before they reach the pipeline).
    """
    return raw_answer.strip().translate(_CYRILLIC_TO_LATIN).upper()


def _is_valid_entry(raw: dict, valid_answers: set[str]) -> bool:
    """Return True if this entry is a clean, scorable MCQ.

    Filters out two types of bad entries found in the wild:
    - Non-MCQ entries accidentally included (answer is a full sentence)
    - Entries whose answer is outside the expected letter set

    Args:
        raw: The raw JSON dict for one question.
        valid_answers: Set of acceptable answer letters (e.g. {'A','B','C','D','E'}).

    Returns:
        True if the entry should be kept, False if it should be dropped.
    """
    answer = _normalize_answer(raw["answer"])
    return len(answer) == 1 and answer in valid_answers


def _normalize_entry(raw: dict) -> dict:
    """Convert one raw HiMed entry into our standard dict shape.

    Handles both Trad and West HiMed schemas. Keeps only the fields
    needed for evaluation; missing optional fields default to empty
    strings so the dict shape is consistent across variants.
    """
    return {
        "question": _format_question(raw),
        "answer": _normalize_answer(raw["answer"]),  # normalize 'a'→'A', Cyrillic→Latin
        "subject": raw.get("subject", "").strip(),
        "system": raw.get("system", "").strip(),
        "source": raw.get("source", "").strip(),
    }


def load_himed(
    variant: str | None = None,
    file_path: Path | None = None,
    sample_size: int | None = None,
) -> list[dict]:
    """Load HiMed dataset as a list of standardized MCQ dicts.

    Args:
        variant: 'trad' (Ayurveda) or 'west' (modern medicine).
                 Mutually exclusive with file_path.
        file_path: Direct path to a JSON file (mostly for tests).
                   Mutually exclusive with variant.
        sample_size: If given, randomly sample this many items.
                     Uses fixed seed (42) for reproducibility.

    Returns:
        List of dicts with keys: question, answer, subject, system, source.

    Examples:
        >>> items = load_himed(variant="trad")          # all Trad questions
        >>> items = load_himed(variant="west", sample_size=10)  # 10 Western
    """
    path = _resolve_path(variant=variant, file_path=file_path)

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Determine which answer letters are valid for this variant.
    # Trad always uses A-E; West uses A-J (up to 10 options).
    # We detect variant from the file path when file_path= was used directly.
    if variant == "trad" or (file_path and "Trad" in str(file_path)):
        valid_answers = _VALID_TRAD_ANSWERS
    else:
        valid_answers = _VALID_WEST_ANSWERS

    # Filter bad entries, then normalize the rest.
    before = len(raw_data)
    clean_data = [e for e in raw_data if _is_valid_entry(e, valid_answers)]
    dropped = before - len(clean_data)
    if dropped:
        print(f"[himed_loader] Dropped {dropped} invalid entr{'y' if dropped == 1 else 'ies'} "
              f"from {path.name} (non-letter or out-of-range answer field).")

    items = [_normalize_entry(entry) for entry in clean_data]

    if sample_size is not None:
        random.seed(42)
        items = random.sample(items, min(sample_size, len(items)))

    return items
