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


def _normalize_entry(raw: dict) -> dict:
    """Convert one raw HiMed entry into our standard dict shape.

    Handles both Trad and West HiMed schemas. Keeps only the fields
    needed for evaluation; missing optional fields default to empty
    strings so the dict shape is consistent across variants.
    """
    return {
        "question": _format_question(raw),
        "answer": raw["answer"].strip().upper(),  # normalize 'a' → 'A'
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

    items = [_normalize_entry(entry) for entry in raw_data]

    if sample_size is not None:
        random.seed(42)
        items = random.sample(items, min(sample_size, len(items)))

    return items
