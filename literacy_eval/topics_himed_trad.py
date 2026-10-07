"""HiMed-Trad reference set for ground-truth-anchored literacy evaluation.

WHAT THIS FILE DOES:
Builds a small, frozen set of HiMed-Trad questions that serve two roles:
  1. PROMPTS — each question stem becomes an open-ended "explain this"
     topic sent to every model at every literacy level.
  2. GROUND TRUTH — each item keeps its correct option and its chain-of-
     thought (cot), which we split into "key facts" for accuracy scoring.

WHY HiMed-Trad (NOT HiMed-West):
HiMed-West has no rationale field — only question/options/answer. Trad
has a `cot` on every item (1-3 sentences explaining why the answer is
right). That gives us dataset-provided key facts instead of inventing
our own reference. See literacy_eval/docs/himed_trad_evaluation_plan.md.

WORKFLOW (two commands, with a human review in between):

    # 1. Draft: sample items, write a CSV for human review
    python -m literacy_eval.topics_himed_trad --draft

    # 2. Open literacy_eval/reference/himed_trad_review.csv, then for each row:
    #      - edit topic_hi_final   (open-ended question, must NOT reveal the answer)
    #      - edit key_fact_1..3    (2-3 short, checkable facts from the cot)
    #      - set exclude=y         to drop a bad item (a reserve replaces it)
    #      - set reviewed=y        once you're happy with the row
    #    Blank *_final / key_fact cells fall back to the draft values.

    # 2b. (Optional) If you excluded so many items that some (system, subject)
    #     cells dropped below PER_CELL, top them up with fresh candidates.
    #     New rows are appended unreviewed — review them, then continue.
    python -m literacy_eval.topics_himed_trad --replace

    # 3. Finalize: freeze the reviewed items into the JSON everyone reads
    python -m literacy_eval.topics_himed_trad --finalize

WHY A FROZEN JSON:
Every model must see identical prompts, and every scorer must read
identical key facts. The JSON (himed_trad_reference.json) is the single
source of truth — re-running --draft does not change it.

WHY RESERVES:
Each (system, subject) cell gets 2 primary items + 1 reserve. If the
reviewer excludes a primary, finalize promotes that cell's reserve, so
the sample stays balanced without re-sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

_DATA_PATH = Path(__file__).parent.parent / "data" / "himed" / "HiMed-Trad_Bench.json"
REFERENCE_DIR = Path(__file__).parent / "reference"
REVIEW_CSV = REFERENCE_DIR / "himed_trad_review.csv"
REFERENCE_JSON = REFERENCE_DIR / "himed_trad_reference.json"

# Fixed seed for reproducible sampling. Do not change after the first run.
SAMPLE_SEED = 42
PER_CELL = 2        # primary items per (system, subject) cell
RESERVES_PER_CELL = 1

# Sowa-Rigpa has only 10 items — too few to stratify, so it's excluded.
SYSTEMS = ["Ayurveda", "Homoeopathy", "Naturopathy", "Siddha", "Unani", "Yoga"]
SUBJECTS = ["etiology", "diagnosis", "treatment", "prognosis", "medical knowledge"]

# Prefix added to each topic so the model answers within the right system.
# Without it, a model would explain e.g. fever in Western terms and be
# scored "wrong" against an Ayurvedic answer key.
SYSTEM_HI = {
    "Ayurveda": "आयुर्वेद",
    "Homoeopathy": "होम्योपैथी",
    "Naturopathy": "प्राकृतिक चिकित्सा",
    "Siddha": "सिद्ध चिकित्सा",
    "Unani": "यूनानी चिकित्सा",
    "Yoga": "योग",
}

OPTION_LETTERS = "ABCDE"
MAX_KEY_FACTS = 3

# Stems containing these can't be answered without seeing the options
# (negation/exception questions, or explicit references to "the options").
_NEGATION_PATTERNS = ["नहीं है", "नहीं होता", "नहीं होती", "नहीं किया", "सिवाय", "अतिरिक्त", "अलावा"]
_OPTION_REFERENCE_PATTERNS = ["उपरोक्त", "विकल्प", "दिए गए"]

# Sentences that just restate the answer are not usable key facts. This
# catches both word orders seen in the CoT — "सही विकल्प है X" and
# "(इसलिए) विकल्प X सही है" — plus meta phrasing that names an option letter
# ("विकल्प C") or refers to the item itself ("इस आइटम / इस प्रश्न में").
# A frozen key fact must never mention the options, or the accuracy judge
# (which is shown the options) would be scoring against a leaked answer.
_META_FACT = re.compile(
    r"सही\s+(?:विकल्प|उत्तर)"      # "... सही विकल्प/उत्तर ..."
    r"|विकल्प\s*[A-Ea-e]"          # "विकल्प C", "विकल्प D सबसे उपयुक्त ..."
    r"|इस\s+आइटम"                  # "इस आइटम के ... संदर्भ में मुख्य उत्तर ..."
    r"|इस\s+प्रश्न"                # "इस प्रश्न में ..."
)

# An option marker is a letter A-E followed by '.' or ')' and whitespace,
# at the start of the text or after whitespace.
_OPTION_MARKER = re.compile(r"(?:(?<=\s)|^)([A-E])[\.\)]\s+")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def normalize_answer(raw: str) -> str | None:
    """Normalize an answer field to a single letter A-E, or None if malformed.

    The dataset has a few oddities: a lowercase 'e', a Cyrillic 'Е', and one
    answer that is a full sentence instead of a letter.
    """
    ans = raw.strip().replace("Е", "E").upper()
    return ans if len(ans) == 1 and ans in OPTION_LETTERS else None


def parse_question(question: str) -> tuple[str, dict[str, str]] | None:
    """Split an MCQ question into (stem, {A: ..., ..., E: ...}).

    Options appear in three formats in the data: newline-separated,
    inline on one line, and separated by <br> tags. All are handled by
    normalizing <br> to newlines and scanning for A-E markers in order.

    Returns None if the question doesn't contain a complete A-E sequence.
    """
    text = question.replace("<br>", "\n")
    markers = list(_OPTION_MARKER.finditer(text))

    # Find a run of markers A, B, C, D, E in order. Restart whenever an 'A'
    # appears, so a stray "A." inside the stem doesn't break parsing.
    run: list[re.Match] = []
    for m in markers:
        letter = m.group(1)
        if letter == "A":
            run = [m]
        elif run and letter == OPTION_LETTERS[len(run)]:
            run.append(m)
            if len(run) == len(OPTION_LETTERS):
                break
    if len(run) != len(OPTION_LETTERS):
        return None

    stem = text[: run[0].start()].strip()
    options = {}
    for i, m in enumerate(run):
        end = run[i + 1].start() if i + 1 < len(run) else len(text)
        options[m.group(1)] = text[m.end():end].strip()
    return stem, options


def is_answerable(stem: str) -> bool:
    """Can this stem be answered as an open-ended question, without options?

    Rejects: empty stems, stems that don't end in '?', chat transcripts
    (User:/Assistant:), negation/exception stems, and stems that refer to
    "the options" / "the above".
    """
    if not stem or not stem.endswith("?"):
        return False
    if "User:" in stem or "Assistant:" in stem:
        return False
    if any(p in stem for p in _NEGATION_PATTERNS):
        return False
    if any(p in stem for p in _OPTION_REFERENCE_PATTERNS):
        return False
    return True


def draft_topic(stem: str, system: str) -> str:
    """Deterministic first-pass rewrite of an MCQ stem into an open-ended topic.

    Strips the generic "पारंपरिक चिकित्सा में" frame and the "निम्नलिखित में से"
    lead-in, then prefixes the specific system (e.g. "आयुर्वेद के अनुसार:").
    The human reviewer polishes this into topic_hi_final.
    """
    topic = re.sub(r"^पारंपरिक चिकित्सा (?:में|के अनुसार),?\s*", "", stem)
    topic = re.sub(r"निम्नलिखित में से\s*", "", topic)
    topic = re.sub(r"\s+", " ", topic).strip()
    return f"{SYSTEM_HI[system]} के अनुसार: {topic}"


def draft_key_facts(cot: str) -> list[str]:
    """Split a CoT into candidate key facts (one per sentence).

    Sentences that only restate the answer ("सही विकल्प है ...") are dropped.
    Returns at most MAX_KEY_FACTS facts; the reviewer can edit/merge/split.
    """
    sentences = [s.strip() for s in re.split(r"[।?!]", cot) if s.strip()]
    facts = [s for s in sentences if not _META_FACT.search(s)]
    return facts[:MAX_KEY_FACTS]


def make_item_id(question_number: str) -> str:
    """Stable ID, prefixed to avoid collisions with curated/West topic IDs."""
    return f"trad_{question_number}"


def load_candidates(data_path: Path = _DATA_PATH) -> list[dict]:
    """Load HiMed-Trad and return parsed, answerable items in SYSTEMS/SUBJECTS."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"HiMed-Trad dataset not found at {data_path}. "
            "Check that data/himed/ is present in the project root."
        )
    with open(data_path, encoding="utf-8") as f:
        raw_items = json.load(f)

    candidates = []
    for item in raw_items:
        if item.get("system") not in SYSTEMS or item.get("subject") not in SUBJECTS:
            continue
        answer = normalize_answer(item.get("answer", ""))
        parsed = parse_question(item.get("question", ""))
        if answer is None or parsed is None:
            continue
        stem, options = parsed
        if not is_answerable(stem):
            continue
        cot = (item.get("cot") or "").strip()
        if not cot:
            continue
        candidates.append({
            "item_id": make_item_id(item["question_number"]),
            "system": item["system"],
            "subject": item["subject"],
            "original_question": item["question"],
            "stem": stem,
            "options": options,
            "correct_answer": answer,
            "correct_option_text": options[answer],
            "cot": cot,
        })
    return candidates


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def stratified_sample(
    candidates: list[dict],
    per_cell: int = PER_CELL,
    reserves_per_cell: int = RESERVES_PER_CELL,
    seed: int = SAMPLE_SEED,
) -> list[dict]:
    """Sample per (system, subject) cell, marking each item primary or reserve.

    Candidates are sorted by item_id before sampling so the result depends
    only on the seed, not on JSON order. If a cell has fewer items than
    per_cell, all of them become primaries and the shortfall is made up
    from the same system's other subjects (so each system stays at
    per_cell * len(SUBJECTS) primaries).

    Returns items with an added "role" ("primary" or "reserve") field.
    """
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for c in sorted(candidates, key=lambda c: c["item_id"]):
        by_cell.setdefault((c["system"], c["subject"]), []).append(c)

    selected = []
    for system in SYSTEMS:
        leftovers = []
        shortfall = 0
        for subject in SUBJECTS:
            pool = list(by_cell.get((system, subject), []))
            rng.shuffle(pool)
            take = per_cell + reserves_per_cell
            chosen, rest = pool[:take], pool[take:]
            n_primary = min(per_cell, len(chosen))
            shortfall += per_cell - n_primary
            for i, item in enumerate(chosen):
                selected.append({**item, "role": "primary" if i < n_primary else "reserve"})
            leftovers.extend(rest)
        # Fill any shortfall from this system's unused items (other subjects).
        rng.shuffle(leftovers)
        for item in leftovers[:shortfall]:
            selected.append({**item, "role": "primary"})
    return selected


# ---------------------------------------------------------------------------
# Review CSV (draft) and frozen JSON (finalize)
# ---------------------------------------------------------------------------

REVIEW_FIELDS = [
    "item_id", "role", "system", "subject",
    "original_question", "options_json", "correct_answer", "correct_option_text", "cot",
    "topic_hi_draft", "topic_hi_final",
    "key_fact_1_draft", "key_fact_2_draft", "key_fact_3_draft",
    "key_fact_1", "key_fact_2", "key_fact_3",
    "answer_in_topic", "exclude", "reviewed", "notes",
]


def _item_to_review_row(item: dict) -> dict:
    """Build one review-CSV row (all draft cells filled, review cells blank)."""
    topic = draft_topic(item["stem"], item["system"])
    facts = draft_key_facts(item["cot"]) + [""] * MAX_KEY_FACTS
    return {
        "item_id": item["item_id"],
        "role": item["role"],
        "system": item["system"],
        "subject": item["subject"],
        "original_question": item["original_question"],
        "options_json": json.dumps(item["options"], ensure_ascii=False),
        "correct_answer": item["correct_answer"],
        "correct_option_text": item["correct_option_text"],
        "cot": item["cot"],
        "topic_hi_draft": topic,
        "topic_hi_final": "",
        "key_fact_1_draft": facts[0],
        "key_fact_2_draft": facts[1],
        "key_fact_3_draft": facts[2],
        "key_fact_1": "",
        "key_fact_2": "",
        "key_fact_3": "",
        # Flag if the answer text leaks into the prompt — reviewer should reword.
        "answer_in_topic": "y" if item["correct_option_text"] in topic else "",
        "exclude": "",
        "reviewed": "",
        "notes": "",
    }


def write_review_csv(items: list[dict], path: Path = REVIEW_CSV, overwrite: bool = False) -> None:
    """Write sampled items to a CSV for human review.

    Refuses to overwrite an existing review file (it may contain your
    edits) unless overwrite=True.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists and may contain review edits. "
            "Pass --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel opens Devanagari correctly.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(_item_to_review_row(item))


def _is_yes(value: str) -> bool:
    return value.strip().lower() in {"y", "yes", "1", "true"}


def _row_to_reference(row: dict) -> dict:
    """Convert a reviewed CSV row into a reference item (run_eval topic format)."""
    topic_hi = row["topic_hi_final"].strip() or row["topic_hi_draft"].strip()
    facts = []
    for i in range(1, MAX_KEY_FACTS + 1):
        fact = row[f"key_fact_{i}"].strip() or row[f"key_fact_{i}_draft"].strip()
        if fact:
            facts.append(fact)
    return {
        # Core fields — same shape as topics.py / topics_himed.py
        "id": row["item_id"],
        "topic_hi": topic_hi,
        "topic_en": f"(HiMed-Trad {row['system']}/{row['subject']})",
        "category": f"{row['system']}/{row['subject']}",
        # Ground truth for scoring
        "system": row["system"],
        "subject": row["subject"],
        "original_question": row["original_question"],
        "options": json.loads(row["options_json"]),
        "correct_answer": row["correct_answer"],
        "correct_option_text": row["correct_option_text"],
        "cot": row["cot"],
        "key_facts": facts,
        "source": "HiMed-Trad",
    }


def finalize(
    review_path: Path = REVIEW_CSV,
    output_path: Path = REFERENCE_JSON,
    allow_unreviewed: bool = False,
) -> list[dict]:
    """Freeze the reviewed CSV into the reference JSON.

    For each (system, subject) cell: keep non-excluded primaries, and for
    each excluded primary promote the cell's next non-excluded reserve.

    Raises:
        ValueError: if a selected row isn't marked reviewed (unless
            allow_unreviewed), has no key facts, or its topic contains
            the correct answer text.
    """
    with open(review_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_cell.setdefault((row["system"], row["subject"]), []).append(row)

    selected = []
    for cell_rows in by_cell.values():
        primaries = [r for r in cell_rows if r["role"] == "primary"]
        reserves = [r for r in cell_rows if r["role"] == "reserve" and not _is_yes(r["exclude"])]
        for row in primaries:
            if not _is_yes(row["exclude"]):
                selected.append(row)
            elif reserves:
                selected.append(reserves.pop(0))

    problems = []
    for row in selected:
        ref = _row_to_reference(row)
        if not allow_unreviewed and not _is_yes(row["reviewed"]):
            problems.append(f"{row['item_id']}: not marked reviewed")
        if not ref["key_facts"]:
            problems.append(f"{row['item_id']}: no key facts")
        if row["correct_option_text"] and row["correct_option_text"] in ref["topic_hi"]:
            problems.append(f"{row['item_id']}: topic contains the correct answer text")
        for fact in ref["key_facts"]:
            if _META_FACT.search(fact):
                problems.append(
                    f"{row['item_id']}: key fact names an option / is meta "
                    f"(reword or blank it): {fact[:60]}"
                )
    if problems:
        raise ValueError("Cannot finalize:\n  " + "\n  ".join(problems))

    reference = sorted((_row_to_reference(r) for r in selected), key=lambda r: r["id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)
    return reference


def load_himed_trad_topics(path: Path = REFERENCE_JSON) -> list[dict]:
    """Load the frozen reference set (for run_eval.py and the scorers)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m literacy_eval.topics_himed_trad --draft`, "
            "review the CSV, then run `--finalize`."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _selected_count_by_cell(rows: list[dict]) -> dict[tuple[str, str], int]:
    """Number of items each cell would contribute to finalize, right now.

    Mirrors finalize's selection exactly: every non-excluded primary is
    kept, and each excluded primary promotes one non-excluded reserve
    (while reserves last). Used to find cells that fell below PER_CELL.
    """
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_cell.setdefault((row["system"], row["subject"]), []).append(row)

    counts: dict[tuple[str, str], int] = {}
    for cell, cell_rows in by_cell.items():
        primaries = [r for r in cell_rows if r["role"] == "primary"]
        reserves = [r for r in cell_rows if r["role"] == "reserve" and not _is_yes(r["exclude"])]
        selected = 0
        for row in primaries:
            if not _is_yes(row["exclude"]):
                selected += 1
            elif reserves:
                reserves.pop(0)
                selected += 1
        counts[cell] = selected
    return counts


def replacement_draw(
    review_path: Path = REVIEW_CSV,
    per_cell: int = PER_CELL,
    seed: int = SAMPLE_SEED,
    candidates: list[dict] | None = None,
) -> list[dict]:
    """Top up cells that fell below `per_cell` selected items after exclusions.

    Reads the existing review CSV, computes how many items each cell would
    currently contribute to finalize (see _selected_count_by_cell), and for
    any cell below `per_cell` appends exactly the shortfall as new PRIMARY
    candidates. New rows are appended with review cells blank (reviewed=""),
    so they still go through human review before finalize.

    No reserves are added: a reserve would let an already-excluded primary
    get promoted too, overshooting `per_cell`. If a newly added primary is
    later excluded, just run --replace again.

    Deterministic: draws in a fixed cell order with `seed`, skipping any
    item_id already present in the CSV. Returns the newly appended rows.
    """
    with open(review_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    existing_ids = {r["item_id"] for r in rows}
    selected = _selected_count_by_cell(rows)

    if candidates is None:
        candidates = load_candidates()
    pool_by_cell: dict[tuple[str, str], list[dict]] = {}
    for c in sorted(candidates, key=lambda c: c["item_id"]):
        if c["item_id"] not in existing_ids:
            pool_by_cell.setdefault((c["system"], c["subject"]), []).append(c)

    rng = random.Random(seed)
    new_rows: list[dict] = []
    shortfalls: list[str] = []
    # Fixed cell order so the draw is reproducible.
    for system in SYSTEMS:
        for subject in SUBJECTS:
            cell = (system, subject)
            deficit = per_cell - selected.get(cell, 0)
            if deficit <= 0:
                continue
            pool = list(pool_by_cell.get(cell, []))
            rng.shuffle(pool)
            drawn = pool[:deficit]
            if len(drawn) < deficit:
                shortfalls.append(f"{system}/{subject} (needed {deficit}, found {len(drawn)})")
            for item in drawn:
                new_rows.append(_item_to_review_row({**item, "role": "primary"}))

    if new_rows:
        with open(review_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            writer.writerows(new_rows)
    if shortfalls:
        print("WARNING — not enough spare candidates for: " + "; ".join(shortfalls))
    return new_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HiMed-Trad reference set.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draft", action="store_true", help="Sample items and write the review CSV")
    mode.add_argument("--replace", action="store_true",
                      help="Top up cells that fell below PER_CELL after exclusions (appends to the CSV)")
    mode.add_argument("--finalize", action="store_true", help="Freeze the reviewed CSV into JSON")
    parser.add_argument("--overwrite", action="store_true", help="With --draft: replace an existing review CSV")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="With --finalize: don't require reviewed=y (for testing only)")
    args = parser.parse_args()

    if args.draft:
        candidates = load_candidates()
        items = stratified_sample(candidates)
        write_review_csv(items, overwrite=args.overwrite)
        n_primary = sum(i["role"] == "primary" for i in items)
        n_leak = sum(1 for i in items if i["correct_option_text"] in draft_topic(i["stem"], i["system"]))
        print(f"{len(candidates)} answerable candidates → {n_primary} primary + "
              f"{len(items) - n_primary} reserve items")
        print(f"{n_leak} drafts contain the answer text (flagged answer_in_topic=y)")
        print(f"Review CSV written to {REVIEW_CSV}")
    elif args.replace:
        new_rows = replacement_draw()
        if new_rows:
            cells = sorted({f"{r['system']}/{r['subject']}" for r in new_rows})
            print(f"Appended {len(new_rows)} replacement rows to {REVIEW_CSV}")
            print(f"Cells topped up: {', '.join(cells)}")
            print("Review the new rows (reviewed is blank) before running --finalize.")
        else:
            print("No cells are below PER_CELL — nothing to replace.")
    else:
        reference = finalize(allow_unreviewed=args.allow_unreviewed)
        print(f"Froze {len(reference)} items to {REFERENCE_JSON}")


if __name__ == "__main__":
    main()
