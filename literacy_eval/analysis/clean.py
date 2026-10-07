"""Step 3 of the HiMed-Trad evaluation: extract answers and flag unusable ones.

WHAT THIS FILE DOES:
Reads the 15 raw output CSVs (5 models x 3 levels) and writes ONE long table,
analysis_outputs/clean.csv, with one row per (model, level, item). For each
response it:
  1. extracts the user-facing answer (HiMed-8B prints "## Thinking ...
     ## Response ..."; only the part after "## Response" is what a user sees)
  2. sets rule-based failure flags (refusal, non_answer, echo, degenerate,
     non_hindi, too_short) and an informational `truncated` flag
  3. assigns ONE failure_type (first matching flag wins) and usable_rules

WHY RULES ARE ONLY A FIRST PASS:
Rules catch the obvious failures (loops, echoes, refusals). They cannot catch
fluent-looking nonsense (e.g. Apollo's "'माना' शब्द का उपयोग 'माना' के लिए...").
The judge's `coherent` field (judge.py) is the second gate. Report rule-based
failure rates as a LOWER BOUND on unusable output.

THRESHOLDS were calibrated on the real outputs (see the plan doc): e.g.
too_short is < 3 Hindi words, NOT 15, because HiMed-8B's low-level answers
are genuinely short but real ("उष्ण पदार्थों से परहेज करें.").

    python -m literacy_eval.analysis.clean
"""

from __future__ import annotations

import argparse
import difflib
import re
from pathlib import Path

import pandas as pd

MODELS = ["airavata", "apollo", "himed", "nanda", "mmed_llama"]
LEVELS = ["low", "mid", "high"]

_LIT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = _LIT_DIR / "outputs" / "himed_trad_open"
ANALYSIS_DIR = _LIT_DIR / "analysis_outputs"
CLEAN_CSV = ANALYSIS_DIR / "clean.csv"
REFERENCE_JSON = _LIT_DIR / "reference" / "himed_trad_reference.json"

# --- Thresholds (calibrated 2026-10-01 on outputs/himed_trad_open) ---
ECHO_SIMILARITY = 0.6     # answer start vs question (difflib ratio)
ECHO_MAX_NEW_WORDS = 3    # an echo adds fewer than this many words not in the question
INSTRUCTION_COPY_MAX = 0.3  # share of answer 4-grams copied from the prompt instructions
REPEAT_4GRAM_MAX = 0.3     # share of repeated word 4-grams
MIN_DEVANAGARI_SHARE = 0.5
MIN_HINDI_WORDS = 3
REFUSAL_MAX_WORDS = 100    # longer = disclaimer followed by a real answer

# Devanagari letters/marks, excluding danda (U+0964/5), digits (U+0966-F)
# and the abbreviation sign (U+0970).
HINDI_WORD = re.compile(r"[\u0900-\u0963\u0971-\u097F]+")

_RESPONSE_HEADING = re.compile(r"^#+\s*Response\s*$", re.M)
_THINKING_HEADING = re.compile(r"^#+\s*Thinking\d*\s*$", re.M)

# Nanda's refusals: "मैं इसमें सहायता नहीं कर सकता", "...असमर्थ हूं".
_REFUSAL = re.compile(r"सहायता नहीं कर सकत|मदद नहीं कर सकत|असमर्थ हू[ँं]")
# Airavata's non-answers: "मैं नहीं जानता", "कृपया मुझे ... अधिक जानकारी प्रदान करें".
_NON_ANSWER = re.compile(r"नहीं जानत[ाी]|अधिक जानकारी प्रदान कर")
_CHAR_RUN = re.compile(r"(.)\1{9,}")                    # "यययययययययय"
_NUMBER_LIST = re.compile(r"(?:\d+\s*,\s*){10,}")       # "1,2,3,4,..."
_WORD_LIST = re.compile(r"(?:[^\s,।]+\s*,\s*){15,}")    # "भय,भूख,बसंत,..."
_SENTENCE_END = tuple("।?!.")


def extract_answer(raw: str) -> tuple[str, str]:
    """Split a raw response into (answer, thinking).

    - "## Thinking ... ## Response ..." -> (text after Response, thinking)
    - "## Thinking ..." with no Response heading -> ("", thinking)
      (the model ran out of tokens while still thinking)
    - anything else -> (whole response, "")
    A trailing "." after a danda (HiMed writes "।.") is removed.
    """
    raw = raw or ""
    match = _RESPONSE_HEADING.search(raw)
    if match:
        thinking = _THINKING_HEADING.sub("", raw[: match.start()]).strip()
        answer = raw[match.end():].strip()
    elif _THINKING_HEADING.search(raw):
        return "", _THINKING_HEADING.sub("", raw).strip()
    else:
        answer, thinking = raw.strip(), ""
    answer = re.sub(r"([।?!])\.\s*$", r"\1", answer)
    return answer, thinking


def question_stem(topic_hi: str) -> str:
    """Drop the "आयुर्वेद के अनुसार:" system prefix added to every topic."""
    return topic_hi.split(":", 1)[1].strip() if ":" in topic_hi else topic_hi.strip()


def hindi_words(text: str) -> list[str]:
    return HINDI_WORD.findall(text)


def repeated_4gram_share(text: str) -> float:
    """Share of word 4-grams that are repeats (0 = no repetition)."""
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def devanagari_share(text: str) -> float:
    """Share of alphabetic characters that are Devanagari."""
    letters = [c for c in text if c.isalpha() or "\u0900" <= c <= "\u097F"]
    if not letters:
        return 0.0
    return sum("\u0900" <= c <= "\u097F" for c in letters) / len(letters)


def has_other_script(text: str) -> bool:
    """Letters that are neither Devanagari nor Latin (e.g. MMed's "奏曲する")."""
    return any(
        c.isalpha() and not ("\u0900" <= c <= "\u097F") and not c.isascii()
        and not ("\u00C0" <= c <= "\u024F")   # accented Latin is fine
        for c in text
    )


def instruction_text(prompt: str, topic_hi: str) -> str:
    """The prompt minus the topic itself (so copying the topic isn't counted)."""
    return prompt.replace(topic_hi, " ")


def copied_4gram_share(answer: str, source: str) -> float:
    """Share of the answer's word 4-grams that also occur in `source`."""
    words = answer.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    if not grams:
        return 0.0
    src = source.split()
    src_grams = {tuple(src[i:i + 4]) for i in range(len(src) - 3)}
    return sum(g in src_grams for g in grams) / len(grams)


def flag_response(answer: str, topic_hi: str, prompt: str = "") -> dict:
    """Compute all rule-based flags for one extracted answer.

    Echo has two forms, both seen in Airavata's output:
      - question echo: the answer starts by restating the question (with or
        without the "आयुर्वेद के अनुसार" prefix) AND is still a question
        (ends in "?") or adds < 3 words that aren't in the question.
        Restating then answering is normal (HiMed: "जिन्हें सरलता से दस्त
        आते हैं, उन्हें 'आम' कोष्ठ का रोगी कहा जाता है।") and is NOT an echo.
      - instruction echo: >= 30% of the answer's 4-grams are copied from the
        prompt's instructions ("- सामान्य हिंदी का प्रयोग करें ...")
    """
    head = answer[: len(topic_hi) + 20]
    similarity = max(
        difflib.SequenceMatcher(None, topic_hi, head).ratio(),
        difflib.SequenceMatcher(None, question_stem(topic_hi), head).ratio(),
    ) if answer else 0.0
    starts_with_question = similarity >= ECHO_SIMILARITY
    question_words = set(hindi_words(topic_hi))
    new_words = sum(w not in question_words for w in hindi_words(answer))
    still_a_question = answer.rstrip().endswith("?")
    instruction_copy = copied_4gram_share(answer, instruction_text(prompt, topic_hi))
    rep = repeated_4gram_share(answer)
    dev = devanagari_share(answer)
    n_hi = len(hindi_words(answer))

    flags = {
        "refusal": bool(_REFUSAL.search(answer)) and n_hi < REFUSAL_MAX_WORDS,
        "non_answer": bool(_NON_ANSWER.search(answer)),
        "echo": (
            (starts_with_question and (still_a_question or new_words < ECHO_MAX_NEW_WORDS))
            or instruction_copy >= INSTRUCTION_COPY_MAX
        ),
        "degenerate": (
            rep > REPEAT_4GRAM_MAX
            or bool(_CHAR_RUN.search(answer))
            or bool(_NUMBER_LIST.search(answer))
            or bool(_WORD_LIST.search(answer))
            or has_other_script(answer)
        ),
        # Needs letters: an empty answer is too_short, not non_hindi.
        "non_hindi": any(c.isalpha() for c in answer) and dev < MIN_DEVANAGARI_SHARE,
        "too_short": n_hi < MIN_HINDI_WORDS,
    }
    # Informational only — never makes a response unusable.
    flags["refusal_language"] = bool(_REFUSAL.search(answer))
    flags["starts_with_question"] = starts_with_question
    flags["instruction_copy"] = round(instruction_copy, 3)
    flags["truncated"] = bool(answer) and not answer.rstrip().endswith(_SENTENCE_END)
    flags["repeat_4gram"] = round(rep, 3)
    flags["devanagari_share"] = round(dev, 3)
    flags["n_hindi_words"] = n_hi

    flags["failure_type"] = next(
        (name for name in ("refusal", "non_answer", "echo", "degenerate", "non_hindi", "too_short") if flags[name]),
        "ok",
    )
    flags["usable_rules"] = flags["failure_type"] == "ok"
    return flags


def build_clean_table(input_dir: Path = DEFAULT_INPUT_DIR) -> pd.DataFrame:
    """Read every {model}_{level}_himed_trad_raw.csv and return the long table."""
    rows = []
    for model in MODELS:
        for level in LEVELS:
            path = Path(input_dir) / f"{model}_{level}_himed_trad_raw.csv"
            if not path.exists():
                print(f"  WARNING: missing {path.name}, skipping")
                continue
            df = pd.read_csv(path, dtype=str).fillna("")
            if df["topic_id"].nunique() != len(df):
                raise ValueError(f"{path.name}: duplicate topic_id rows")
            for rec in df.to_dict("records"):
                answer, thinking = extract_answer(rec["model_response"])
                system, _, subject = rec["category"].partition("/")
                rows.append({
                    "model": model,
                    "level": level,
                    "item_id": rec["topic_id"],
                    "system": system,
                    "subject": subject,
                    "topic_hi": rec["topic_hi"],
                    "raw_response": rec["model_response"],
                    "answer": answer,
                    "thinking": thinking,
                    **flag_response(answer, rec["topic_hi"], rec["prompt"]),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out", type=Path, default=CLEAN_CSV)
    args = parser.parse_args()

    table = build_clean_table(args.input_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} rows to {args.out}\n")
    print(pd.crosstab([table["model"], table["level"]], table["failure_type"]).to_string())


if __name__ == "__main__":
    main()
