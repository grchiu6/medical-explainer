"""Step 4 of the HiMed-Trad evaluation: deterministic readability features.

WHAT THIS FILE DOES:
Adds readability/vocabulary columns to analysis_outputs/clean.csv and writes
analysis_outputs/features.csv. Every feature is computed from the answer text
alone — no reference, no LLM, no Hindi reader needed.

WHY EVERY FEATURE IS A RATE (per word / per sentence):
Output length is capped by max_new_tokens and differs across models, and
HiMed-8B's low answers are short because they say LESS, not because the
language is simpler. Rates make features comparable across lengths.

FEATURES (lower = simpler for all except mean_zipf):
  asl          words per sentence (Sinha et al. 2012: ASL)
  awl          visual units per word (Sinha et al. 2012: AWL). A visual unit
               is a consonant or conjunct cluster, an independent vowel, a
               vowel sign (matra) or a nasal/visarga mark.
  psw_rate     share of words with >= 3 syllables (aksharas)
  juk_rate     conjunct clusters (jukta-akshar) per word
  rare_rate    share of Hindi words with wordfreq Zipf < 3
  mean_zipf    mean wordfreq Zipf of Hindi words (HIGHER = more common words)
  latin_rate   share of words in Latin script (English terms)
  copy_rate    share of the answer's content words copied from the question
  rh1, rh2     Sinha et al. (2012) Hindi readability models, with PSW and JUK
               taken per 100 words (the paper uses raw counts per text, which
               would make longer answers look harder). Rough indices only.

INSTRUCTION-COMPLIANCE FEATURES (each tests one line of the level prompts):
  short_sentence_share  share of sentences with 5-10 words. The low prompt
               says "छोटे और सीधे वाक्य लिखें (5-10 शब्दों के)".
  term_rate    share of Hindi words that are domain terms. The term lexicon
               is built automatically from the 60 reference items: every
               rare word (Zipf < 3, not a stopword) in a question, option,
               CoT or key fact. The low prompt forbids medical terms; the
               high prompt asks for technical terminology.
  gloss_rate   parenthetical explanations "शब्द (अर्थ)" per 100 words. The
               mid prompt says to give the simple meaning of medical terms.
               A rough proxy: it also counts English glosses like "(gas)".

    python -m literacy_eval.analysis.features
"""

from __future__ import annotations

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from wordfreq import zipf_frequency

from literacy_eval.analysis.clean import ANALYSIS_DIR, CLEAN_CSV, HINDI_WORD, REFERENCE_JSON

FEATURES_CSV = ANALYSIS_DIR / "features.csv"
FEATURE_COLUMNS = [
    "n_words", "n_sentences", "asl", "awl", "psw_rate", "juk_rate",
    "rare_rate", "mean_zipf", "latin_rate", "copy_rate", "rh1", "rh2",
    "short_sentence_share", "term_rate", "gloss_rate",
]
RARE_ZIPF = 3.0
SHORT_SENTENCE_WORDS = (5, 10)   # inclusive bounds from the low-level prompt

_C = "[\u0915-\u0939\u0958-\u095F]"   # consonants (incl. nukta forms)
_NUKTA = "\u093C"
_HALANT = "\u094D"
_VOWEL_SIGN = "[\u093E-\u094C\u0962\u0963]"
_INDEP_VOWEL = "[\u0904-\u0914\u0960\u0961]"
_MARK = "[\u0900-\u0903]"            # candrabindu, anusvara, visarga

# One orthographic syllable: an optional conjunct prefix, a consonant, an
# optional vowel sign or final halant, then nasal marks — or an independent vowel.
_AKSHARA = re.compile(
    rf"(?:{_C}{_NUKTA}?{_HALANT})*{_C}{_NUKTA}?(?:{_VOWEL_SIGN}|{_HALANT}(?!{_C}))?{_MARK}*"
    rf"|{_INDEP_VOWEL}{_MARK}*"
)
# A consonant cluster: consonants joined by halant count as ONE visual unit.
_CLUSTER = re.compile(rf"(?:{_C}{_NUKTA}?{_HALANT})*{_C}{_NUKTA}?")
_CONJUNCT = re.compile(rf"(?:{_C}{_NUKTA}?{_HALANT})+{_C}")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_SENTENCE_SPLIT = re.compile(r"[।॥?!\n]+|\.(?=\s|$)")
_GLOSS = re.compile(r"\([^()]{1,60}\)")

# Function words excluded from copy_rate (they appear in every question).
STOPWORDS = frozenset(
    "के का की को में से है हैं और या पर यह वह एक भी तो कि जो इस उस इसके उसके "
    "होता होती होते हो जाता जाती जाते किया करें करना कर लिए साथ अनुसार क्या कौन "
    "किस किसी कौन-सा कौन-सी नहीं ने ही जब तब यदि अगर उसे उन्हें इसे".split()
)


def count_aksharas(word: str) -> int:
    """Syllable approximation: number of orthographic syllables (aksharas)."""
    return len(_AKSHARA.findall(word))


def count_visual_units(word: str) -> int:
    """Sinha et al. (2012) word length: clusters + vowels + vowel signs + marks."""
    return (
        len(_CLUSTER.findall(word))
        + len(re.findall(_INDEP_VOWEL, word))
        + len(re.findall(_VOWEL_SIGN, word))
        + len(re.findall(_MARK, word))
    )


def count_conjuncts(word: str) -> int:
    """Number of conjunct clusters (jukta-akshar) — स्वास्थ्य has 2 (स्व, स्थ्य)."""
    return len(_CONJUNCT.findall(word))


def split_sentences(text: str) -> list[str]:
    """Split on danda, ?, !, newline and '.' followed by space/end."""
    parts = _SENTENCE_SPLIT.split(text)
    return [p for p in parts if HINDI_WORD.search(p) or _LATIN_WORD.search(p)]


@lru_cache(maxsize=None)
def _zipf(word: str) -> float:
    return zipf_frequency(word, "hi")


def _sentence_words(sentence: str) -> int:
    return len(HINDI_WORD.findall(sentence)) + len(_LATIN_WORD.findall(sentence))


def build_term_lexicon(reference_items: list[dict]) -> frozenset[str]:
    """Rare non-stopword Hindi words from the reference items' text."""
    terms = set()
    for item in reference_items:
        texts = [item["topic_hi"], item["cot"], *item["options"].values(), *item["key_facts"]]
        for word in HINDI_WORD.findall(" ".join(texts)):
            if word not in STOPWORDS and _zipf(word) < RARE_ZIPF:
                terms.add(word)
    return frozenset(terms)


def load_term_lexicon(path: Path = REFERENCE_JSON) -> frozenset[str]:
    with open(path, encoding="utf-8") as f:
        return build_term_lexicon(json.load(f))


def compute_features(answer: str, topic_hi: str, terms: frozenset[str] = frozenset()) -> dict:
    """All readability features for one answer. Empty dict values -> NaN."""
    hi = HINDI_WORD.findall(answer)
    latin = _LATIN_WORD.findall(answer)
    n_words = len(hi) + len(latin)
    sentences = split_sentences(answer)
    if n_words == 0 or not sentences or not hi:
        return {col: float("nan") for col in FEATURE_COLUMNS}

    aksharas = [count_aksharas(w) for w in hi]
    awl = sum(count_visual_units(w) for w in hi) / len(hi)
    psw_rate = sum(a >= 3 for a in aksharas) / len(hi)
    juk_rate = sum(count_conjuncts(w) for w in hi) / len(hi)
    zipfs = [_zipf(w) for w in hi]

    content = {w for w in hi if w not in STOPWORDS}
    question_words = set(HINDI_WORD.findall(topic_hi))
    copy_rate = len(content & question_words) / len(content) if content else float("nan")

    lo, hi_bound = SHORT_SENTENCE_WORDS
    short = sum(lo <= _sentence_words(s) <= hi_bound for s in sentences)

    return {
        "n_words": n_words,
        "n_sentences": len(sentences),
        "asl": n_words / len(sentences),
        "awl": awl,
        "psw_rate": psw_rate,
        "juk_rate": juk_rate,
        "rare_rate": sum(z < RARE_ZIPF for z in zipfs) / len(hi),
        "mean_zipf": sum(zipfs) / len(zipfs),
        "latin_rate": len(latin) / n_words,
        "copy_rate": copy_rate,
        "rh1": -2.34 + 2.14 * awl + 0.01 * (psw_rate * 100),
        "rh2": 0.211 + 1.37 * awl + 0.005 * (juk_rate * 100),
        "short_sentence_share": short / len(sentences),
        "term_rate": sum(w in terms for w in hi) / len(hi),
        "gloss_rate": len(_GLOSS.findall(answer)) / n_words * 100,
    }


def add_features(clean: pd.DataFrame, terms: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Return `clean` with FEATURE_COLUMNS added (NaN for unusable rows)."""
    feats = [
        compute_features(a, t, terms) if usable else {c: float("nan") for c in FEATURE_COLUMNS}
        for a, t, usable in zip(clean["answer"].fillna(""), clean["topic_hi"], clean["usable_rules"])
    ]
    return pd.concat([clean.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clean", type=Path, default=CLEAN_CSV)
    parser.add_argument("--out", type=Path, default=FEATURES_CSV)
    args = parser.parse_args()

    terms = load_term_lexicon()
    print(f"Term lexicon: {len(terms)} words from the reference items")
    table = add_features(pd.read_csv(args.clean), terms)
    table.to_csv(args.out, index=False)
    print(f"Wrote {len(table)} rows to {args.out}\n")
    usable = table[table["usable_rules"]]
    print(usable.groupby(["model", "level"])[["asl", "short_sentence_share", "term_rate",
                                              "latin_rate", "gloss_rate", "rh1"]]
          .median().round(2).to_string())


if __name__ == "__main__":
    main()
