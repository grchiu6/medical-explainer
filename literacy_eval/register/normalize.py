"""The ONE text normalizer, used on every training passage AND every LLM output.

WHAT THIS FILE DOES:
normalize(text) maps any Hindi text, whether OCR output from NCERT/ICMR scans or
an LLM answer with markdown, to the same plain form: one line of running text.

WHY ONE SHARED FUNCTION:
The classifier must separate 5th grade / 10th grade / professional by
*register*, not by where the text came from. Every rule below removes a
difference that is about the source, not the language:

  1. Unicode NFC, then drop nukta, ZWJ/ZWNJ, soft hyphen and BOM. OCR often
     misses the nukta (ज़ -> ज), while LLMs write it. Keeping it would make
     "has nuktas" a cue for LLM text.
  2. All sentence ends -> danda (।). OCR reads । as "|", HiMed writes "।.",
     and some models end Hindi sentences with ".".
  3. Every number -> "0". OCR garbles digits ("2015" -> "20:5", "6i8"), and
     ICMR is full of statistics, so "has many numbers" would be a shortcut
     to "professional". The 0 keeps the fact that a number was there.
  4. Latin-script tokens removed. In ICMR they are mostly OCR junk ("creed",
     "ceed"); in LLM outputs they are English terms. Kept, they would teach
     "English words = professional". English use in LLM outputs is reported
     separately (prepare_outputs.latin_share), not fed to the classifier.
  5. Markdown, list markers, bracketed references, quote marks and stray
     symbols removed.
  6. Line breaks and repeated whitespace collapsed into single spaces.

normalize() is idempotent: normalize(normalize(x)) == normalize(x).
"""

from __future__ import annotations

import re
import unicodedata

_INVISIBLE = dict.fromkeys(map(ord, "‌‍­﻿़"), None)   # ZWNJ, ZWJ, soft hyphen, BOM, nukta
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_LIST_MARKER = re.compile(r"^\s*(?:#+|>|[-*•●▪◦+]|\(?\d+[.)]|\(?[a-zA-Z][.)])\s+", re.M)
_EMPHASIS = re.compile(r"\*\*|__|`")
_BRACKET_REF = re.compile(r"\[[^\]\n]{0,40}\]")
_NUMBER_LIKE = re.compile(r"(?<![ऀ-ॿ])[+-]?[\dA-Za-z:.,/%]*\d[\dA-Za-z:.,/%]*")
_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_DANDA_LIKE = re.compile(r"\s*(?:\|\||॥|\|)\s*")
_DOT_AFTER_HINDI = re.compile(r"(?<=[ऀ-ॿ])\s*\.(?=\s|$|[\"'”)])")
_DANDA_DOT = re.compile(r"।\s*\.")
_ALLOWED = re.compile(r"[^ऀ-ॿ0-9\s।?!,;:()-]")   # Devanagari block (U+0900-097F); quotes go too
_EMPTY_PARENS = re.compile(r"\(\s*[,;:-]*\s*\)")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([।?!,;:)])")
_SPACE_AFTER_OPEN = re.compile(r"\(\s+")
_REPEATED_PUNCT = re.compile(r"([।?!,;:])(?:\s*\1)+")
_DANDA_RUN = re.compile(r"[।?!](?:\s*[।?!])+")
_ORPHAN_HYPHEN = re.compile(r"(?<!\S)-+(?!\S)")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return the shared normalized form of `text` (see module docstring for the rules)."""
    text = unicodedata.normalize("NFC", text or "").translate(_INVISIBLE)
    text = text.translate(_DEVANAGARI_DIGITS)
    text = _LIST_MARKER.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _BRACKET_REF.sub(" ", text)
    text = _DANDA_LIKE.sub("। ", text)
    text = _DOT_AFTER_HINDI.sub("।", text)
    text = _DANDA_DOT.sub("।", text)
    text = _NUMBER_LIKE.sub(" 0 ", text)
    text = _LATIN_TOKEN.sub(" ", text)
    text = _ALLOWED.sub(" ", text)
    text = _ORPHAN_HYPHEN.sub(" ", text)
    text = _EMPTY_PARENS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    text = _SPACE_AFTER_OPEN.sub("(", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _DANDA_RUN.sub(lambda m: m.group(0)[-1] if m.group(0)[-1] in "?!" else "।", text)
    text = _EMPTY_PARENS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return re.sub(r"^[।?!,;:\s]+", "", text)


_LATIN_COUNT = re.compile(r"[A-Za-z]{2,}")
_HINDI_COUNT = re.compile(r"[ऀ-ॣॱ-ॿ]+")


def latin_share(text: str) -> float:
    """Share of word tokens written in Latin script, measured BEFORE normalize() removes them."""
    latin = len(_LATIN_COUNT.findall(text or ""))
    hindi = len(_HINDI_COUNT.findall(text or ""))
    return latin / (latin + hindi) if latin + hindi else 0.0
