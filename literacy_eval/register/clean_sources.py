"""Stage 3 of the register corpus: keep only running prose from the OCR'd pages.

WHAT THIS FILE DOES:
Reads Tesseract's .tsv output (every word with its paragraph number and a
confidence score) and rebuilds paragraphs in reading order. Then it REMOVES
material that is not prose, in four layers:

  page      drop whole pages: book front matter / answer keys, NCERT teacher
            pages ("शिक्षकों के लिए"), ICMR contents / editorial / news pages,
            ICMR pages with too few health words (non-medical articles)
  paragraph drop low-OCR-confidence paragraphs, short ones (headings,
            captions, mastheads, junk like "aE ii"), mostly non-Devanagari
            ones (reference lists), NCERT teacher boxes, ICMR author lines
  sentence  (NCERT only) drop exercise/activity sentences addressed to the
            student: "...नाम लिखिए", "तुम ... पहचान सकते हो?", "आइए, ..."
  furniture strip print artifacts inside text: "Chapter 1.indd 5",
            date stamps, NCERT codes like "0536CH01"

WHY ONLY REMOVE, NEVER REWRITE:
The classifier learns from the language that is left, so cleaning must not
change it. Rewriting is normalize.py's job, and normalize applies identically
to every source and to the LLM outputs.

Every rule counts what it drops (DropCounter), so build_corpus can report it
and we can check that no rule is removing too much.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from literacy_eval.register.ocr import page_out_dir
from literacy_eval.register.sources import RAW_DIR

# ---------- thresholds (calibrated on OCR samples; see corpus_report.md) ----------
MIN_PARA_CONF = 70           # mean Tesseract word confidence (0-100)
MIN_PARA_HINDI_WORDS = 6     # shorter = heading, caption, masthead or junk
MIN_DEVANAGARI_SHARE = 0.85  # share of letters that are Devanagari
MIN_ICMR_HEALTH_HITS = 3     # health-term hits needed to keep an ICMR page

# NCERT chapter files that are not chapters: front matter (ps) and answer keys (an).
EXCLUDED_FILE_SUFFIXES = ("ps", "an")
# Class 10 Science (jhsc1): keep biology/health chapters only (topic control).
# 5 जैव प्रक्रम, 6 नियंत्रण एवं समन्वय, 7 जीव जनन कैसे करते हैं, 8 आनुवंशिकता, 13 हमारा पर्यावरण
JHSC1_KEEP = {"jhsc105", "jhsc106", "jhsc107", "jhsc108", "jhsc113"}

_FURNITURE = re.compile(
    r"Chapter\s*\S{0,3}\.?\s*indd\s*\d*"          # "Chapter l.indd 5"
    r"|\d{1,2}\s*-\s*\d{1,2}\s*-\s*\d{2,4}\s*[\d:.\s]*"   # "27-00-2026 4:7:"
    r"|\b\d{4}\s*CH\s*\d+\b"                       # "0536CH01"
    r"|(?:Reprint|rePrint)\s*\d{4}-\d{2,4}",
    re.I,
)
_DEVANAGARI_LETTER = re.compile(r"[ऀ-ॣॱ-ॿ]")
_LETTER = re.compile(r"[ऀ-ॣॱ-ॿA-Za-z]")
_HINDI_WORD = re.compile(r"[ऀ-ॣॱ-ॿ]+")
_SENTENCE = re.compile(r"[^।?!|]+[।?!|]?")

# NCERT teacher-facing text (adult register inside a Class 5 book).
_TEACHER_PAGE = re.compile(r"शिक्षकों के लिए|इकाई के विषय में|अध्यापकों के लिए|शिक्षक संकेत")
_TEACHER_PARA = re.compile(
    r"विद्यार्थियों|विद्यार्थी|शिक्षक|अध्यापक|इस इकाई|यह इकाई|इस अध्याय में|अधिगम|प्रोत्साहित कीजिए|सुगमकर्ता|सगमकर्ता"
)
# NCERT exercise / activity sentences addressed to the student.
_EXERCISE_VERB = re.compile(
    r"(?:लिखि|लिखो|पहचानि|पहचानो|बताइ|बताओ|चर्चा की|चर्चा करो|पता की|पता करो|सूची बना|चित्र बना"
    r"|भरि|भरो|मिलान|गतिविधि|क्रियाकलाप|प्रयोग करके देख|सोचि|सोचो|अनुमान लगा)"
)
_LETS_DO = re.compile(r"^\s*आइए")
_READER_PRONOUN = re.compile(r"(?<![ऀ-ॿ])(?:तुम|तुम्हें|तुमने|तुमको|तुम्हारे|तुम्हारी|तुम्हारा|आप|आपको|आपके|आपकी|आपका|आपने)(?![ऀ-ॿ])")
# Commands to the student: polite "-इए/-िए" (देखिए, कीजिए, बनाइए) and familiar "-ओ" (देखो, लगाओ, करो).
# Words that merely END in -िए but are not commands (लिए "for", चाहिए "should", किए/दिए past forms) are excluded.
_POLITE_IMPERATIVE = re.compile(r"(?<![ऀ-ॿ])[ऀ-ॿ]+(?:िए|इए)(?![ऀ-ॿ])")
_NOT_IMPERATIVE = {"लिए", "इसलिए", "किसलिए", "जिसलिए", "उसलिए", "चाहिए", "किए", "दिए", "पिए", "जिए", "सिए", "छुए"}
_FAMILIAR_IMPERATIVE = re.compile(
    r"(?<![ऀ-ॿ])(?:देखो|करो|लगाओ|बनाओ|लिखो|बताओ|सोचो|पूछो|खोजो|ढूँढो|ढूंढो|गिनो|चखो|सूँघो|छुओ|खेलो"
    r"|सुनो|पढ़ो|पढो|बोलो|भरो|जोड़ो|जोडो|मिलाओ|रखो|लाओ|दिखाओ|समझाओ|पहचानो)(?![ऀ-ॿ])"
)
_EXERCISE_HEADING = re.compile(r"^\s*(?:प्रश्न|अभ्यास|क्रियाकलाप|चित्र\s*\d|सारणी)")

# ICMR pages that are not research prose.
_ICMR_NON_ARTICLE = re.compile(
    r"संपादक मंडल|सम्पादक मंडल|प्रधान संपादक|मुख्य संपादक|विषय सूची|अनुक्रमणिका|हिन्दी पखवाड़|हिंदी पखवाड़"
    r"|राजभाषा|पुरस्कार समारोह|संस्थान समाचार|कार्यशाला का आयोजन|सेवानिवृत्त|नियुक्ति"
)
# Event reports (meetings, conferences, inaugurations): >= MIN_EVENT_HITS hits drops the page.
# ("सम्पन्न"/"आयोजित" are NOT used: research articles say "अध्ययन सम्पन्न किया गया" = "the study was conducted".)
_ICMR_EVENT = re.compile(r"बैठक|सम्मेलन|संगोष्ठी|कार्यशाला|उद्घाटन|मुख्य अतिथि|अध्यक्षता|स्वागत|धन्यवाद ज्ञापन|अभियान का आयोजन")
MIN_EVENT_HITS = 3
# Sentences that name the publisher or a person: source markers, not register (found by shortcut_check).
_ICMR_SOURCE_SENTENCE = re.compile(
    r"आई\s*सी\s*एम\s*आर|आईसीएमआर|आयुर्विज्ञान अनुसंधान परिषद|परिषद|(?<![ऀ-ॿ])(?:डॉ|डा)\.?(?![ऀ-ॿ])"
    r"|महानिदेशक|सचिव"
)
_ICMR_AUTHOR_LINE = re.compile(r"(?:डॉ|डा)\.?\s|वैज्ञानिक\s*['\"]?[ई-ह]|संस्थान,|परिषद,|नई दिल्ली\s*$")
_HEALTH_TERMS = (
    "रोग", "बीमारी", "स्वास्थ्य", "उपचार", "चिकित्सा", "संक्रमण", "रोगी", "लक्षण", "निदान", "औषध", "दवा",
    "टीका", "टीकाकरण", "पोषण", "रक्त", "कैंसर", "मधुमेह", "क्षयरोग", "मलेरिया", "विषाणु", "जीवाणु",
    "गर्भ", "शिशु", "मृत्यु", "हृदय", "मानसिक", "अस्पताल", "प्रतिरोध", "आहार",
)


@dataclass
class Paragraph:
    text: str
    conf: float


class DropCounter(Counter):
    """Counts dropped units per rule, e.g. drops['para:short'] += 1."""


def page_paragraphs(tsv_path: Path) -> list[Paragraph]:
    """Rebuild paragraphs from a Tesseract TSV, in reading order, with mean word confidence."""
    groups: dict[tuple[int, int], list[tuple[str, float]]] = {}
    with tsv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            word = (row.get("text") or "").strip()
            conf = float(row.get("conf") or -1)
            if row.get("level") != "5" or not word or conf < 0:
                continue
            key = (int(row["block_num"]), int(row["par_num"]))
            groups.setdefault(key, []).append((word, conf))
    paragraphs = []
    for words in groups.values():   # dict keeps insertion order = Tesseract's reading order
        text = " ".join(w for w, _ in words)
        paragraphs.append(Paragraph(text=text, conf=sum(c for _, c in words) / len(words)))
    return paragraphs


def strip_furniture(text: str) -> str:
    return re.sub(r"\s+", " ", _FURNITURE.sub(" ", text)).strip()


def devanagari_share(text: str) -> float:
    letters = _LETTER.findall(text)
    return len(_DEVANAGARI_LETTER.findall(text)) / len(letters) if letters else 0.0


def is_exercise_sentence(sentence: str) -> bool:
    """A sentence addressed to the student as an exercise or activity instruction."""
    if _LETS_DO.search(sentence) or _EXERCISE_HEADING.search(sentence) or _EXERCISE_VERB.search(sentence):
        return True
    if _FAMILIAR_IMPERATIVE.search(sentence):
        return True
    if any(w not in _NOT_IMPERATIVE and not w.endswith("लिए") for w in _POLITE_IMPERATIVE.findall(sentence)):
        return True
    return sentence.rstrip().endswith("?") and bool(_READER_PRONOUN.search(sentence))


def drop_exercise_sentences(text: str, drops: DropCounter) -> str:
    kept = []
    for sentence in _SENTENCE.findall(text):
        if not sentence.strip():
            continue
        if is_exercise_sentence(sentence):
            drops["sentence:ncert_exercise"] += 1
        else:
            kept.append(sentence.strip())
    return " ".join(kept)


def drop_icmr_source_sentences(text: str, drops: DropCounter) -> str:
    kept = []
    for sentence in _SENTENCE.findall(text):
        if not sentence.strip():
            continue
        if _ICMR_SOURCE_SENTENCE.search(sentence):
            drops["sentence:icmr_source_marker"] += 1
        else:
            kept.append(sentence.strip())
    return " ".join(kept)


def health_hits(text: str) -> int:
    return sum(text.count(term) for term in _HEALTH_TERMS)


def keep_file(source: str, file: str) -> bool:
    stem = Path(file).stem
    if source != "ncert":
        return True
    if stem.endswith(EXCLUDED_FILE_SUFFIXES):
        return False
    return not stem.startswith("jhsc1") or stem in JHSC1_KEEP


def clean_page(source: str, paragraphs: list[Paragraph], drops: DropCounter) -> list[str]:
    """Return the prose paragraphs of one page (possibly none)."""
    page_text = " ".join(p.text for p in paragraphs)
    if source == "ncert" and _TEACHER_PAGE.search(page_text):
        drops["page:ncert_teacher"] += 1
        return []
    if source == "icmr":
        if _ICMR_NON_ARTICLE.search(page_text):
            drops["page:icmr_non_article"] += 1
            return []
        if health_hits(page_text) < MIN_ICMR_HEALTH_HITS:
            drops["page:icmr_non_medical"] += 1
            return []
        if len(_ICMR_EVENT.findall(page_text)) >= MIN_EVENT_HITS:
            drops["page:icmr_event_report"] += 1
            return []

    kept = []
    for para in paragraphs:
        text = strip_furniture(para.text)
        if para.conf < MIN_PARA_CONF:
            drops["para:low_confidence"] += 1
            continue
        if devanagari_share(text) < MIN_DEVANAGARI_SHARE:
            drops["para:non_devanagari"] += 1
            continue
        if source == "ncert":
            if _TEACHER_PARA.search(text):
                drops["para:ncert_teacher"] += 1
                continue
            text = drop_exercise_sentences(text, drops)
        else:
            if _ICMR_AUTHOR_LINE.search(text) and len(_HINDI_WORD.findall(text)) < 25:
                drops["para:icmr_author_line"] += 1
                continue
            text = drop_icmr_source_sentences(text, drops)
        if len(_HINDI_WORD.findall(text)) < MIN_PARA_HINDI_WORDS:
            drops["para:short"] += 1
            continue
        kept.append(text)
    return kept


def clean_file(source: str, file: str, pages: int, drops: DropCounter) -> list[dict]:
    """Clean every OCR'd page of one PDF. Returns [{page, paragraph_index, text}, ...]."""
    out_dir = page_out_dir(file)
    rows = []
    for page in range(1, pages + 1):
        tsv = out_dir / f"p{page:03d}.tsv"
        if not tsv.exists():
            drops["page:missing_ocr"] += 1
            continue
        for i, text in enumerate(clean_page(source, page_paragraphs(tsv), drops)):
            rows.append({"page": page, "paragraph_index": i, "text": text})
    return rows


__all__ = ["DropCounter", "Paragraph", "RAW_DIR", "clean_file", "clean_page", "keep_file", "page_paragraphs"]
