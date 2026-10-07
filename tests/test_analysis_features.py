"""Tests for literacy_eval/analysis/features.py — Hindi readability features.

Counts on known words were checked by hand against the definitions in
Sinha et al. (2012): a visual unit is a consonant/conjunct cluster, an
independent vowel, a vowel sign or a nasal/visarga mark.
"""

import math

import pytest

from literacy_eval.analysis import features as F


@pytest.mark.parametrize("word, aksharas, visual_units, conjuncts", [
    ("मधुमेह", 4, 6, 0),      # म धु मे ह | म ध ु म े ह
    ("स्वास्थ्य", 2, 3, 2),   # स्वा स्थ्य | स्व ा स्थ्य | स्व, स्थ्य
    ("क्या", 1, 2, 1),
    ("हिंदी", 2, 5, 0),       # हिं दी | ह ि ं द ी
    ("आयुर्वेद", 4, 6, 1),    # आ यु र्वे द
    ("है", 1, 2, 0),
])
def test_hindi_word_counts(word, aksharas, visual_units, conjuncts):
    assert F.count_aksharas(word) == aksharas
    assert F.count_visual_units(word) == visual_units
    assert F.count_conjuncts(word) == conjuncts


def test_split_sentences_on_danda_question_newline_and_period():
    assert F.split_sentences("यह एक है। वह दो है? तीसरा.\nचौथा") == ["यह एक है", " वह दो है", " तीसरा", "चौथा"]


def test_compute_features_simple_text():
    f = F.compute_features("यह दवा है। वह दवा है।", "दवा क्या है?")
    assert f["n_words"] == 6 and f["n_sentences"] == 2
    assert f["asl"] == 3.0
    assert f["latin_rate"] == 0.0
    # यह, वह, है are stopwords, so the only content word is दवा — which is in the question
    assert f["copy_rate"] == 1.0


def test_compute_features_counts_english_words():
    f = F.compute_features("मधुमेह (diabetes) एक रोग है।", "")
    assert f["latin_rate"] == pytest.approx(1 / 5)


def test_rh1_matches_formula():
    f = F.compute_features("मधुमेह रोग है।", "")
    expected = -2.34 + 2.14 * f["awl"] + 0.01 * (f["psw_rate"] * 100)
    assert f["rh1"] == pytest.approx(expected)


def test_common_words_have_higher_zipf_than_rare_terms():
    common = F.compute_features("यह बहुत अच्छा है।", "")
    rare = F.compute_features("कोष्ठ विरेचन अभिष्यन्द।", "")
    assert common["mean_zipf"] > rare["mean_zipf"]
    assert rare["rare_rate"] > common["rare_rate"]


def test_empty_or_non_hindi_answer_gives_nan():
    assert all(math.isnan(v) for v in F.compute_features("", "").values())
    assert math.isnan(F.compute_features("only english here", "")["awl"])


# ---------- instruction-compliance and terminology features ----------

def test_short_sentence_share_counts_5_to_10_word_sentences():
    text = "यह एक बहुत छोटा वाक्य है।" + " यह वाक्य बहुत लंबा है क्योंकि इसमें बारह से भी अधिक शब्द लिखे गए हैं।"
    f = F.compute_features(text, "")
    assert f["short_sentence_share"] == 0.5          # 6 words: yes; 15 words: no


def test_gloss_rate_counts_parenthetical_explanations_per_100_words():
    text = "मधुमेह (शुगर की बीमारी) एक रोग है और रक्तचाप (बीपी) भी एक रोग है।"
    f = F.compute_features(text, "")
    assert f["gloss_rate"] == pytest.approx(2 / f["n_words"] * 100)


def test_build_term_lexicon_keeps_rare_words_from_reference_items():
    items = [{"topic_hi": "आयुर्वेद के अनुसार: कोष्ठ क्या है?", "options": {"A": "मृदु कोष्ठ", "B": "है"},
              "cot": "विरेचन से दस्त होते हैं।", "key_facts": ["अभिष्यन्द एक रोग है"]}]
    terms = F.build_term_lexicon(items)
    assert {"कोष्ठ", "विरेचन", "अभिष्यन्द"} <= terms
    assert "है" not in terms and "के" not in terms    # common words are not terms


def test_term_rate_uses_lexicon():
    terms = frozenset({"कोष्ठ", "विरेचन"})
    f = F.compute_features("मृदु कोष्ठ में विरेचन जल्दी होता है।", "", terms)
    assert f["term_rate"] == pytest.approx(2 / 7)          # 7 words, 2 of them terms
    assert F.compute_features("मृदु कोष्ठ में विरेचन जल्दी होता है।", "")["term_rate"] == 0.0
