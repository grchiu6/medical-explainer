"""Tests for literacy_eval/register/normalize.py — the shared normalizer.

Example strings come from real OCR output (ICMR Patrika, NCERT) and from the
HiMed-Trad LLM outputs, so each rule is pinned to an artifact we actually saw.
"""

import pytest

from literacy_eval.register.normalize import latin_share, normalize


def test_ocr_pipe_becomes_danda():
    # ICMR OCR: "...रहती S| अतः" and "...पाई हैं|"
    assert normalize("परिवार नियोजन संबंधी जरूरतें पूरी नहीं हो पाई हैं|") == "परिवार नियोजन संबंधी जरूरतें पूरी नहीं हो पाई हैं।"


def test_himed_danda_dot_and_hindi_full_stop():
    assert normalize("उष्ण पदार्थों से परहेज करें।.") == "उष्ण पदार्थों से परहेज करें।"
    assert normalize("यह एक रोग है. इसका इलाज है.") == "यह एक रोग है। इसका इलाज है।"


@pytest.mark.parametrize("raw", ["वर्ष 2015 में", "वर्ष 20:5 में", "वर्ष 20i5 में", "वर्ष २०१५ में"])
def test_numbers_and_ocr_garbled_numbers_become_zero(raw):
    assert normalize(raw) == "वर्ष 0 में"


def test_percentages_and_decimals_become_zero():
    assert normalize("कुल प्रजनन क्षमता दर (3.12) है और 48% महिलाएं") == "कुल प्रजनन क्षमता दर (0) है और 0 महिलाएं"


def test_latin_ocr_junk_and_english_terms_removed():
    assert normalize("की वर्तमान सभी ceed आगामी") == "की वर्तमान सभी आगामी"
    # the parentheses left empty by removing the English gloss go too
    assert normalize('प्लास्मोडियम ("Plasmodium") परजीवी') == "प्लास्मोडियम परजीवी"


def test_nukta_and_invisible_characters_removed():
    assert normalize("ज़रूरी") == normalize("जरूरी") == "जरूरी"
    assert normalize("क‍क्षा­") == "कक्षा"


def test_markdown_and_list_markers_removed():
    raw = "## कारण\n\n- **पहला** कारण है।\n1. दूसरा कारण है।\n* तीसरा।"
    assert normalize(raw) == "कारण पहला कारण है। दूसरा कारण है। तीसरा।"


def test_ncert_bullet_question_line():
    assert normalize("+ खेल में झुम्पा ने मछली कैसे पहचान ली?") == "खेल में झुम्पा ने मछली कैसे पहचान ली?"


def test_line_breaks_reflowed_into_running_text():
    raw = "भारत की आधी आबादी प्रजननशील आयु वर्ग के अन्तर्गत है\nऔर अधिकतर आबादी गांवों में रहती है।"
    assert normalize(raw) == "भारत की आधी आबादी प्रजननशील आयु वर्ग के अन्तर्गत है और अधिकतर आबादी गांवों में रहती है।"


def test_bracketed_reference_numbers_removed():
    assert normalize("यह सिद्ध हुआ है [12]। आगे") == "यह सिद्ध हुआ है। आगे"


def test_question_mark_kept():
    assert normalize("क्या तुम जानते हो?") == "क्या तुम जानते हो?"


@pytest.mark.parametrize("raw", [
    "## Thinking\nसोच\n## Response\n- **पहला** कारण (fever) है|| वर्ष 20:5 में 48% [3].",
    "+ स्वाद Usa\nअलग-अलग स्वाद की कुछ चीजें इकड्ठी करो",
    "",
    "   ",
])
def test_idempotent(raw):
    once = normalize(raw)
    assert normalize(once) == once


def test_latin_share_counts_before_normalization():
    assert latin_share("मधुमेह diabetes है") == pytest.approx(1 / 3)
    assert latin_share("") == 0.0
