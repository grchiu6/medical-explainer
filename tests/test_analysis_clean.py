"""Tests for literacy_eval/analysis/clean.py — answer extraction and usability flags.

Every example string is (a shortened form of) a real output from
outputs/himed_trad_open/, so the rules are pinned to the failures we saw.
"""

import pandas as pd
import pytest

from literacy_eval.analysis import clean as C

TOPIC = "आयुर्वेद के अनुसार: अगर किसी व्यक्ति को सरलता से दस्त आ जाते हैं, तो उसे किस कोष्ठ का रोगी माना जाता है?"
PROMPT = (
    "आप एक चिकित्सा शिक्षक हैं।\n- सामान्य हिंदी का प्रयोग करें, कुछ चिकित्सा शब्दों के साथ\n"
    "- जब कोई चिकित्सा शब्द प्रयोग करें तो उसका सरल अर्थ भी बताएं\n- मध्यम लंबाई के वाक्य लिखें\n\n"
    f"चिकित्सा विषय: {TOPIC}\n\nहिंदी में व्याख्या:"
)


# ---------- extract_answer ----------

def test_extract_answer_splits_thinking_and_response():
    raw = "## Thinking\n\nसोच का भाग।\n\n## Response\n\nउष्ण पदार्थों से परहेज करें।."
    answer, thinking = C.extract_answer(raw)
    assert answer == "उष्ण पदार्थों से परहेज करें।"     # trailing "." after danda removed
    assert thinking == "सोच का भाग।"


def test_extract_answer_handles_numbered_thinking_heading():
    answer, thinking = C.extract_answer("## Thinking1\n\nसोच।\n\n## Response\n\nउत्तर है।")
    assert (answer, thinking) == ("उत्तर है।", "सोच।")


def test_extract_answer_thinking_without_response_is_empty_answer():
    answer, thinking = C.extract_answer("## Thinking\n\nसोच जो कभी ख़त्म नहीं हुई")
    assert answer == ""
    assert thinking.startswith("सोच")


def test_extract_answer_plain_text_passes_through():
    assert C.extract_answer("  सीधा उत्तर।  ") == ("सीधा उत्तर।", "")


# ---------- flag_response: one test per failure type ----------

def test_question_echo_is_flagged():
    answer = "आयुर्वेद के अनुसार, यदि किसी व्यक्ति को सरलता से दस्त आ जाता है, तो उसे किस कोष्ठ का रोगी माना जाता है?"
    assert C.flag_response(answer, TOPIC, PROMPT)["failure_type"] == "echo"


def test_restating_then_answering_is_not_echo():
    answer = "जिन्हें सरलता से दस्त आते हैं, उन्हें 'आम' कोष्ठ का रोगी कहा जाता है।"
    flags = C.flag_response(answer, TOPIC, PROMPT)
    assert flags["starts_with_question"] is True
    assert flags["failure_type"] == "ok"


def test_instruction_echo_is_flagged():
    answer = ("- सामान्य हिंदी का प्रयोग करें, कुछ चिकित्सा शब्दों के साथ\n"
              "- जब कोई चिकित्सा शब्द प्रयोग करें तो उसका सरल अर्थ भी बताएं\n- मध्यम लंबाई के वाक्य लिखें")
    assert C.flag_response(answer, TOPIC, PROMPT)["failure_type"] == "echo"


def test_short_refusal_is_flagged_but_long_hedged_answer_is_not():
    assert C.flag_response("मुझे खेद है, लेकिन मैं इसमें सहायता नहीं कर सकता।", TOPIC)["failure_type"] == "refusal"
    hedged = "मैं इसमें सहायता नहीं कर सकता, फिर भी " + "पित्त दोष से पाचन बिगड़ता है। " * 30
    flags = C.flag_response(hedged, TOPIC)
    assert flags["refusal"] is False and flags["refusal_language"] is True


def test_non_answer_is_flagged():
    assert C.flag_response("मैं नहीं जानता।", TOPIC)["failure_type"] == "non_answer"


@pytest.mark.parametrize("answer", [
    "पेट में दस्त (gas) के कारण पेट में दस्त " * 10,                       # 4-gram loop
    "डॉक्टर के बतलाये हुए उपायों के अनुयययययययययययययय",                      # char run
    "कोश्थ-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",                          # number list
    "ऐसे लोगों को भय,भूख,बसंत,राग,लज्जा,धिक्कार,क्रोध,प्रमाद,मूर्छ,अनिद्र,अनायास,अनाधीन,अनवर्तन,अनाध्यम,अनाधिय,अनाधिघ",
    "ड्रम奏曲 के लय पर सभी ड्रम奏曲कार अपने इलाज करते हैं",                    # CJK characters
])
def test_degenerate_outputs_are_flagged(answer):
    assert C.flag_response(answer, TOPIC)["failure_type"] == "degenerate"


def test_mostly_english_is_non_hindi():
    answer = "Ayurveda में, Koshta = organ system = body part = region of the body"
    assert C.flag_response(answer, TOPIC)["failure_type"] == "non_hindi"


def test_one_word_answer_is_too_short_but_five_words_is_ok():
    assert C.flag_response("घृत।", TOPIC)["failure_type"] == "too_short"
    assert C.flag_response("उष्ण पदार्थों से परहेज करें।", TOPIC)["failure_type"] == "ok"


def test_truncated_is_informational_only():
    flags = C.flag_response("पित्त दोष पाचन को नियंत्रित करता है और जब यह", TOPIC)
    assert flags["truncated"] is True
    assert flags["usable_rules"] is True


# ---------- build_clean_table ----------

def _write_raw(path, model, level, responses):
    pd.DataFrame({
        "topic_id": [f"trad_{i}" for i in range(len(responses))],
        "topic_hi": [TOPIC] * len(responses),
        "category": ["Ayurveda/treatment"] * len(responses),
        "prompt": [PROMPT] * len(responses),
        "model_response": responses,
    }).to_csv(path / f"{model}_{level}_himed_trad_raw.csv", index=False)


def test_build_clean_table_long_format(tmp_path):
    _write_raw(tmp_path, "himed", "low", ["## Thinking\nx\n## Response\nउष्ण पदार्थों से परहेज करें।", "घृत।"])
    table = C.build_clean_table(tmp_path)
    assert list(table["model"]) == ["himed", "himed"]
    assert list(table["system"]) == ["Ayurveda", "Ayurveda"]
    assert list(table["failure_type"]) == ["ok", "too_short"]


def test_build_clean_table_rejects_duplicate_items(tmp_path):
    _write_raw(tmp_path, "himed", "low", ["क", "ख"])
    df = pd.read_csv(tmp_path / "himed_low_himed_trad_raw.csv")
    df["topic_id"] = "trad_0"
    df.to_csv(tmp_path / "himed_low_himed_trad_raw.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        C.build_clean_table(tmp_path)


def test_empty_answer_from_unfinished_thinking_is_too_short_not_a_crash():
    answer, _ = C.extract_answer("## Thinking\n\nसोच जो कभी ख़त्म नहीं हुई")
    flags = C.flag_response(answer, TOPIC, PROMPT)
    assert flags["failure_type"] == "too_short"
    assert flags["truncated"] is False
