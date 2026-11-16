import pytest

from performance_eval.score import extract_answer, score_rows


def test_after_marker_extracts_letter_following_english_marker():
    text = "यह बीमारी के कारण... इसलिए सही विकल्प है।\nAnswer: C"
    assert extract_answer(text, mode="after_marker") == "C"


def test_after_marker_falls_back_to_last_standalone_letter_when_marker_missing():
    text = "विकल्प A गलत है। विकल्प B सही है।"
    assert extract_answer(text, mode="after_marker") == "B"


def test_after_marker_returns_none_when_no_letter_anywhere():
    text = "यह उत्तर देना मुश्किल है।"
    assert extract_answer(text, mode="after_marker") is None


def test_after_marker_prefers_marker_over_earlier_mentions():
    text = "शुरू में लगता है A सही है, लेकिन सोचने पर...\nAnswer: D"
    assert extract_answer(text, mode="after_marker") == "D"


def test_first_letter_mode_delegates_to_score_mcq_extract_letter():
    # mirrors score_mcq.extract_letter's own documented behavior
    assert extract_answer("A. वात रोग", mode="first_letter") == "A"
    assert extract_answer("", mode="first_letter") is None


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        extract_answer("Answer: A", mode="bogus")


def test_score_rows_marks_correct_and_incorrect():
    rows = [
        {"item_id": "x1", "reference_answer": "B", "model_answer": "Answer: B"},
        {"item_id": "x2", "reference_answer": "A", "model_answer": "Answer: C"},
    ]
    scored = score_rows(rows, mode="after_marker")
    assert scored[0]["is_correct"] is True
    assert scored[1]["is_correct"] is False


def test_score_rows_counts_unparseable_as_incorrect_not_skipped():
    rows = [{"item_id": "x1", "reference_answer": "A", "model_answer": "कुछ नहीं पता"}]
    scored = score_rows(rows, mode="after_marker")
    assert scored[0]["parsed_letter"] is None
    assert scored[0]["is_correct"] is False  # NONE counts as wrong (spec resolution 4)


def test_score_rows_passes_input_keys_through_and_normalizes_reference():
    rows = [{"item_id": "x1", "reference_answer": " b ", "model_answer": "Answer: B"}]
    scored = score_rows(rows, mode="after_marker")
    assert scored[0]["item_id"] == "x1"
    assert scored[0]["reference_answer"] == " b "
    assert scored[0]["model_answer"] == "Answer: B"
    assert scored[0]["parsed_letter"] == "B"
    assert scored[0]["is_correct"] is True


def test_score_rows_invalid_reference_raises_naming_item_id():
    rows = [{"item_id": "bad42", "reference_answer": "K", "model_answer": "Answer: K"}]
    with pytest.raises(ValueError, match="bad42"):
        score_rows(rows, mode="after_marker")


# ---- C1 / R18: HiMed "## Thinking ... ## Response" ----

THINK_THEN_RESPONSE = "## Thinking\nविकल्प A पर विचार करें, फिर विकल्प C...\n## Response\nB"
TRUNCATED_THINKING = "## Thinking\nविकल्प A पर विचार करें, फिर विकल्प C"


def test_first_letter_mode_scores_only_the_response_part():
    assert extract_answer(THINK_THEN_RESPONSE, mode="first_letter") == "B"


def test_after_marker_mode_scores_only_the_response_part():
    text = "## Thinking\nAnswer: A लगता है...\n## Response\nतर्क...\nAnswer: D"
    assert extract_answer(text, mode="after_marker") == "D"


@pytest.mark.parametrize("mode", ["first_letter", "after_marker"])
def test_truncated_thinking_without_response_returns_none(mode):
    assert extract_answer(TRUNCATED_THINKING, mode=mode) is None


def test_thinking_digit_variant_is_stripped():
    text = "## Thinking1\nविकल्प A गलत...\n## Response\nC"
    assert extract_answer(text, mode="first_letter") == "C"
    assert extract_answer("## Thinking1\nविकल्प A", mode="after_marker") is None


def test_plain_text_without_headers_is_unchanged():
    assert extract_answer("A. वात रोग", mode="first_letter") == "A"
    assert extract_answer("तर्क...\nAnswer: B", mode="after_marker") == "B"


# ---- I6 / R24: marker_found ----

def test_score_rows_marker_found_true_only_for_explicit_marker():
    rows = [
        {"item_id": "x1", "reference_answer": "B", "model_answer": "तर्क...\nAnswer: B"},
        {"item_id": "x2", "reference_answer": "B", "model_answer": "विकल्प A गलत, विकल्प B सही"},
        {"item_id": "x3", "reference_answer": "B", "model_answer": "कुछ नहीं"},
    ]
    scored = score_rows(rows, mode="after_marker")
    assert [r["marker_found"] for r in scored] == [True, False, False]
    assert scored[1]["parsed_letter"] == "B"  # fallback still parses


def test_score_rows_marker_found_false_in_first_letter_mode():
    rows = [{"item_id": "x1", "reference_answer": "B", "model_answer": "Answer: B"}]
    assert score_rows(rows, mode="first_letter")[0]["marker_found"] is False


# ---- R26 minor 1: out-of-range letters ----

Q4 = "प्रश्न?\nA. एक\nB. दो\nC. तीन\nD. चार"


@pytest.mark.parametrize("answer", ["E", "J", "I think A"])
def test_letter_beyond_option_count_becomes_none(answer):
    rows = [{"item_id": "x1", "reference_answer": "A", "question_with_options": Q4,
             "model_answer": answer}]
    scored = score_rows(rows, mode="first_letter")
    assert scored[0]["parsed_letter"] is None
    assert scored[0]["is_correct"] is False


def test_n_options_key_takes_precedence_over_counting_lines():
    rows = [{"item_id": "x1", "reference_answer": "A", "n_options": 5,
             "question_with_options": Q4, "model_answer": "E"}]
    assert score_rows(rows, mode="first_letter")[0]["parsed_letter"] == "E"


def test_in_range_letter_kept_and_after_marker_out_of_range_is_none():
    rows = [{"item_id": "x1", "reference_answer": "D", "question_with_options": Q4,
             "model_answer": "Answer: D"},
            {"item_id": "x2", "reference_answer": "D", "question_with_options": Q4,
             "model_answer": "Answer: F"}]
    scored = score_rows(rows, mode="after_marker")
    assert scored[0]["parsed_letter"] == "D"
    assert scored[1]["parsed_letter"] is None


# ---- R26 minor 2: Hindi marker ----

def test_after_marker_accepts_hindi_marker():
    assert extract_answer("तर्क... विकल्प A गलत।\nउत्तर: C", mode="after_marker") == "C"
    assert extract_answer("तर्क...\nउत्तर B", mode="after_marker") == "B"
    rows = [{"item_id": "x1", "reference_answer": "C", "model_answer": "तर्क...\nउत्तर: C"}]
    assert score_rows(rows, mode="after_marker")[0]["marker_found"] is True
