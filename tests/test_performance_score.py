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
