from performance_eval.rag.query import build_query

ITEM = {"id": "m1", "question": "प्रश्न?", "options": {"A": "x", "B": "y", "C": "z", "D": "w"}, "answer_letter": "B"}

def test_query_is_question_plus_option_texts():
    assert build_query(ITEM) == "प्रश्न?\nx\ny\nz\nw"

def test_query_does_not_depend_on_answer_letter():
    assert {build_query({**ITEM, "answer_letter": l}) for l in "ABCD"} == {build_query(ITEM)}

def test_query_ignores_unknown_fields():
    assert build_query({**ITEM, "explanation": "LEAK", "reference_answer": "LEAK"}) == build_query(ITEM)
