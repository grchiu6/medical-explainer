import pytest
from performance_eval.prompts import build_prompt


def test_baseline_prompt_has_no_cot_or_fewshot_instructions():
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2)
    assert "Answer:" not in p  # no marker required — strict single-letter only
    assert "step by step" not in p.lower()
    assert "प्रश्न?" in p


def test_baseline_prompt_lists_exact_option_letters():
    p = build_prompt("प्रश्न?\nA. एक\nB. दो\nC. तीन", n_options=3)
    assert "A, B, or C" in p  # not "A, B, C, D, or E"


def test_cot_prompt_requires_answer_marker_and_reasoning():
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2, use_cot=True)
    assert "step by step" in p.lower() or "step-by-step" in p.lower()
    assert "Answer:" in p


def test_fewshot_prompt_renders_exemplars_before_target_question_in_order():
    exemplars = [
        {"question_with_options": "उदाहरण 1?\nA. क\nB. ख", "answer_letter": "A"},
        {"question_with_options": "उदाहरण 2?\nA. ग\nB. घ", "answer_letter": "B"},
    ]
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2, exemplars=exemplars)
    assert p.index("उदाहरण 1?") < p.index("उदाहरण 2?") < p.index("प्रश्न?")
    assert "Answer: A" in p and "Answer: B" in p  # exemplar answers appear


def test_fewshot_and_cot_compose():
    exemplars = [{"question_with_options": "उदाहरण?\nA. क\nB. ख", "answer_letter": "A"}]
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2, use_cot=True, exemplars=exemplars)
    assert "Answer:" in p
    assert "उदाहरण?" in p


def test_build_prompt_handles_ten_options():
    stem = "प्रश्न?\n" + "\n".join(f"{chr(65+i)}. विकल्प{i}" for i in range(10))
    p = build_prompt(stem, n_options=10)
    assert "A, B, C, D, E, F, G, H, I, or J" in p


def test_no_stray_blank_lines():
    ex = [{"question_with_options": "q?\nA. क\nB. ख", "answer_letter": "A"}]
    for cot in (False, True):
        for e in (None, ex):
            assert "\n\n\n" not in build_prompt("प्रश्न?\nA. एक\nB. दो", 2, cot, e)


def test_cot_prompt_ends_with_reasoning_cue_not_bare_answer():
    cue = "Reasoning (step by step, in Hindi):"
    ex = [{"question_with_options": "q?\nA. क\nB. ख", "answer_letter": "A"}]
    for e in (None, ex):
        p = build_prompt("प्रश्न?\nA. एक\nB. दो", 2, use_cot=True, exemplars=e)
        assert p.rstrip().endswith(cue)
        assert not p.rstrip().endswith("Answer")


_GQ = "प्रश्न?\nA. a\nB. b\nC. c\nD. d"
# Captured from build_prompt BEFORE the passages change.
_GOLDEN_BASELINE = 'You are answering a multiple-choice medical question in Hindi. Read the question and the options carefully, then respond with ONLY the letter of the correct answer (A, B, C, or D). Do not explain. Do not write anything except the single letter.\n\nQuestion: प्रश्न?\nA. a\nB. b\nC. c\nD. d\n\nAnswer (single letter only):'
_GOLDEN_COT = 'You are answering a multiple-choice medical question in Hindi. Read the question and the options carefully. Think step by step in Hindi. On the final line, write exactly `Answer: X` where X is the single letter of the correct answer (A, B, C, or D). Do not write anything after that line.\n\nQuestion: प्रश्न?\nA. a\nB. b\nC. c\nD. d\n\nReasoning (step by step, in Hindi):'


def test_prompt_without_passages_unchanged():
    assert build_prompt(_GQ, 4) == _GOLDEN_BASELINE
    assert build_prompt(_GQ, 4, passages=None) == _GOLDEN_BASELINE
    assert build_prompt(_GQ, 4, use_cot=True, passages=None) == _GOLDEN_COT


def test_prompt_with_passages_starts_with_block_and_truncates():
    p = build_prompt(_GQ, 4, passages=[{"title": "T1", "content": "x" * 1500}, {"title": "T2", "content": "y"}])
    assert p.startswith("Reference passages (in English). They may or may not be relevant to the question. Use them only if they help.\n\n[1] T1\n")
    assert "[1] T1\n" + "x" * 1000 + "\n\n[2] T2\ny" in p and "x" * 1001 not in p
    assert p.endswith("Answer (single letter only):")
    assert _GOLDEN_BASELINE in p


def test_passages_block_comes_first_with_cot_and_fewshot():
    ex = [{"question_with_options": "E?\nA. a", "answer_letter": "A"}]
    p = build_prompt(_GQ, 4, use_cot=True, exemplars=ex, passages=[{"title": "T", "content": "c"}])
    assert p.startswith("Reference passages") and p.index("Here are example questions") > p.index("[1] T")
    assert p.endswith("Reasoning (step by step, in Hindi):")


from performance_eval.prompts import build_prompt as _bp  # noqa: E402

_Q = 'प्रश्न\nA. a\nB. b\nC. c\nD. d'
_HINDI_BASE = 'You are answering a multiple-choice medical question in Hindi. Read the question and the options carefully, then respond with ONLY the letter of the correct answer (A, B, C, or D). Do not explain. Do not write anything except the single letter.\n\nQuestion: प्रश्न\nA. a\nB. b\nC. c\nD. d\n\nAnswer (single letter only):'
_HINDI_COT = 'You are answering a multiple-choice medical question in Hindi. Read the question and the options carefully. Think step by step in Hindi. On the final line, write exactly `Answer: X` where X is the single letter of the correct answer (A, B, C, or D). Do not write anything after that line.\n\nQuestion: प्रश्न\nA. a\nB. b\nC. c\nD. d\n\nReasoning (step by step, in Hindi):'


def test_hindi_prompt_unchanged():
    assert _bp(_Q, 4) == _HINDI_BASE
    assert _bp(_Q, 4, use_cot=True) == _HINDI_COT
    assert _bp(_Q, 4, language="hi") == _HINDI_BASE


@pytest.mark.parametrize("cot", [False, True])
def test_english_prompt_says_english(cot):
    p = _bp("Which?\nA. a\nB. b", 2, use_cot=cot, language="en")
    assert "in English" in p and "in Hindi" not in p


def test_unknown_language_raises():
    with pytest.raises(ValueError):
        _bp(_Q, 4, language="fr")
