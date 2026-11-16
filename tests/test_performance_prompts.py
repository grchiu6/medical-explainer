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
