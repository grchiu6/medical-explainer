"""Tests for the shared prompt template (models/base.py).

These tests verify that build_prompt() creates prompts that:
1. Include the question and context we pass in
2. Work correctly even when no context is provided
"""

from models.base import build_mcq_prompt, build_prompt


def test_build_prompt_contains_question_and_context():
    """The generated prompt must include both the question and context text.

    Why this test matters: If the prompt accidentally drops the context
    or question, the model would be answering blindly without the
    information it needs. This catches that bug.
    """
    prompt = build_prompt(
        question="मधुमेह के लक्षण क्या हैं?",
        context="मधुमेह एक चयापचय रोग है।"
    )
    # The prompt string should contain our exact question and context
    assert "मधुमेह के लक्षण क्या हैं?" in prompt
    assert "मधुमेह एक चयापचय रोग है।" in prompt


def test_build_prompt_works_without_context():
    """The prompt should still work when context is an empty string.

    Why this test matters: Some questions might not have context.
    The function shouldn't crash or produce a broken prompt in that case.
    """
    prompt = build_prompt(
        question="मधुमेह के लक्षण क्या हैं?",
        context=""
    )
    assert "मधुमेह के लक्षण क्या हैं?" in prompt


def test_build_mcq_prompt_contains_question_with_options():
    """MCQ prompt must include the full question text (with embedded A-E options).

    Why: For HiMed, the options are part of the question string itself
    (\"Question?\\nA. ...\\nB. ...\"). If the prompt builder strips them,
    the model has no choices to pick from.
    """
    question_with_options = (
        "जीवाणु के कौन से तत्व से निर्मित होने की बात कही गई है?\n"
        "A. पञ्चमहाभूत\nB. विशुद्ध रासायनिक यौगिक\n"
        "C. मिथकीय पदार्थ\nD. नाभिकीय कण\nE. बिजली"
    )
    prompt = build_mcq_prompt(question_with_options)

    assert "जीवाणु के कौन से तत्व" in prompt, "Question stem should be in prompt"
    assert "A. पञ्चमहाभूत" in prompt, "Option A should be in prompt"
    assert "E. बिजली" in prompt, "Option E should be in prompt"


def test_build_mcq_prompt_instructs_letter_only():
    """MCQ prompt must instruct the model to answer with only a letter.

    Why: We want the model to output just 'A' (or 'B' etc.), not a full
    explanation. The simpler the output, the easier to parse + score.
    Without this instruction, models tend to write paragraphs.
    """
    prompt = build_mcq_prompt("Test question?\nA. one\nB. two\nC. three\nD. four\nE. five")

    # The instruction should mention all 5 letters as valid choices
    # AND should clearly say to output only the letter.
    prompt_lower = prompt.lower()
    assert "letter" in prompt_lower, "Prompt should mention 'letter'"
    # Mentions of the letter range — flexible match (could be "A-E", "A, B, C, D, or E", etc.)
    has_letter_range = any(
        marker in prompt for marker in ["A-E", "A, B, C, D, or E", "(A, B, C, D, E)"]
    )
    assert has_letter_range, "Prompt should specify the valid letter range"
