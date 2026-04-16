"""Tests for the shared prompt template (models/base.py).

These tests verify that build_prompt() creates prompts that:
1. Include the question and context we pass in
2. Work correctly even when no context is provided
"""

from models.base import build_prompt


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
