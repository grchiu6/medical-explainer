"""Tests for Gemini model integration.

NOTE: This test calls the real Gemini API, so it:
- Requires GEMINI_API_KEY to be set in .env
- Will be SKIPPED if the key isn't available
- Uses one API call (counts against your free tier quota)
"""

import os

import pytest

# Load .env so the test can check for the API key
from dotenv import load_dotenv
load_dotenv()

from models.base import build_prompt
from models.gemini import get_answer


# pytest.mark.skipif is a "decorator" that tells pytest:
# "Only run this test IF the condition is False."
# If GEMINI_API_KEY is not set, the test gets skipped (not failed).
# This is important because we don't want tests to fail just because
# someone hasn't set up their API key yet.
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set — skipping live API test"
)
def test_get_answer_returns_nonempty_string():
    """Gemini should return a non-empty string for a Hindi QA question.

    This is a 'smoke test' — it doesn't check if the answer is CORRECT,
    just that the API call works and returns something. We check correctness
    later with BLEU/ROUGE scoring across the full dataset.
    """
    prompt = build_prompt(
        question="मधुमेह के लक्षण क्या हैं?",
        context="मधुमेह एक चयापचय रोग है जिसमें रक्त शर्करा का स्तर बढ़ जाता है।",
    )
    answer = get_answer(prompt)
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 0, "Answer should not be empty"
