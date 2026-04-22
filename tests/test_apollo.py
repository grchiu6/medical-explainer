"""Tests for Apollo multilingual medical model integration.

NOTE: Same as Airavata — requires a GPU, skips on laptops.
Apollo is a larger model (~9B parameters) so it needs more GPU
memory than Airavata. On the HPC cluster, request a GPU with
at least 24GB VRAM.
"""

import pytest
import torch

from models.apollo import get_answer
from models.base import build_prompt


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No GPU available — skipping local model test"
)
def test_get_answer_returns_nonempty_string():
    """Apollo should return a non-empty string for a Hindi medical question."""
    prompt = build_prompt(
        question="मधुमेह के लक्षण क्या हैं?",
        context="मधुमेह एक चयापचय रोग है जिसमें रक्त शर्करा का स्तर बढ़ जाता है।",
    )
    answer = get_answer(prompt)
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 0, "Answer should not be empty"
