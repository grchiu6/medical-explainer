"""Tests for Airavata model integration.

NOTE: This test requires a GPU to run. It will be automatically
SKIPPED on machines without a GPU (like your laptop).
It will only PASS when run on the HPC cluster.

Why skip instead of fail? Because we don't want a "FAILED" test
report just because someone ran tests on their laptop. The test
is still valuable — it runs automatically on the HPC cluster
to verify the model integration works.
"""

import pytest
import torch

from models.airavata import get_answer
from models.base import build_prompt


# Check if a GPU (CUDA device) is available.
# torch.cuda.is_available() returns True if the machine has an
# NVIDIA GPU with CUDA drivers installed.
# Your laptop: False (no NVIDIA GPU)
# HPC cluster: True (has NVIDIA GPUs)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No GPU available — skipping local model test"
)
def test_get_answer_returns_nonempty_string():
    """Airavata should return a non-empty string for a Hindi question.

    Same smoke test pattern as Gemini — we just check that the model
    loads and produces output. Correctness is measured later with
    BLEU/ROUGE across the full dataset.
    """
    prompt = build_prompt(
        question="मधुमेह के लक्षण क्या हैं?",
        context="मधुमेह एक चयापचय रोग है जिसमें रक्त शर्करा का स्तर बढ़ जाता है।",
    )
    answer = get_answer(prompt)
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 0, "Answer should not be empty"
