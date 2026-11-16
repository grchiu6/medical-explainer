"""Llama-3-Nanda-10B-Chat — Hindi-specialist bilingual model.

WHAT THIS MODEL IS:
Nanda is a 10B parameter bilingual (Hindi + English) model developed by
MBZUAI, Inception, and Cerebras. It was pre-trained on 65 billion Hindi
tokens — one of the largest Hindi training corpora for an open model —
and then instruction-tuned for chat.

WHY IT'S A STRONG CANDIDATE FOR THIS RESEARCH:
Your research question is about adapting Hindi medical explanations to
different literacy levels. Nanda is specifically optimized for Hindi
language quality — it should produce more natural, idiomatic Hindi than
Apollo or Airavata, which are either medically-focused or general.
This creates a useful research contrast:
  - Apollo: medical knowledge, moderate Hindi
  - Airavata: strong Hindi, no medical specialization
  - Nanda: strong Hindi (largest Hindi corpus), general knowledge

PROMPT FORMAT (from model card):
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    {user_message}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

This is the standard Llama 3 chat format. The model was evaluated on
MMLU, HellaSwag, ARC-Challenge, and multiple Hindi benchmarks, and
achieved state-of-the-art for open Hindi models at time of release.

IMPORTANT: Nanda requires trust_remote_code=True because it has a
custom model class. This is safe — it just means the model's authors
provided custom Python code for their architecture alongside the weights.

Model: MBZUAI/Llama-3-Nanda-10B-Chat
Paper: https://github.com/mbzuai-nlp/Llama-3-Nanda-10B-Chat/blob/main/Llama-3-Nanda-10B-Chat-Paper.pdf
License: Llama 3 (requires HuggingFace account + license acceptance)
GPU requirement: ~20GB VRAM in bfloat16
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_NAME = "MBZUAI/Llama-3-Nanda-10B-Chat"

_tokenizer = None
_model = None


def _load_model():
    """Download and load the Nanda model onto the GPU.

    NOTE: You must accept the model's license on HuggingFace before
    downloading. Go to https://huggingface.co/MBZUAI/Llama-3-Nanda-10B-Chat
    and click "Agree and access repository" while logged in. Then run
    `huggingface-cli login` on the cluster with your HF token.

    trust_remote_code=True is required — Nanda has a custom model class.
    """
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading {_MODEL_NAME}... (this may take a few minutes on first run)")
    print("NOTE: If this fails with 401/403, run: huggingface-cli login")

    _tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_NAME,
        trust_remote_code=True,
    )

    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",          # Automatically distribute across available GPUs
        trust_remote_code=True,
    )

    # hf_device_map is only present when accelerate dispatches across devices;
    # fall back to the first parameter's device for a simple status print
    device_info = getattr(_model, "hf_device_map", None) or next(_model.parameters()).device
    print(f"Model loaded. Device: {device_info}")


def _format_for_nanda(prompt_text: str) -> str:
    """Wrap the prompt in Llama 3 chat format as used by Nanda.

    Nanda uses the standard Llama 3 chat format with special header tokens.
    The system message tells the model what role it's playing.
    The user message contains our literacy-level prompt.

    From the Nanda model card:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful AI assistant.<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {Question}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>

    We keep the system message minimal and in English (consistent with
    Nanda's training) and put the full Hindi instruction in the user turn.

    Args:
        prompt_text: The pre-built prompt from literacy_eval/prompts.py.

    Returns:
        The prompt formatted in Llama 3 chat style.
    """
    system_message = "You are a helpful medical educator. Follow the instructions and respond only in Hindi."

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>"
        f"{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>"
        f"{prompt_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )


_DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 512,
    "do_sample": True,
    "top_p": 0.95,
    "temperature": 0.2,
    "repetition_penalty": 1.3,   # from Nanda model card — prevents loops
}


def _merge_gen_kwargs(overrides: dict | None) -> dict:
    """Merge override generation kwargs onto this model's defaults."""
    return {**_DEFAULT_GEN_KWARGS, **(overrides or {})}


def get_answer(prompt: str, gen_kwargs: dict | None = None) -> str:
    """Generate an answer using Nanda-10B.

    Same interface as airavata.py and apollo.py — takes the pre-built
    prompt from the literacy eval pipeline and returns a Hindi response.

    Generation parameters follow the Nanda model card recommendation:
    top_p=0.95, temperature=0.2, repetition_penalty=1.3, do_sample=True.

    Args:
        prompt: The fully-formatted prompt string from run_eval.py.
        gen_kwargs: Optional generation kwargs that override _DEFAULT_GEN_KWARGS
            for this call only (e.g. {"do_sample": False} for greedy decoding).
            None keeps the defaults.

    Returns:
        The model's Hindi answer as a string.
    """
    _load_model()

    formatted = _format_for_nanda(prompt)

    # find which device the model's first layer is on
    device = next(_model.parameters()).device
    inputs = _tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            **_merge_gen_kwargs(gen_kwargs),
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return answer.strip()
