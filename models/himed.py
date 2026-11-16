"""HiMed-8B — Hindi medical instruction-tuned model.

WHAT THIS MODEL IS:
HiMed-8B is a large language model developed by FreedomIntelligence
(same group as Apollo) specifically to improve Hindi medical reasoning.
It was built by fine-tuning Llama-3.1-8B-Instruct using a three-stage
training framework:
  1. Language Adaptation — strengthen Hindi capability
  2. Reasoning Cold-Start — teach medical reasoning structure
  3. DSR-RL (Decaying Scaffolding Reward Reinforcement Learning) —
     shifts the model from guided reasoning to task-optimal outputs

WHY IT'S THE STRONGEST CANDIDATE FOR THIS RESEARCH:
Your research question is about adapting Hindi medical explanations to
different health-literacy levels. HiMed-8B combines exactly the two
properties that matter:
  - HINDI: explicitly trained for Hindi language quality
  - MEDICAL: trained on the HiMed dataset (Western + Ayurvedic medicine)
  - INSTRUCTION-TUNED: follows chat instructions (unlike MMed-Llama base)

Compare this to your other models:
  - Apollo: medical + multilingual, but not Hindi-specialized
  - Airavata: Hindi-specialized, but not medical
  - Nanda: strong Hindi, but no medical specialization
  - HiMed-8B: BOTH Hindi AND medical — this is the most targeted model

PROMPT FORMAT:
HiMed-8B inherits the Llama 3.1 chat template from its base model.
The correct way to format prompts is using `apply_chat_template()` from
the tokenizer — this automatically handles the special tokens
(<|start_header_id|>, <|eot_id|>, etc.) correctly.

DO NOT manually construct the token strings — let the tokenizer do it.

Model: FreedomIntelligence/HiMed-8B
Base: Llama-3.1-8B-Instruct
License: Llama 3.1 (open access, no license gate needed)
GPU requirement: ~16GB VRAM in bfloat16
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_NAME = "FreedomIntelligence/HiMed-8B"

_tokenizer = None
_model = None


def _load_model():
    """Download and load HiMed-8B onto the GPU.

    HiMed-8B uses standard Llama 3.1 architecture — no trust_remote_code
    needed. No HuggingFace license gate either (unlike Nanda).
    """
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading {_MODEL_NAME}... (this may take a few minutes on first run)")

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("HiMed-8B loaded.")


def _format_for_himed(prompt_text: str) -> str:
    """Format the prompt using Llama 3.1 chat template via apply_chat_template.

    HiMed-8B inherits Llama-3.1-8B-Instruct's chat template. Using
    apply_chat_template() is the recommended approach — it handles the
    exact special token placement automatically, which is important
    because getting the token order slightly wrong leads to degraded output.

    We use a minimal system message in Hindi to be consistent with our
    research setup (Hindi-instruction baseline).

    Args:
        prompt_text: The pre-built Hindi instruction prompt from prompts.py.

    Returns:
        The formatted prompt string ready for tokenization.
    """
    messages = [
        {
            "role": "system",
            "content": "आप एक सहायक चिकित्सा शिक्षक हैं। दिए गए निर्देशों का पालन करते हुए हिंदी में उत्तर दें।"
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]

    # apply_chat_template returns a string with the correct token structure
    # add_generation_prompt=True appends the assistant header so the model
    # starts generating a response immediately rather than reflecting on the prompt
    return _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


_DEFAULT_GEN_KWARGS = {
    # 1024 since 2026-10-01: at 512, 19/60 high-level answers ran out of
    # tokens inside "## Thinking". Low/mid were generated at 512 (0 and 1
    # of 60 truncated), high was regenerated at 1024.
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.3,    # low temperature = focused, minimal hallucination
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}


def _merge_gen_kwargs(overrides: dict | None) -> dict:
    """Merge override generation kwargs onto this model's defaults."""
    return {**_DEFAULT_GEN_KWARGS, **(overrides or {})}


def get_answer(prompt: str, gen_kwargs: dict | None = None) -> str:
    """Generate a Hindi medical explanation using HiMed-8B.

    Same interface as all other model wrappers — takes the pre-built prompt
    from run_eval.py, returns the model's Hindi response as a string.

    Args:
        prompt: The fully-formatted Hindi instruction prompt from run_eval.py.
        gen_kwargs: Optional generation kwargs that override _DEFAULT_GEN_KWARGS
            for this call only (e.g. {"do_sample": False} for greedy decoding).
            None keeps the defaults.

    Returns:
        The model's Hindi response as a string.
    """
    _load_model()

    formatted = _format_for_himed(prompt)

    # Tokenize the formatted prompt
    device = next(_model.parameters()).device
    inputs = _tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            **_merge_gen_kwargs(gen_kwargs),
        )

    # Decode only the newly generated tokens (skip the prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return answer.strip()
