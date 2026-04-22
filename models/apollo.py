"""Apollo-7B multilingual medical model integration.

WHAT THIS FILE DOES:
Same thing as airavata.py — downloads a model from HuggingFace, loads it
onto a GPU, and generates answers locally. The key difference is the MODEL
itself and its PROMPT FORMAT.

WHAT MAKES APOLLO DIFFERENT FROM AIRAVATA:
- Airavata: trained on general Hindi text (knows Hindi well, but not
  specifically medicine)
- Apollo: trained on medical text in 6 languages including Hindi
  (knows medicine AND Hindi, but may be weaker at Hindi than Airavata
  since it splits its attention across 6 languages)

This is exactly the kind of tradeoff your research explores: does a
medical-specialist model or a Hindi-specialist model do better on
Hindi medical questions?

APOLLO'S PROMPT FORMAT (from its HuggingFace model card):
    User:{query}
    Assistant:{response}<|endoftext|>

This is simpler than Airavata's format. Each model learns its own
format during instruction tuning — you MUST match it exactly or the
model gets confused and produces garbage output.

Model: FreedomIntelligence/Apollo-7B (based on Gemma)
Source: https://huggingface.co/FreedomIntelligence/Apollo-7B
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# The model's identifier on HuggingFace.
_MODEL_NAME = "FreedomIntelligence/Apollo-7B"

# Same lazy initialization pattern as airavata.py.
_tokenizer = None
_model = None


def _load_model():
    """Download and load the Apollo model onto the GPU.

    Same pattern as airavata.py — download once, cache, reuse.
    Apollo is ~9B parameters, so it needs ~18GB GPU memory in bfloat16.
    Make sure to request a GPU with at least 24GB VRAM on the cluster.
    """
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading {_MODEL_NAME}... (this may take a few minutes on first run)")

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME, torch_dtype=torch.bfloat16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    print(f"Model loaded on {device}.")


def _format_for_apollo(prompt_text: str) -> str:
    """Wrap the prompt in Apollo's expected chat format.

    Apollo's format (from its model card):
        User:{query}
        Assistant:

    Much simpler than Airavata's format. No special tokens like
    <|user|> — just plain text prefixes.

    Why we don't add <|endoftext|> at the end: that token marks
    the END of the model's response. We only provide the beginning
    ("Assistant:") and let the model generate the rest. The model
    will add <|endoftext|> itself when it's done.

    Args:
        prompt_text: The pre-built prompt text (from evaluate.py).

    Returns:
        The prompt wrapped in Apollo's chat format.
    """
    return f"User:{prompt_text}\nAssistant:"


def get_answer(prompt: str) -> str:
    """Generate an answer using Apollo-7B.

    Same interface as gemini.py and airavata.py — takes a pre-built
    prompt, returns an answer string. The caller (evaluate.py) is
    responsible for choosing the right prompt template for the task.

    Args:
        prompt: The fully-formatted prompt string.

    Returns:
        The model's answer as a string.
    """
    # Step 1: Load model if not already loaded
    _load_model()

    # Step 2: Wrap the prompt in Apollo's chat format
    formatted = _format_for_apollo(prompt)

    # Step 3: Tokenize (text → numbers)
    device = next(_model.parameters()).device
    inputs = _tokenizer(formatted, return_tensors="pt").to(device)

    # Step 4: Generate (numbers → new numbers)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    # Step 5: Decode only the new tokens (skip the prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return answer.strip()
