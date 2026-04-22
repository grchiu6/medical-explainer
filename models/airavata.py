"""Airavata (ai4bharat) Hindi instruction-tuned model integration.

WHAT THIS FILE DOES:
Downloads the Airavata model from HuggingFace, loads it onto a GPU,
and generates answers to Hindi questions locally (no internet needed
after the first download).

HOW LOCAL MODEL INFERENCE WORKS (step by step):
1. Load the model weights from HuggingFace (downloaded once, cached)
2. Load the tokenizer (converts text to numbers and back)
3. Format our prompt in Airavata's expected chat format
4. Convert the prompt text into numbers (tokenization)
5. Feed the numbers into the model on the GPU
6. The model outputs new numbers (predictions)
7. Convert those numbers back into text (detokenization)
8. Return the text as the answer

WHAT IS A TOKENIZER?
Language models don't understand text directly — they work with numbers.
A tokenizer is like a translator between human text and numbers:
- "मधुमेह" (diabetes) might become [1234, 5678]
- Each number represents a piece of the word (called a "token")
- The model processes these numbers and outputs new numbers
- The tokenizer converts those output numbers back into text

WHAT IS torch.bfloat16?
Models have billions of numbers (parameters). By default, each number
uses 32 bits of memory (float32). bfloat16 uses only 16 bits — half
the memory. This lets us fit larger models onto the GPU with minimal
quality loss. It's standard practice for inference (generating answers).

Model: ai4bharat/Airavata (based on OpenHathi / Llama 2)
Prompt format: https://huggingface.co/ai4bharat/Airavata
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# The model's name on HuggingFace — like a URL for the model.
# HuggingFace is like GitHub but for ML models.
_MODEL_NAME = "ai4bharat/Airavata"

# These will hold the loaded model and tokenizer.
# None initially — loaded on first use (lazy initialization,
# same pattern we used in gemini.py).
_tokenizer = None
_model = None


def _load_model():
    """Download and load the Airavata model onto the GPU.

    This is slow (several minutes on first run) because it downloads
    ~14GB of model weights. After the first run, HuggingFace caches
    the files locally so subsequent loads are faster.

    The 'global' keyword lets us modify the module-level _tokenizer
    and _model variables from inside this function.

    When you'd use this pattern: Any time you have an expensive
    initialization step that should only happen once. The first call
    to get_answer() triggers the load; all subsequent calls reuse
    the already-loaded model.
    """
    global _tokenizer, _model
    if _model is not None:
        return  # Already loaded — skip

    print(f"Loading {_MODEL_NAME}... (this may take a few minutes on first run)")

    # AutoTokenizer.from_pretrained() downloads the tokenizer files
    # from HuggingFace. "pretrained" means someone already trained this
    # model and we're downloading their work.
    # padding_side="left" means: if we need to add padding (empty space)
    # to make inputs the same length, add it to the left side.
    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, padding_side="left")

    # Some tokenizers don't have a "pad token" (a special token used
    # for padding). Setting it to the "end of sequence" token is a
    # common workaround.
    _tokenizer.pad_token = _tokenizer.eos_token

    # AutoModelForCausalLM.from_pretrained() downloads the actual model
    # weights (~14GB) from HuggingFace.
    # - "CausalLM" means "causal language model" — a model that predicts
    #   the next word given all previous words (like autocomplete)
    # - torch_dtype=torch.bfloat16 — use half-precision to save GPU memory
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME, torch_dtype=torch.bfloat16
    )

    # .to(device) moves the model onto the GPU.
    # "cuda" = NVIDIA GPU, "cpu" = regular processor (much slower)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    print(f"Model loaded on {device}.")


def _format_for_airavata(prompt_text: str) -> str:
    """Wrap the prompt in Airavata's expected chat format.

    Every model has its own special format for prompts. Airavata
    expects this specific structure (from its model card):

        <|user|>
        {your message here}
        <|assistant|>
        {model generates response here}

    The <|user|> and <|assistant|> are special tokens that tell the
    model "this is what the human said" and "now you should respond."
    The newline after <|assistant|> is required — without it, the
    model's output quality drops significantly.

    Why each model has its own format: During instruction tuning,
    the model was trained with this specific format. Using a different
    format confuses the model because it's never seen that pattern
    during training.

    When you'd need to know this: Any time you use a HuggingFace model
    for the first time, check its model card (README) for the expected
    prompt format. Getting this wrong is one of the most common mistakes.

    Args:
        prompt_text: The pre-built prompt text (from evaluate.py).

    Returns:
        The prompt wrapped in Airavata's chat format.
    """
    return f"<|user|>\n{prompt_text}\n<|assistant|>\n"


def get_answer(prompt: str) -> str:
    """Generate an answer using Airavata.

    Same interface as gemini.py's get_answer() — takes a pre-built
    prompt, returns an answer string. The evaluation runner builds
    the right prompt for the task (extractive, MCQ, etc.) and passes
    it in. This file just handles inference — it doesn't care what
    kind of task the prompt represents.

    Args:
        prompt: The fully-formatted prompt string.

    Returns:
        The model's answer as a string.
    """
    # Step 1: Load model if not already loaded
    _load_model()

    # Step 2: Wrap the prompt in Airavata's expected chat format
    formatted = _format_for_airavata(prompt)

    # Step 3: Tokenize — convert text to numbers
    # .to(device) moves the numbers onto the GPU so the model can process them
    device = next(_model.parameters()).device  # find which device the model is on
    inputs = _tokenizer(formatted, return_tensors="pt").to(device)
    # "pt" means "return PyTorch tensors" (the number format PyTorch uses)
    # inputs is a dict with 'input_ids' (the token numbers) and
    # 'attention_mask' (which tokens are real vs padding)

    # Step 4: Generate — feed numbers into model, get new numbers out
    # torch.no_grad() tells PyTorch "we're not training, just generating"
    # This saves memory because PyTorch doesn't need to track gradients
    # (gradients are only needed during training, not inference)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,          # the tokenized prompt
            max_new_tokens=256,  # generate at most 256 new tokens
            do_sample=False,   # don't add randomness — always pick the
                               # most likely next token (deterministic)
        )
    # **inputs is Python "unpacking" — it passes each key-value pair
    # from the dict as a separate argument to .generate()

    # Step 5: Decode — convert output numbers back to text
    # We only want the NEW tokens the model generated, not the prompt
    # we fed in. So we slice off the input tokens.
    input_length = inputs["input_ids"].shape[1]  # how many tokens was our prompt?
    generated_tokens = outputs[0][input_length:]  # everything after the prompt
    answer = _tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # skip_special_tokens=True removes things like </s> (end of sequence)
    # from the output — we just want the readable text

    return answer.strip()
