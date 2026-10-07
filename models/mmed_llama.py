"""MMed-Llama-3-8B — multilingual medical foundation model.

WHAT THIS MODEL IS:
MMed-Llama-3-8B is a medical-domain LLM from FreedomIntelligence (the
same group that made Apollo). It was created by continuing pre-training
Llama-3-8B on MMedC — a multilingual medical corpus covering English,
Chinese, Japanese, French, Russian, and Hindi.

CRITICAL DIFFERENCE FROM OTHER MODELS:
MMed-Llama-3-8B is a FOUNDATION MODEL, not an instruction-tuned model.
This means:
  - It was NOT fine-tuned to follow chat instructions
  - It is a text *completer*, not a conversational assistant
  - It continues text rather than answering questions
  - You cannot use the same instruction-style prompt as Nanda/Airavata/Apollo

WHY INCLUDE IT ANYWAY?
Including a base medical model is scientifically valuable because it
isolates the effect of instruction tuning from medical pre-training:
  - Apollo: medical pre-training + instruction tuning
  - MMed-Llama: medical pre-training, NO instruction tuning
  - Airavata: Hindi pre-training + instruction tuning
Comparing their outputs shows whether instruction tuning matters more
than domain-specific pre-training for register adaptation.

HOW WE ADAPT THE PROMPT FOR A BASE MODEL:
Instead of instruction-style prompts ("explain this to a 5th grader"),
we use few-shot prompting — show the model 1 example of the pattern
we want, then let it continue. This is the standard way to elicit
structured behavior from base models.

Prompt structure:
    [Example topic at target level, then:]
    विषय: {topic}
    व्याख्या:
    [model continues from here]

LANGUAGES IN MMedC (the training corpus):
English, Chinese, Japanese, French, Russian, Hindi
Hindi is included but is one of 6 languages — so Hindi quality may be
lower than Nanda (Hindi-specialist) or Airavata (Hindi-specialist).

Model: Henrychur/MMed-Llama-3-8B
Paper: https://arxiv.org/abs/2402.13963
License: Llama 3
GPU requirement: ~16GB VRAM in bfloat16
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_MODEL_NAME = "Henrychur/MMed-Llama-3-8B"

_tokenizer = None
_model = None


def _load_model():
    """Download and load MMed-Llama-3-8B onto the GPU.

    Standard base model loading — no trust_remote_code needed since
    MMed-Llama uses the standard Llama 3 architecture.
    """
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading {_MODEL_NAME}... (this may take a few minutes on first run)")

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)

    # Set pad token — base Llama models often lack one by default
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("MMed-Llama loaded.")


def _build_few_shot_prompt(prompt_text: str, literacy_level: str) -> str:
    """Build a few-shot prompt for a base (non-instruction-tuned) model.

    Base models complete text, they don't follow instructions. The
    trick is to show the model a brief example of the pattern, then
    present the real topic and let the model continue.

    We give it a 1-shot example using a different topic (fever/बुखार)
    so it understands the expected format without confusing it with
    the actual topic we want explained.

    The few-shot example is kept simple and level-appropriate:
    - low:  plain, simple Hindi continuation
    - mid:  moderate Hindi with parenthetical definitions
    - high: technical Hindi

    Args:
        prompt_text: The pre-built instruction prompt (we extract the
                     topic from it and use the level to pick the example).
        literacy_level: 'low', 'mid', or 'high'.

    Returns:
        A few-shot formatted prompt for the base model.
    """
    # 1-shot examples — one per literacy level, using बुखार (fever) as demo
    EXAMPLES = {
        "low": (
            "विषय: बुखार क्या है और यह क्यों होता है?\n"
            "व्याख्या: बुखार तब होता है जब शरीर में गर्मी बढ़ जाती है। "
            "यह तब होता है जब शरीर किसी बीमारी से लड़ता है। "
            "बुखार में बहुत पानी पिएं और आराम करें।\n\n"
        ),
        "mid": (
            "विषय: बुखार क्या है और यह क्यों होता है?\n"
            "व्याख्या: बुखार (Fever) शरीर के तापमान का 98.6°F से अधिक हो जाना है। "
            "यह तब होता है जब प्रतिरक्षा प्रणाली (immune system) संक्रमण से लड़ती है। "
            "बुखार के लक्षण: कंपकंपी, सिरदर्द और थकान हो सकते हैं। "
            "उपचार में पर्याप्त पानी, आराम और ज्वरनाशक दवाएं शामिल हैं।\n\n"
        ),
        "high": (
            "विषय: बुखार क्या है और यह क्यों होता है?\n"
            "व्याख्या: बुखार (Pyrexia) एक थर्मोरेगुलेटरी प्रतिक्रिया है जिसमें "
            "हाइपोथैलेमस का तापमान सेट पॉइंट (set point) ऊपर उठ जाता है, "
            "जो साइटोकिन्स (IL-1, IL-6, TNF-α) द्वारा प्रोस्टाग्लैंडिन E₂ के माध्यम से "
            "प्रेरित होता है। नैदानिक मानदंड: तापमान >38°C (100.4°F)।\n\n"
        ),
    }

    example = EXAMPLES.get(literacy_level, EXAMPLES["mid"])

    # Extract the Hindi topic from the instruction prompt
    # The prompt ends with "\nहिंदी में व्याख्या:" — the topic is on the line before
    topic = "चिकित्सा विषय"
    for line in prompt_text.split("\n"):
        if line.startswith("चिकित्सा विषय:"):
            topic = line.replace("चिकित्सा विषय:", "").strip()
            break

    return (
        f"{example}"
        f"विषय: {topic}\n"
        f"व्याख्या:"
    )


def get_answer(prompt: str, literacy_level: str = "mid") -> str:
    """Generate an answer using MMed-Llama-3-8B (base model).

    Unlike instruction-tuned models, this uses few-shot prompting to
    elicit structured output. The literacy_level controls which few-shot
    example is prepended to guide the model's output register.

    NOTE ON INTERFACE: The run_eval.py pipeline calls get_answer(prompt)
    with a single argument. For MMed we need the literacy level to pick
    the right few-shot example. The literacy level is embedded in the
    prompt itself (as part of the instruction text), so we parse it out.
    Alternatively, you can pass it explicitly in extended usage.

    Args:
        prompt: The fully-formatted prompt from run_eval.py.
        literacy_level: Used to select the few-shot example register.
                        Parsed from the prompt if not passed explicitly.

    Returns:
        The model's Hindi text completion as a string.
    """
    _load_model()

    # Try to detect literacy level from the prompt if not passed explicitly
    detected_level = "mid"  # default
    if "पांचवीं कक्षा" in prompt:
        detected_level = "low"
    elif "दसवीं कक्षा" in prompt:
        detected_level = "mid"
    elif "चिकित्सा छात्र" in prompt or "विशेषज्ञ" in prompt:
        detected_level = "high"

    # Use passed level if provided, otherwise use detected
    level = literacy_level if literacy_level != "mid" else detected_level

    formatted = _build_few_shot_prompt(prompt, level)

    device = next(_model.parameters()).device
    inputs = _tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            # Stop at double-newline — our few-shot examples end with \n\n
            # so the model should stop generating after completing one response
            eos_token_id=_tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    answer = _tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Trim at first double-newline to avoid the model starting a second topic
    if "\n\n" in answer:
        answer = answer.split("\n\n")[0]

    return answer.strip()
