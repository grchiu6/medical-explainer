"""Shared prompt template for all models.

WHY THIS FILE EXISTS:
When comparing multiple models, every model must receive the exact same
prompt. If Model A gets a better-worded prompt than Model B, and Model A
scores higher, you can't tell if it's because the model is better or
because the prompt was better. This is called "controlling variables."

This file defines ONE prompt template that all models use. The only thing
that differs between models is HOW the prompt is delivered (API call vs.
local inference), not WHAT the prompt says.

HOW IT WORKS:
- build_prompt(question, context) returns a formatted string
- Each model file (gemini.py, airavata.py, apollo.py) calls this function
- The model then wraps the result in its own format (e.g., Airavata adds
  <|user|> tags) and sends it for inference
"""

# Two versions of the prompt: one for when we have context, one without.
# We use triple-quoted strings (\"\"\"...\"\"\") for multi-line text.
# The {question} and {context} are placeholders that get filled in
# by Python's .format() method when we call build_prompt().

_PROMPT_WITH_CONTEXT = """You are a question-answering assistant. Answer the following question in Hindi based on the provided context. Give a concise, accurate answer.

Context: {context}

Question: {question}

Answer:"""

_PROMPT_WITHOUT_CONTEXT = """You are a question-answering assistant. Answer the following question in Hindi. Give a concise, accurate answer.

Question: {question}

Answer:"""


def build_prompt(question: str, context: str) -> str:
    """Build the prompt string for a given question and optional context.

    This is the ONLY function other files should call from this module.

    Args:
        question: The question to answer (in Hindi).
        context: The passage providing context. Empty string if none.

    Returns:
        A formatted prompt string ready to send to any model.

    Example:
        >>> prompt = build_prompt("मधुमेह क्या है?", "मधुमेह एक रोग है।")
        >>> print(prompt)
        You are a question-answering assistant...
        Context: मधुमेह एक रोग है।
        Question: मधुमेह क्या है?
        Answer:
    """
    if context.strip():
        return _PROMPT_WITH_CONTEXT.format(question=question, context=context)
    return _PROMPT_WITHOUT_CONTEXT.format(question=question)
