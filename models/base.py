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

# MCQ prompt: instructions in English (model follows English instructions
# better), question in Hindi (the actual content). The strict
# "ONLY the letter" instruction makes the output trivial to parse.
# Without it, models tend to output explanations like
# "उत्तर A है क्योंकि..." which is harder to score reliably.
#
# {letters} is filled in dynamically by build_mcq_prompt() based on
# how many options the specific question has (A-C up to A-J).
# This prevents misleading a 10-option question by saying "pick A-E".
_MCQ_PROMPT = """You are answering a multiple-choice medical question in Hindi. Read the question and the options carefully, then respond with ONLY the letter of the correct answer ({letters}). Do not explain. Do not write anything except the single letter.

Question: {question}

Answer (single letter only):"""

import re as _re

# Matches option labels at the start of a line: "A. ", "B. ", ... "J. "
_OPTION_LABEL_RE = _re.compile(r'^([A-J])\. ', _re.MULTILINE)


def _last_option_letter(question: str) -> str:
    """Return the last option letter present in a formatted question string.

    Scans for patterns like "A. ", "B. ", ... "J. " at the start of any
    line. The alphabetically largest letter found is the last option.

    Used to build a prompt that accurately lists the valid choices:
    a 3-option question gets "A, B, or C"; a 10-option question gets
    "A, B, C, D, E, F, G, H, I, or J".

    Returns:
        A single uppercase letter 'A'-'J'. Defaults to 'E' if no option
        labels are found (should not happen with well-formed data).
    """
    matches = _OPTION_LABEL_RE.findall(question)
    return max(matches) if matches else "E"


def _make_letter_list(last_letter: str) -> str:
    """Build a human-readable letter list string for the MCQ prompt.

    Examples:
        'D' → 'A, B, C, or D'
        'E' → 'A, B, C, D, or E'
        'J' → 'A, B, C, D, E, F, G, H, I, or J'
    """
    letters = [chr(ord('A') + i) for i in range(ord(last_letter) - ord('A') + 1)]
    if len(letters) == 1:
        return letters[0]
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


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


def build_mcq_prompt(question: str) -> str:
    """Build the prompt for a multiple-choice question (HiMed format).

    The question string is expected to already include the options inline,
    e.g. "Question?\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\nE. ...".
    The HiMed loader produces questions in this exact form.

    The valid answer letters in the prompt are derived automatically
    from the options present in the question string. So a 10-option
    question will say "A, B, C, D, E, F, G, H, I, or J" rather than
    the misleading "A, B, C, D, or E".

    Args:
        question: Hindi question text with embedded A-J options.

    Returns:
        A formatted prompt string ready to send to any model.
    """
    last = _last_option_letter(question)
    letters = _make_letter_list(last)
    return _MCQ_PROMPT.format(question=question, letters=letters)
