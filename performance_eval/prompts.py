"""Condition-aware MCQ prompt builder (baseline / CoT / few-shot).

Instructions are in English; question content is in Hindi. When CoT is on, the
model must end with the literal marker ``Answer: X`` so the scorer has one fixed
string to search for.
"""

_BASELINE_INSTRUCTION = (
    "You are answering a multiple-choice medical question in Hindi. Read the "
    "question and the options carefully, then respond with ONLY the letter of "
    "the correct answer ({letters}). Do not explain. Do not write anything "
    "except the single letter."
)

_COT_INSTRUCTION = (
    "You are answering a multiple-choice medical question in Hindi. Read the "
    "question and the options carefully. Think step by step in Hindi. On the "
    "final line, write exactly `Answer: X` where X is the single letter of the "
    "correct answer ({letters}). Do not write anything after that line."
)

_PASSAGES_HEADER = (
    "Reference passages (in English). They may or may not be relevant to the "
    "question. Use them only if they help."
)
PASSAGE_MAX_CHARS = 1000
# --language value -> phrase replacing "in Hindi" in the question description (plan A3).
_LANGUAGE_PHRASES = {
    "hi": "in Hindi",
    "en": "in English",
    "hi_q_en_opts": "in Hindi (answer options in English)",
    "en_q_hi_opts": "in English (answer options in Hindi)",
}
LANGUAGES = tuple(_LANGUAGE_PHRASES)
_MIXED_LANGUAGES = ("hi_q_en_opts", "en_q_hi_opts")
# --reasoning-language value -> phrase used in the CoT reasoning instructions only.
_REASONING_PHRASES = {"hi": "in Hindi", "en": "in English"}
_REASONING_SLOT = "\x00REASONING\x00"


def _make_letter_list(n_options: int) -> str:
    """'A, B, or C' style list for the first n_options letters."""
    letters = [chr(ord("A") + i) for i in range(n_options)]
    if len(letters) == 1:
        return letters[0]
    return ", ".join(letters[:-1]) + f", or {letters[-1]}"


def build_prompt(
    question_with_options: str,
    n_options: int,
    use_cot: bool = False,
    exemplars: list[dict] | None = None,
    passages: list[dict] | None = None,
    language: str = "hi",
    reasoning_language: str | None = None,
) -> str:
    """Build the prompt for one item under one condition.

    exemplars: optional list of {"question_with_options", "answer_letter"}
    rendered before the target question, in the order given.
    passages: optional retrieved passages ({"title", "content"}); rendered as a
    reference block that always comes first. None leaves the prompt unchanged.
    language: "hi" (default; prompt text unchanged), "en" for translated items (every "in Hindi"
    becomes "in English"), or a mixed value ("hi_q_en_opts" / "en_q_hi_opts") that changes the
    question description only; mixed values are baseline-only (no CoT).
    reasoning_language: CoT only. None (default) = reason in the item language, prompt unchanged.
    "hi"/"en" changes only "Think step by step in ..." and the closing reasoning cue; the
    question description still follows language.
    """
    if language not in _LANGUAGE_PHRASES:
        raise ValueError(f"language must be one of {LANGUAGES}, got {language!r}")
    if use_cot and language in _MIXED_LANGUAGES:
        raise ValueError(f"language {language!r} is not supported with CoT")
    if reasoning_language is not None:
        if not use_cot:
            raise ValueError("reasoning_language requires use_cot")
        if reasoning_language not in _REASONING_PHRASES:
            raise ValueError(f"reasoning_language must be one of {sorted(_REASONING_PHRASES)}, "
                             f"got {reasoning_language!r}")
    parts = []
    if passages:
        blocks = [
            f"[{i}] {p['title']}\n{p['content'][:PASSAGE_MAX_CHARS]}"
            for i, p in enumerate(passages, 1)
        ]
        parts.append(_PASSAGES_HEADER + "\n\n" + "\n\n".join(blocks))
    if exemplars:
        blocks = [
            f"Question: {ex['question_with_options']}\nAnswer: {ex['answer_letter']}"
            for ex in exemplars
        ]
        parts.append("Here are example questions:\n\n" + "\n\n".join(blocks))

    lang = _LANGUAGE_PHRASES[language]
    reasoning = lang if reasoning_language is None else _REASONING_PHRASES[reasoning_language]
    # Reasoning phrases go through a slot so the question-description replacement can't touch them.
    template = (_COT_INSTRUCTION.replace("step by step in Hindi", f"step by step {_REASONING_SLOT}")
                if use_cot else _BASELINE_INSTRUCTION)
    instruction = template.format(letters=_make_letter_list(n_options)).replace("in Hindi", lang)
    parts.append(instruction.replace(_REASONING_SLOT, reasoning))

    # CoT: end on a reasoning cue (not "Answer"), so the model reasons first
    # and only then writes the final `Answer: X` line.
    closing = f"Reasoning (step by step, {reasoning}):" if use_cot else "Answer (single letter only):"
    parts.append(f"Question: {question_with_options}\n\n{closing}")
    return "\n\n".join(parts)
