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
) -> str:
    """Build the prompt for one item under one condition.

    exemplars: optional list of {"question_with_options", "answer_letter"}
    rendered before the target question, in the order given.
    """
    parts = []
    if exemplars:
        blocks = [
            f"Question: {ex['question_with_options']}\nAnswer: {ex['answer_letter']}"
            for ex in exemplars
        ]
        parts.append("Here are example questions:\n\n" + "\n\n".join(blocks))

    template = _COT_INSTRUCTION if use_cot else _BASELINE_INSTRUCTION
    parts.append(template.format(letters=_make_letter_list(n_options)))

    # CoT: end on a reasoning cue (not "Answer"), so the model reasons first
    # and only then writes the final `Answer: X` line.
    closing = (
        "Reasoning (step by step, in Hindi):" if use_cot else "Answer (single letter only):"
    )
    parts.append(f"Question: {question_with_options}\n\n{closing}")
    return "\n\n".join(parts)
