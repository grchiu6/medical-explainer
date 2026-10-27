"""M-Prometheus direct-assessment prompts.

Template text is the Prometheus-2 absolute-grading prompt (prometheus-eval `ABSOLUTE_PROMPT` and
`ABSOLUTE_PROMPT_WO_REF`), with the output-format line as shown on the M-Prometheus model card
("Feedback: (write a feedback for criteria) [RESULT] ..."). Checked against
huggingface.co/Unbabel/M-Prometheus-14B and github.com/prometheus-eval/prometheus-eval on 2026-10-27.

WHY str.replace AND NOT str.format: answers and CoTs can contain "{", "}" or "###". Placeholders are
filled in one pass over the template, so text that was filled in is never scanned again.
"""

from __future__ import annotations

import re

from literacy_eval.judge.rubrics import Rubric

_STEPS = """1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations."""

ABSOLUTE_PROMPT = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
{_STEPS}

###The instruction to evaluate:
<<INSTRUCTION>>

###Response to evaluate:
<<RESPONSE>>

###Reference Answer (Score 5):
<<REFERENCE>>

###Score Rubrics:
<<RUBRIC>>

###Feedback: """

ABSOLUTE_PROMPT_WO_REF = f"""###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
{_STEPS}

###The instruction to evaluate:
<<INSTRUCTION>>

###Response to evaluate:
<<RESPONSE>>

###Score Rubrics:
<<RUBRIC>>

###Feedback: """

_PLACEHOLDER = re.compile(r"<<(INSTRUCTION|RESPONSE|REFERENCE|RUBRIC)>>")


def format_rubric(r: Rubric) -> str:
    lines = [f"[{r.criteria}]"] + [f"Score {n}: {desc}" for n, desc in enumerate(r.scores, 1)]
    return "\n".join(lines)


def build_prompt(instruction: str, response: str, rubric: Rubric, reference: str | None = None) -> str:
    """Fill the with-reference template if `reference` is given, else the no-reference one."""
    values = {"INSTRUCTION": instruction, "RESPONSE": response, "RUBRIC": format_rubric(rubric),
              "REFERENCE": reference or ""}
    template = ABSOLUTE_PROMPT if reference is not None else ABSOLUTE_PROMPT_WO_REF
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
