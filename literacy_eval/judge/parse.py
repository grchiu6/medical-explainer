"""Read feedback and score out of an M-Prometheus reply: "Feedback: ... [RESULT] 4".

The model card writes the tag as [RESULT], the paper as [[RESULT]]; both are accepted. If the reply
mentions the tag more than once (e.g. quoting the format), the LAST tagged score is the verdict.
"""

from __future__ import annotations

import re

_RESULT = re.compile(r"\[\[?RESULT\]\]?\s*:?\s*(\d+)")
_PREFIX = re.compile(r"^\s*Feedback\s*:\s*", re.I)


class ParseError(ValueError):
    """The reply has no tagged score, or the score is outside 1-5."""


def parse_result(text: str) -> tuple[str, int]:
    matches = list(_RESULT.finditer(text or ""))
    if not matches:
        raise ParseError(f"no [RESULT] score in reply: {text[-200:]!r}")
    last = matches[-1]
    score = int(last.group(1))
    if not 1 <= score <= 5:
        raise ParseError(f"score {score} outside 1-5")
    feedback = _PREFIX.sub("", text[:last.start()]).strip()
    return feedback, score
