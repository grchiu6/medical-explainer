"""Gemini 3.8 Flash for MCQ evaluation (paid tier), with token/cost tracking and a budget guard.

Same interface as the local model wrappers: get_answer(prompt, gen_kwargs=None) -> str.
HF-only gen_kwargs (do_sample, max_new_tokens, ...) are ignored; the only key read is
"thinking_level". Decoding is temperature 0 with the lowest thinking level, so the
baseline is comparable to greedy single-letter answers from the open models.

literacy_eval keeps using models/gemini.py (2.5 Flash); this file is separate on purpose.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

MODEL_ID = "gemini-3.8-flash"
# USD per 1M tokens, paid tier, valid through 2026-12-31 (prices double on 2027-01-01).
PRICE_IN_PER_M = 0.75
PRICE_OUT_PER_M = 3.75  # output price includes thinking tokens
DEFAULT_THINKING_LEVEL = "MINIMAL"
DEFAULT_BUDGET_USD = 2.0
_RETRY_WAITS = (2, 4, 8)

_client = None
# Running total of ALL Gemini spend (every run and the translation); the budget applies to this.
TOTAL_PATH = Path(__file__).resolve().parents[1] / "performance_eval" / "outputs" / "gemini_total_usage.json"


class BudgetExceeded(RuntimeError):
    pass


def _empty_usage() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
            "cost_usd": 0.0, "blocked": 0}


USAGE = _empty_usage()


def load_usage(path: Path) -> None:
    """Restore counters from a previous (interrupted) run so cost accumulates across resumes."""
    path = Path(path)
    if path.is_file():
        USAGE.update(json.loads(path.read_text(encoding="utf-8")))


def save_usage(path: Path) -> None:
    Path(path).write_text(json.dumps(USAGE, indent=2), encoding="utf-8")


def _get_client():
    global _client
    if _client is None:
        from google import genai

        load_dotenv()
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=key)
    return _client


def total_cost() -> float:
    if not TOTAL_PATH.is_file():
        return 0.0
    return float(json.loads(TOTAL_PATH.read_text(encoding="utf-8"))["cost_usd"])


def _add_to_total(cost: float) -> None:
    TOTAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOTAL_PATH.write_text(json.dumps({"cost_usd": total_cost() + cost}, indent=2), encoding="utf-8")


def _budget() -> float:
    return float(os.getenv("GEMINI_BUDGET_USD", DEFAULT_BUDGET_USD))


def _record(resp) -> None:
    u = getattr(resp, "usage_metadata", None)
    p = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
    o = (getattr(u, "candidates_token_count", 0) or 0) if u else 0
    t = (getattr(u, "thoughts_token_count", 0) or 0) if u else 0
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += p
    USAGE["output_tokens"] += o
    USAGE["thoughts_tokens"] += t
    cost = (p * PRICE_IN_PER_M + (o + t) * PRICE_OUT_PER_M) / 1e6
    USAGE["cost_usd"] += cost
    _add_to_total(cost)


def get_answer(prompt: str, gen_kwargs: dict | None = None) -> str:
    from google.genai import types

    spent = max(total_cost(), USAGE["cost_usd"])
    if spent >= _budget():
        raise BudgetExceeded(f"total Gemini spend ${spent:.4f} >= budget ${_budget():.2f} "
                             f"(GEMINI_BUDGET_USD; running total in {TOTAL_PATH})")
    level = (gen_kwargs or {}).get("thinking_level", DEFAULT_THINKING_LEVEL)
    config = types.GenerateContentConfig(
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level=level),
    )
    client = _get_client()
    for attempt, wait in enumerate((*_RETRY_WAITS, None)):
        try:
            resp = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
            break
        except Exception as e:  # network / 5xx / rate limit
            code = getattr(e, "code", None)
            permanent = isinstance(code, int) and 400 <= code < 500 and code != 429
            if wait is None or permanent:  # e.g. 402 no credit, 400 bad request: retrying cannot help
                raise
            print(f"Gemini error ({e}); retry {attempt + 1} in {wait}s")
            time.sleep(wait)
    _record(resp)
    text = getattr(resp, "text", None)
    if not text:
        USAGE["blocked"] += 1
        reason = resp.candidates[0].finish_reason if getattr(resp, "candidates", None) else "no candidates"
        print(f"Gemini returned no text (finish_reason={reason}); recorded as empty answer")
        return ""
    return text.strip()
