# MCQ Performance Techniques Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `performance_eval/` package that measures Hindi-medical MCQ accuracy for HiMed-8B and Nanda on MILU's Hindi "Health and Medicine" set, and tests whether each of CoT, few-shot, and decoding tuning raises accuracy — one technique at a time, with clean statistical attribution.

**Architecture:** A data loader pulls and freezes a disjoint test/few-shot-pool split from MILU. A condition-aware prompt builder composes baseline/CoT/few-shot prompts from one function. A run harness (mirroring `evaluate.py`'s resume-by-id pattern) generates answers per (model, condition) into a CSV. A scorer extracts the answer letter in one of two modes (first-letter for plain MCQ, after-marker/last-letter for CoT) and writes per-item correctness. An analysis module pairs conditions by item id and runs McNemar (reusing `literacy_eval.analysis.stats`) to decide whether each rung is kept.

**Tech Stack:** Python, `transformers`/`torch` (existing model wrappers), `datasets` (HF Hub, already a dependency), `scipy` (already a dependency, via `literacy_eval.analysis.stats`), `pytest`.

**Spec:** `performance_eval/docs/2026-11-15-mcq-performance-techniques-design.md`

## Global Constraints

- Deterministic (greedy, `do_sample=False`) decoding for C0–C3; sampling re-enters only at C4 as a tuned variable (spec resolution 1).
- CoT conditions must be given enough `max_new_tokens` to reach the answer marker; this is a correctness floor applied at C2, not a C4 "decoding" knob (spec resolution 2).
- Evaluate items in MILU's native option order; add a shuffled-subsample robustness check rather than reshuffling everything (spec resolution 3).
- Accuracy = `correct / total`, unparseable (`NONE`) counted as wrong; report the NONE rate separately. A NONE→correct-letter transition between conditions counts as a flip in the paired comparison (spec resolution 4).
- A rung is kept only if it shows a positive accuracy delta **and** two-sided McNemar p < 0.05 after Holm correction within each model's ladder (spec resolution 5).
- No LLM judge anywhere in this track — scoring is deterministic string/letter matching only.
- Never retrieve from or prompt with MILU's answer key/explanation (no such field is expected here, but this stays true of every future input to the prompt builder).
- `performance_eval/` is a new sibling package to `literacy_eval/`; nothing under `literacy_eval/` or any model wrapper's *default* behavior changes.
- PACE cluster compute; outputs/caches under `$SCRATCH` (`/home/hice1/gchiu6/scratch`), matching `scripts/literacy_himed_trad_run.sh` conventions (`$HF_HOME`, `$TRITON_CACHE_DIR`, `$PIP_CACHE_DIR` under scratch).

## Review Focus

- **Variable option count.** MILU items may not all have exactly 4 options; the loader and letter-list derivation must handle whatever count is actually present per item, not assume 4. (Task 1, Task 3)
- **CoT that never emits the marker.** A model can ramble past its token budget without ever writing `Answer: X`; the scorer must fall back to the last standalone letter, and if no letter exists anywhere, return `None` (counted as wrong, visible in the NONE rate) rather than raising or mis-parsing. (Task 4)
- **Few-shot/test leakage.** The split that freezes `test` vs `fewshot_pool` must be provably disjoint (by item id) every time it's called with the same seed. (Task 1)
- **Duplicate item ids.** If MILU has (or the loader somehow produces) two items sharing an id, resume logic and McNemar item-pairing both silently misbehave; the loader must detect and fail loudly on duplicates. (Task 1)
- **Resume after partial crash.** Re-running `run.py` on a condition CSV that already has some item ids must skip exactly those ids and not re-query the model for them. (Task 5)

---

## Task 1: MILU data loader + frozen test/few-shot splits

**Files:**
- Create: `performance_eval/__init__.py` (empty)
- Create: `performance_eval/data_milu.py`
- Modify: `.env.example` (add `HF_TOKEN`)
- Test: `tests/test_performance_data_milu.py`

**Interfaces:**
- Consumes: `datasets.load_dataset` (HF Hub, already in `requirements.txt`).
- Produces (used by Tasks 3, 5, 6):
  - `load_milu_hindi_medicine(hf_token: str | None = None) -> list[dict]` — each dict: `{"id": str, "question": str, "options": dict[str, str], "question_with_options": str, "answer_letter": str, "n_options": int}`. `question_with_options` is `question` followed by `"A. ...\nB. ..."` lines (same visual convention as `data/himed_loader.py:_format_question`), so `performance_eval/prompts.py` can reuse `models.base.build_mcq_prompt`-style formatting unchanged.
  - `split_test_fewshot(items: list[dict], fewshot_n: int = 10, seed: int = 42) -> tuple[list[dict], list[dict]]` — returns `(test_items, fewshot_pool_items)`, disjoint by `id`, deterministic (same pattern as `data/himed_loader.py`'s `random.seed(42)` + `random.sample`).

- [ ] **Step 1: Inspect the real MILU Hindi "Health and Medicine" schema on the cluster and record it**

This is a genuine unknown — MILU is gated on HuggingFace and its exact field names (e.g. whether the answer is stored as a letter, a 0-based index, or raw text; whether options are a list or named fields) were not available to inspect while writing this plan. Before writing the fixture or the loader body:

1. Accept MILU's HF gate and set `HF_TOKEN` in `.env` (`huggingface-cli login` or pass `token=` directly).
2. Run interactively on the cluster:
   ```python
   from datasets import load_dataset
   import os
   ds = load_dataset("ai4bharat/MILU", "hi", token=os.environ.get("HF_TOKEN"))
   rows = [r for r in ds["test"] if r.get("subject") == "Health and Medicine"]
   print(len(rows))
   print(rows[0])
   ```
3. Paste the observed keys, value types, and one full example as a comment block at the top of `tests/test_performance_data_milu.py`, and build the test fixture (Step 2) from that real example — not a guess.

**Step 1 is done when** the real schema is recorded in the test file's header comment, and the count of Hindi "Health and Medicine" rows is confirmed close to 499 (per the design doc's verified Table 9 count; a large mismatch means the subject filter or split name is wrong and must be re-checked before continuing).

- [ ] **Step 2: Write the failing tests for `load_milu_hindi_medicine`**

Using the real schema from Step 1, build a small fixture (a handful of rows, as a Python list of dicts shaped exactly like the real API response, monkeypatching `datasets.load_dataset` to return it — do not hit the network in tests).

```python
def test_load_milu_hindi_medicine_filters_subject_and_language(monkeypatch):
    # fixture includes rows with subject != "Health and Medicine" and
    # language != "hi"; assert only the matching ones come back
    ...

def test_load_milu_hindi_medicine_normalizes_answer_to_letter(monkeypatch):
    # whatever the real answer field type is (index/letter/text),
    # assert the returned "answer_letter" is an uppercase A-J letter
    # matching the correct option's text
    ...

def test_load_milu_hindi_medicine_preserves_variable_option_count(monkeypatch):
    # fixture includes one item with 3 options and one with 5 (or whatever
    # MILU actually has); assert n_options and len(options) match each item,
    # not a hardcoded 4
    ...

def test_load_milu_hindi_medicine_raises_on_duplicate_ids(monkeypatch):
    # fixture has two rows with the same id; assert a ValueError (or similar)
    # is raised rather than silently keeping both
    ...

def test_question_with_options_contains_all_option_lines():
    # given one normalized item, assert every "LETTER. text" line appears
    # in question_with_options
    ...
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_performance_data_milu.py -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` (function not defined yet).

- [ ] **Step 4: Implement `load_milu_hindi_medicine` in `performance_eval/data_milu.py`**

Load via `datasets.load_dataset("ai4bharat/MILU", "hi", token=hf_token)`, filter to the confirmed split/subject, normalize each row to the dict shape above using the real field mapping discovered in Step 1, raise `ValueError` on duplicate ids, build `question_with_options` by appending `f"{letter}. {text}"` lines sorted by letter (mirror `data/himed_loader.py:_format_question`'s West-style branch).

- [ ] **Step 5: Run tests to verify Step 2's tests pass**

Run: `pytest tests/test_performance_data_milu.py -v`
Expected: PASS

- [ ] **Step 6: Write the failing tests for `split_test_fewshot`**

```python
def test_split_test_fewshot_is_disjoint():
    items = [... 20 fake items with unique ids ...]
    test_items, pool = split_test_fewshot(items, fewshot_n=10, seed=42)
    test_ids = {i["id"] for i in test_items}
    pool_ids = {i["id"] for i in pool}
    assert test_ids.isdisjoint(pool_ids)
    assert len(pool) == 10
    assert len(test_items) == len(items) - 10

def test_split_test_fewshot_is_deterministic():
    items = [... ]
    a = split_test_fewshot(items, fewshot_n=10, seed=42)
    b = split_test_fewshot(items, fewshot_n=10, seed=42)
    assert [i["id"] for i in a[0]] == [i["id"] for i in b[0]]
    assert [i["id"] for i in a[1]] == [i["id"] for i in b[1]]
```

- [ ] **Step 7: Run tests to verify they fail, then implement `split_test_fewshot`, then verify they pass**

Run: `pytest tests/test_performance_data_milu.py -v` (expect FAIL, then implement with `random.seed(seed); random.sample(...)` same as `data/himed_loader.py`, then expect PASS).

- [ ] **Step 8: Add `HF_TOKEN` to `.env.example`**

```
# HuggingFace token (needed to access gated datasets like ai4bharat/MILU) —
# get one at https://huggingface.co/settings/tokens, and accept MILU's
# gate at https://huggingface.co/datasets/ai4bharat/MILU first.
HF_TOKEN=your_hf_token_here
```

- [ ] **Step 9: Commit**

```bash
git add performance_eval/__init__.py performance_eval/data_milu.py .env.example tests/test_performance_data_milu.py
git commit -m "feat(performance_eval): load and split MILU Hindi Health-and-Medicine MCQs" -m "..."
```

---

## Task 2: Optional decoding override on the HiMed-8B and Nanda wrappers

**Files:**
- Modify: `models/himed.py:111-150` (the `get_answer` function and a new module-level defaults dict)
- Modify: `models/nanda.py:119-157` (same)
- Test: `tests/test_himed_model.py` (new — pure, no GPU) and `tests/test_nanda_model.py` (new — pure, no GPU). If GPU-dependent tests for these models already exist under different filenames, add these pure tests alongside them rather than renaming anything.

**Interfaces:**
- Produces (used by Task 5): `get_answer(prompt: str, gen_kwargs: dict | None = None) -> str` on both `models/himed.py` and `models/nanda.py`. When `gen_kwargs` is `None`, behavior is byte-for-byte identical to today (so every existing `literacy_eval` caller, which calls `get_answer(prompt)` positionally, is unaffected).

- [ ] **Step 1: Write the failing tests for the pure merge helper**

```python
# tests/test_himed_model.py
from models.himed import _merge_gen_kwargs, _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_none_returns_defaults():
    assert _merge_gen_kwargs(None) == _DEFAULT_GEN_KWARGS

def test_merge_gen_kwargs_overrides_only_given_keys():
    merged = _merge_gen_kwargs({"do_sample": False})
    assert merged["do_sample"] is False
    assert merged["temperature"] == _DEFAULT_GEN_KWARGS["temperature"]
```

Mirror the same two tests in `tests/test_nanda_model.py` against `models.nanda._merge_gen_kwargs` / `_DEFAULT_GEN_KWARGS`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_himed_model.py tests/test_nanda_model.py -v`
Expected: FAIL (`ImportError` — names don't exist yet).

- [ ] **Step 3: Implement `_DEFAULT_GEN_KWARGS` and `_merge_gen_kwargs` in `models/himed.py`**

```python
_DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
}

def _merge_gen_kwargs(overrides: dict | None) -> dict:
    """Merge override generation kwargs onto this model's defaults."""
    return {**_DEFAULT_GEN_KWARGS, **(overrides or {})}
```

Change `get_answer`'s signature to `get_answer(prompt: str, gen_kwargs: dict | None = None) -> str` and replace the inline keyword arguments on `_model.generate(...)` with `**_merge_gen_kwargs(gen_kwargs)`, keeping every current default value unchanged (`max_new_tokens=1024, do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.2` — copy these out of the current body into `_DEFAULT_GEN_KWARGS` exactly).

Do the same in `models/nanda.py`, moving its current inline values (`max_new_tokens=512, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.3`) into its own `_DEFAULT_GEN_KWARGS`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_himed_model.py tests/test_nanda_model.py -v`
Expected: PASS

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `pytest tests/ -v -k "himed or nanda or evaluate"`
Expected: PASS (existing GPU-skip smoke tests still skip cleanly off-cluster; nothing that called `get_answer(prompt)` positionally breaks, since `gen_kwargs` defaults to `None`).

- [ ] **Step 6: Commit**

```bash
git add models/himed.py models/nanda.py tests/test_himed_model.py tests/test_nanda_model.py
git commit -m "feat(models): allow optional gen_kwargs override on HiMed-8B and Nanda" -m "..."
```

---

## Task 3: Condition-aware MCQ prompt builder

**Files:**
- Create: `performance_eval/prompts.py`
- Test: `tests/test_performance_prompts.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure string logic); items shaped as Task 1 produces (`question_with_options: str`, `n_options: int`).
- Produces (used by Task 5): `build_prompt(question_with_options: str, n_options: int, use_cot: bool = False, exemplars: list[dict] | None = None) -> str`. `exemplars`, when given, is a list of `{"question_with_options": str, "answer_letter": str}` dicts (shape of Task 1's items), rendered **before** the target question, in the order given.

Exact copy fixed by the spec: instructions in English, question content in Hindi (matches the convention already used in `models/base.py` and `literacy_eval/judge`). The CoT condition's required final line is the literal English marker `Answer:` followed by the single letter — e.g. `Answer: C` — so the scorer (Task 4) has one fixed string to search for across both models and both languages of instruction.

- [ ] **Step 1: Write the failing tests**

```python
def test_baseline_prompt_has_no_cot_or_fewshot_instructions():
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2)
    assert "Answer:" not in p  # no marker required — strict single-letter only
    assert "step by step" not in p.lower()
    assert "प्रश्न?" in p

def test_baseline_prompt_lists_exact_option_letters():
    p = build_prompt("प्रश्न?\nA. एक\nB. दो\nC. तीन", n_options=3)
    assert "A, B, or C" in p  # not "A, B, C, D, or E"

def test_cot_prompt_requires_answer_marker_and_reasoning():
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2, use_cot=True)
    assert "step by step" in p.lower() or "step-by-step" in p.lower()
    assert "Answer:" in p

def test_fewshot_prompt_renders_exemplars_before_target_question_in_order():
    exemplars = [
        {"question_with_options": "उदाहरण 1?\nA. क\nB. ख", "answer_letter": "A"},
        {"question_with_options": "उदाहरण 2?\nA. ग\nB. घ", "answer_letter": "B"},
    ]
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2, exemplars=exemplars)
    assert p.index("उदाहरण 1?") < p.index("उदाहरण 2?") < p.index("प्रश्न?")
    assert "A" in p and "B" in p  # exemplar answers appear

def test_fewshot_and_cot_compose():
    exemplars = [{"question_with_options": "उदाहरण?\nA. क\nB. ख", "answer_letter": "A"}]
    p = build_prompt("प्रश्न?\nA. एक\nB. दो", n_options=2, use_cot=True, exemplars=exemplars)
    assert "Answer:" in p
    assert "उदाहरण?" in p

def test_build_prompt_handles_ten_options():
    # letters A..J — reuses the same "last letter" style as models/base.py
    stem = "प्रश्न?\n" + "\n".join(f"{chr(65+i)}. विकल्प{i}" for i in range(10))
    p = build_prompt(stem, n_options=10)
    assert "A, B, C, D, E, F, G, H, I, or J" in p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_performance_prompts.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `build_prompt` in `performance_eval/prompts.py`**

Internals needed (small, independent of `models/base.py`'s private helpers — this module owns its own letter-list rendering so it has no cross-module private coupling):
- `_make_letter_list(n_options: int) -> str`: same rendering as `models/base.py:_make_letter_list` (`"A, B, or C"`, `"A, B, C, D, E, F, G, H, I, or J"`), derived from `n_options` directly rather than scanning the question text (Task 1 already provides the true count, so no re-detection is needed here).
- Compose, in order: (a) a `"Here are example questions:"` block rendering each exemplar as its question-with-options followed by `Answer: {answer_letter}` on its own line, only if `exemplars` is given; (b) the instruction block — strict single-letter ("respond with ONLY the letter... Do not explain.") when `use_cot=False`, or a step-by-step-then-marker instruction when `use_cot=True` ("Think step by step in Hindi. On the final line, write exactly `Answer: X` where X is the single letter. Do not write anything after that line."); (c) `f"Question: {question_with_options}\n\nAnswer"` (append `" (single letter only):"` when not `use_cot`, else leave open for reasoning).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_performance_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add performance_eval/prompts.py tests/test_performance_prompts.py
git commit -m "feat(performance_eval): condition-aware MCQ prompt builder (baseline/CoT/few-shot)" -m "..."
```

---

## Task 4: Two-mode answer scorer

**Files:**
- Create: `performance_eval/score.py`
- Test: `tests/test_performance_score.py`

**Interfaces:**
- Consumes: `score_mcq.extract_letter` (reused directly for `"first_letter"` mode — do not duplicate it).
- Produces (used by Task 6): `extract_answer(text: str, mode: str) -> str | None` where `mode` is `"first_letter"` or `"after_marker"`; `score_rows(rows: list[dict], mode: str) -> list[dict]` where each input row has `{"item_id": str, "reference_answer": str, "model_answer": str}` and each output row adds `{"parsed_letter": str | None, "is_correct": bool}`.

- [ ] **Step 1: Write the failing tests for `extract_answer` in `"after_marker"` mode**

```python
def test_after_marker_extracts_letter_following_english_marker():
    text = "यह बीमारी के कारण... इसलिए सही विकल्प है।\nAnswer: C"
    assert extract_answer(text, mode="after_marker") == "C"

def test_after_marker_falls_back_to_last_standalone_letter_when_marker_missing():
    text = "विकल्प A गलत है। विकल्प B सही है।"
    assert extract_answer(text, mode="after_marker") == "B"

def test_after_marker_returns_none_when_no_letter_anywhere():
    text = "यह उत्तर देना मुश्किल है।"
    assert extract_answer(text, mode="after_marker") is None

def test_after_marker_prefers_marker_over_earlier_mentions():
    text = "शुरू में लगता है A सही है, लेकिन सोचने पर...\nAnswer: D"
    assert extract_answer(text, mode="after_marker") == "D"

def test_first_letter_mode_delegates_to_score_mcq_extract_letter():
    # mirrors score_mcq.extract_letter's own documented behavior
    assert extract_answer("A. वात रोग", mode="first_letter") == "A"
    assert extract_answer("", mode="first_letter") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_performance_score.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `extract_answer` in `performance_eval/score.py`**

```python
import re
from score_mcq import extract_letter as _extract_first_letter

_MARKER_RE = re.compile(r"Answer\s*[:：]?\s*([A-Ja-j])(?![A-Za-z])", re.IGNORECASE)
_LAST_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])")

def extract_answer(text: str, mode: str) -> str | None:
    if mode == "first_letter":
        return _extract_first_letter(text)
    if mode == "after_marker":
        marker_matches = list(_MARKER_RE.finditer(text))
        if marker_matches:
            return marker_matches[-1].group(1).upper()
        letter_matches = list(_LAST_LETTER_RE.finditer(text))
        if letter_matches:
            return letter_matches[-1].group(1).upper()
        return None
    raise ValueError(f"Unknown mode: {mode!r}")
```

(Use the **last** marker match, not the first, in case the model restates its answer.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_performance_score.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing tests for `score_rows`**

```python
def test_score_rows_marks_correct_and_incorrect():
    rows = [
        {"item_id": "x1", "reference_answer": "B", "model_answer": "Answer: B"},
        {"item_id": "x2", "reference_answer": "A", "model_answer": "Answer: C"},
    ]
    scored = score_rows(rows, mode="after_marker")
    assert scored[0]["is_correct"] is True
    assert scored[1]["is_correct"] is False

def test_score_rows_counts_unparseable_as_incorrect_not_skipped():
    rows = [{"item_id": "x1", "reference_answer": "A", "model_answer": "कुछ नहीं पता"}]
    scored = score_rows(rows, mode="after_marker")
    assert scored[0]["parsed_letter"] is None
    assert scored[0]["is_correct"] is False  # NONE counts as wrong (spec resolution 4)
```

- [ ] **Step 6: Run tests to verify they fail, implement `score_rows`, then verify they pass**

`score_rows` calls `extract_answer` per row and sets `is_correct = (parsed_letter == reference_answer.strip().upper())`, `False` when `parsed_letter is None`. No row is dropped or marked "skipped" (unlike `score_mcq.py`'s dataset-error skip path, which doesn't apply here since Task 1 already validated every row).

Run: `pytest tests/test_performance_score.py -v`
Expected: FAIL then PASS.

- [ ] **Step 7: Commit**

```bash
git add performance_eval/score.py tests/test_performance_score.py
git commit -m "feat(performance_eval): two-mode (first-letter / after-marker) MCQ scorer" -m "..."
```

---

## Task 5: Run harness — generate answers per (model, condition)

**Files:**
- Create: `performance_eval/run.py`
- Test: `tests/test_performance_run.py`

**Interfaces:**
- Consumes: `data_milu.load_milu_hindi_medicine`, `data_milu.split_test_fewshot` (Task 1); `prompts.build_prompt` (Task 3); any model's `get_answer(prompt, gen_kwargs=None)` (Task 2).
- Produces: CSV at `performance_eval/outputs/milu_hi_med/{model}_{condition}_raw.csv` with fieldnames `["item_id", "question_with_options", "reference_answer", "model_answer"]`. Resume is keyed on `item_id` (not question text — unlike `evaluate.py`, because prompt text differs per condition while `item_id` is stable).

- [ ] **Step 1: Write the failing tests using a fake `get_answer`**

```python
def test_run_condition_writes_one_row_per_item(tmp_path):
    items = [{"id": "m1", "question_with_options": "Q1?\nA. x\nB. y", "answer_letter": "A", "n_options": 2}]
    fake_get_answer = lambda prompt, gen_kwargs=None: "Answer: A"
    out = tmp_path / "out.csv"
    run_condition(items, [], get_answer_fn=fake_get_answer, use_cot=False, use_fewshot=False,
                   gen_kwargs={"do_sample": False}, output_path=out)
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["item_id"] == "m1"
    assert rows[0]["model_answer"] == "Answer: A"

def test_run_condition_resumes_by_item_id_skipping_done_rows(tmp_path):
    out = tmp_path / "out.csv"
    out.write_text("item_id,question_with_options,reference_answer,model_answer\nm1,Q1,A,done\n", encoding="utf-8")
    items = [
        {"id": "m1", "question_with_options": "Q1", "answer_letter": "A", "n_options": 2},
        {"id": "m2", "question_with_options": "Q2", "answer_letter": "B", "n_options": 2},
    ]
    calls = []
    fake_get_answer = lambda prompt, gen_kwargs=None: calls.append(prompt) or "Answer: B"
    run_condition(items, [], get_answer_fn=fake_get_answer, use_cot=False, use_fewshot=False,
                   gen_kwargs={}, output_path=out)
    assert len(calls) == 1  # only m2 was queried
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert {r["item_id"] for r in rows} == {"m1", "m2"}

def test_run_condition_passes_fewshot_pool_into_prompt_when_enabled(tmp_path, monkeypatch):
    captured = {}
    def fake_build_prompt(question_with_options, n_options, use_cot=False, exemplars=None):
        captured["exemplars"] = exemplars
        return "PROMPT"
    monkeypatch.setattr("performance_eval.run.build_prompt", fake_build_prompt)
    items = [{"id": "m1", "question_with_options": "Q1", "answer_letter": "A", "n_options": 2}]
    pool = [{"question_with_options": "ex?", "answer_letter": "A"}]
    out = tmp_path / "out.csv"
    run_condition(items, pool, get_answer_fn=lambda p, gen_kwargs=None: "Answer: A",
                   use_cot=False, use_fewshot=True, gen_kwargs={}, output_path=out)
    assert captured["exemplars"] == pool
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_performance_run.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `run_condition` in `performance_eval/run.py`**

```python
def run_condition(
    items: list[dict],
    fewshot_pool: list[dict],
    get_answer_fn: Callable[..., str],
    use_cot: bool,
    use_fewshot: bool,
    gen_kwargs: dict,
    output_path: Path,
) -> None:
```

Mirror `evaluate.py:run_evaluation`'s structure exactly: load already-done `item_id`s from `output_path` if it exists (set comprehension over the CSV's `item_id` column), filter `items` to the remaining ones, open the CSV in append mode (write header if new), and for each remaining item build the prompt (`build_prompt(item["question_with_options"], item["n_options"], use_cot=use_cot, exemplars=fewshot_pool if use_fewshot else None)`), call `get_answer_fn(prompt, gen_kwargs=gen_kwargs)`, write the row, flush.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_performance_run.py -v`
Expected: PASS

- [ ] **Step 5: Add the CLI entry point**

```python
def main():
    parser = argparse.ArgumentParser(description="Run a model on a MILU MCQ condition")
    parser.add_argument("--model", required=True, choices=["himed", "nanda"])
    parser.add_argument("--condition", required=True, help="label used in the output filename, e.g. c0_baseline")
    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--use-fewshot", action="store_true")
    parser.add_argument("--fewshot-n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                         help="override the model's default max_new_tokens (needed for CoT, per spec resolution 2)")
    parser.add_argument("--sample-decoding", action="store_true",
                         help="C4 only: use the model's default sampling instead of greedy")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs" / "milu_hi_med")
    args = parser.parse_args()
    ...
```

Build `gen_kwargs`: `{"do_sample": False}` unless `--sample-decoding` is passed (per Global Constraints — greedy is the default for C0–C3); merge in `{"max_new_tokens": args.max_new_tokens}` only if given. Load items via `data_milu.load_milu_hindi_medicine()` + `split_test_fewshot(items, fewshot_n=args.fewshot_n)`, import the right model module (`models.himed` or `models.nanda`) by name (same `importlib.import_module` pattern as `evaluate.py:_get_model_function`), and call `run_condition(...)` with `output_path = args.output_dir / f"{args.model}_{args.condition}_raw.csv"`.

- [ ] **Step 6: Commit**

```bash
git add performance_eval/run.py tests/test_performance_run.py
git commit -m "feat(performance_eval): run harness with item-id resume for MILU MCQ conditions" -m "..."
```

---

## Task 6: Analysis — accuracy, CIs, paired McNemar between conditions

**Files:**
- Create: `performance_eval/analyze.py`
- Test: `tests/test_performance_analyze.py`

**Interfaces:**
- Consumes: `score.score_rows` (Task 4); `literacy_eval.analysis.stats.mcnemar_exact`, `wilson_ci`, `holm` (existing, reused as-is — do not reimplement).
- Produces: `accuracy_summary(scored_rows: list[dict]) -> dict` (`{"n", "n_correct", "accuracy", "ci_low", "ci_high", "none_rate"}`); `compare_conditions(prior_scored: list[dict], new_scored: list[dict]) -> dict` (`{"n_paired", "accuracy_delta", "mcnemar_p", "only_prior_correct", "only_new_correct"}`, items aligned by `item_id`); `build_ladder_rows(model: str, ladder: list[dict], n_options_mode: int) -> list[dict]` (applies Holm correction across one model's ladder of `accuracy_summary`/`compare_conditions` results); `write_results_md(summary_rows: list[dict], out_path: Path) -> None`.

- [ ] **Step 1: Write the failing tests for `accuracy_summary`**

```python
def test_accuracy_summary_counts_correct_and_none_rate():
    rows = [
        {"item_id": "a", "is_correct": True, "parsed_letter": "A"},
        {"item_id": "b", "is_correct": False, "parsed_letter": None},
        {"item_id": "c", "is_correct": False, "parsed_letter": "B"},
    ]
    s = accuracy_summary(rows)
    assert s["n"] == 3
    assert s["n_correct"] == 1
    assert abs(s["accuracy"] - 1/3) < 1e-9
    assert abs(s["none_rate"] - 1/3) < 1e-9
    assert s["ci_low"] <= s["accuracy"] <= s["ci_high"]
```

- [ ] **Step 2: Run tests to verify they fail, implement `accuracy_summary`, then verify they pass**

Implementation: `n = len(rows)`, `n_correct = sum(r["is_correct"] for r in rows)`, `none_rate = mean(r["parsed_letter"] is None for r in rows)`, `(ci_low, ci_high) = literacy_eval.analysis.stats.wilson_ci(n_correct, n)`.

Run: `pytest tests/test_performance_analyze.py -v` (FAIL then PASS).

- [ ] **Step 3: Write the failing tests for `compare_conditions`**

```python
def test_compare_conditions_aligns_by_item_id_and_counts_flips():
    prior = [
        {"item_id": "a", "is_correct": True},
        {"item_id": "b", "is_correct": False},
        {"item_id": "c", "is_correct": False},
    ]
    new = [
        {"item_id": "a", "is_correct": True},
        {"item_id": "b", "is_correct": True},   # flipped wrong -> correct
        {"item_id": "c", "is_correct": False},
    ]
    c = compare_conditions(prior, new)
    assert c["n_paired"] == 3
    assert c["only_new_correct"] == 1
    assert c["only_prior_correct"] == 0
    assert c["accuracy_delta"] > 0
    assert 0.0 <= c["mcnemar_p"] <= 1.0

def test_compare_conditions_ignores_items_missing_from_either_side():
    prior = [{"item_id": "a", "is_correct": True}, {"item_id": "b", "is_correct": False}]
    new = [{"item_id": "a", "is_correct": True}]  # "b" missing (e.g. not yet run)
    c = compare_conditions(prior, new)
    assert c["n_paired"] == 1
```

- [ ] **Step 4: Run tests to verify they fail, implement `compare_conditions`, then verify they pass**

Implementation: build `{item_id: is_correct}` dicts for both sides, intersect keys, build two aligned boolean arrays in a fixed common order, call `literacy_eval.analysis.stats.mcnemar_exact(prior_arr, new_arr)` for `only_a`/`only_b`/`p` (map `only_a -> only_prior_correct`, `only_b -> only_new_correct`), compute `accuracy_delta = mean(new_arr) - mean(prior_arr)`.

Run: `pytest tests/test_performance_analyze.py -v` (FAIL then PASS).

- [ ] **Step 5: Write the failing test for `write_results_md`**

```python
def test_write_results_md_includes_chance_baseline_and_holm_column(tmp_path):
    rows = [{"model": "himed", "condition": "c0_baseline", "n": 100, "accuracy": 0.3,
             "ci_low": 0.21, "ci_high": 0.40, "none_rate": 0.05, "n_options_mode": 4,
             "accuracy_delta": None, "mcnemar_p": None, "mcnemar_p_holm": None}]
    out = tmp_path / "results.md"
    write_results_md(rows, out)
    text = out.read_text(encoding="utf-8")
    assert "chance" in text.lower()
    assert "0.3" in text or "30" in text  # accuracy visible
```

- [ ] **Step 6: Run test to verify it fails, implement `write_results_md`, then verify it passes**

Implementation: write a Markdown table, one row per `(model, condition)`, columns `model | condition | n | accuracy (CI) | none_rate | Δ vs prior | McNemar p (Holm) | chance baseline`; chance baseline is `1 / n_options_mode` stated once in a header note (per Global Constraints, this must be visible on the table, not buried).

Run: `pytest tests/test_performance_analyze.py -v` (FAIL then PASS).

- [ ] **Step 7: Write the failing test for `build_ladder_rows`**

This is the piece that actually applies Holm correction across one model's ladder — `write_results_md` only renders rows it's handed; something must first collect a model's condition results in order and Holm-correct their McNemar p-values together, per Global Constraints ("Holm-corrected within each model's ladder").

```python
def test_build_ladder_rows_holm_corrects_across_the_ladder():
    # 3 conditions for one model: baseline has no comparison (mcnemar_p=None),
    # the next two each compare against the prior kept condition.
    ladder = [
        {"condition": "c0_baseline", "summary": {"n": 100, "n_correct": 30, "accuracy": 0.30,
          "ci_low": 0.22, "ci_high": 0.40, "none_rate": 0.05}, "comparison": None},
        {"condition": "c2_cot", "summary": {"n": 100, "n_correct": 38, "accuracy": 0.38,
          "ci_low": 0.29, "ci_high": 0.48, "none_rate": 0.02},
          "comparison": {"n_paired": 100, "accuracy_delta": 0.08, "mcnemar_p": 0.03,
                          "only_prior_correct": 2, "only_new_correct": 10}},
        {"condition": "c3_fewshot", "summary": {"n": 100, "n_correct": 40, "accuracy": 0.40,
          "ci_low": 0.31, "ci_high": 0.50, "none_rate": 0.01},
          "comparison": {"n_paired": 100, "accuracy_delta": 0.02, "mcnemar_p": 0.04,
                          "only_prior_correct": 3, "only_new_correct": 5}},
    ]
    rows = build_ladder_rows("himed", ladder, n_options_mode=4)
    assert rows[0]["mcnemar_p"] is None and rows[0]["mcnemar_p_holm"] is None
    # Holm on [0.03, 0.04] (only the two non-None p-values): first is doubled, must stay <= 1
    assert rows[1]["mcnemar_p_holm"] == rows[1]["mcnemar_p"] * 2
    assert rows[2]["mcnemar_p_holm"] >= rows[2]["mcnemar_p"]
    assert all(r["model"] == "himed" for r in rows)
    assert [r["condition"] for r in rows] == ["c0_baseline", "c2_cot", "c3_fewshot"]
```

- [ ] **Step 8: Run test to verify it fails, implement `build_ladder_rows`, then verify it passes**

```python
def build_ladder_rows(model: str, ladder: list[dict], n_options_mode: int) -> list[dict]:
```

Pull the `mcnemar_p` out of each entry's `comparison` (or `None` for the first), run `literacy_eval.analysis.stats.holm` over just the non-`None` values in ladder order, scatter the corrected values back (in the same positions, `None` stays `None`), and assemble each row as the flat dict shape `write_results_md`'s test (Step 5/6) expects: `{"model", "condition", "n", "accuracy", "ci_low", "ci_high", "none_rate", "n_options_mode", "accuracy_delta", "mcnemar_p", "mcnemar_p_holm"}`.

Run: `pytest tests/test_performance_analyze.py -v` (FAIL then PASS).

- [ ] **Step 9: Commit**

```bash
git add performance_eval/analyze.py tests/test_performance_analyze.py
git commit -m "feat(performance_eval): accuracy/CI/McNemar analysis and results.md writer" -m "..."
```

---

## Task 7: SLURM run + smoke-test scripts

**Files:**
- Create: `scripts/performance_milu_run.sh`
- Create: `scripts/performance_milu_smoke_test.sh`

**Interfaces:**
- Consumes: `performance_eval/run.py`'s CLI (Task 5).
- Produces: nothing consumed by later tasks — this is the operational entry point for the user.

No unit tests apply to shell scripts in this repo's convention (none exist for the `literacy_*` scripts either); verification is running the smoke test on the cluster and reading its output.

- [ ] **Step 1: Write `scripts/performance_milu_smoke_test.sh`**

Copy the structure of `scripts/literacy_himed_trad_smoke_test.sh` exactly (SBATCH header with `--time=00:30:00`, `--gres=gpu:1`, `$SCRATCH`/`$HF_HOME`/`$TRITON_CACHE_DIR`/`$PIP_CACHE_DIR` setup, venv activation), but:
- Usage: `sbatch scripts/performance_milu_smoke_test.sh <model> <condition> [--use-cot] [--use-fewshot]`, validating `model` is `himed` or `nanda` (not the five-way check the literacy scripts use — this track only has two models).
- Runs `python -m performance_eval.run --model "$MODEL" --condition "$CONDITION" "$@" --output-dir "$SCRATCH/performance_eval_outputs"` restricted to a tiny item count for the smoke test — add a `--limit N` flag to `performance_eval/run.py`'s CLI (Task 5's `main()`; trivial addition: slice `items = items[:args.limit]` if `args.limit` is given) and pass `--limit 2` here.
- Prints the resulting CSV at the end (`cat` it), same as the literacy smoke test does.

- [ ] **Step 2: Write `scripts/performance_milu_run.sh`**

Same structure, full run (no `--limit`), `--time=04:00:00` (matching the literacy full-run budget — adjust after the smoke test gives a real per-item timing).

- [ ] **Step 3: Add the `--limit` flag to `performance_eval/run.py`'s `main()`**

(Folded in here since Task 5's `main()` is the file being extended, and this flag only exists to serve the smoke test — no separate test needed beyond Task 5's existing CLI coverage; this is a one-line slice with no new branching logic to unit-test.)

- [ ] **Step 4: Make both scripts executable and commit**

```bash
chmod +x scripts/performance_milu_run.sh scripts/performance_milu_smoke_test.sh
git add scripts/performance_milu_run.sh scripts/performance_milu_smoke_test.sh performance_eval/run.py
git commit -m "feat(performance_eval): SLURM run and smoke-test scripts for PACE" -m "..."
```

- [ ] **Step 5: Verification (manual, on PACE — not a pytest run)**

```bash
sbatch scripts/performance_milu_smoke_test.sh himed c0_baseline
sbatch scripts/performance_milu_smoke_test.sh nanda c0_baseline
```
Expected: each job completes within its time budget and the printed CSV has 2 rows with non-empty `model_answer` values. This confirms the HF gate/token, model loading, and prompt/generation path work end-to-end before committing to the full 499-item run.

---

## After Task 7: running the actual ladder (not a coding task)

Once all seven tasks are merged, the week's experiment is run by hand, adaptively, per the spec's "one-at-a-time, keep only if it wins" rule — this is the research itself, not something a task can pre-script:

1. `sbatch scripts/performance_milu_run.sh himed c0_baseline` and `... nanda c0_baseline`.
2. Read the raw outputs. Decide whether C1 (prompt refinement) is needed at all (skip if C0 is already clean, per the spec).
3. Run C2 (`--use-cot`, with `--max-new-tokens` raised enough for both models to reach the marker — check this empirically from C0's typical output length before picking a number).
4. Run `performance_eval/analyze.py` comparing C0 vs C2 per model; keep C2 only if `accuracy_delta > 0` and Holm-corrected McNemar p < 0.05.
5. Run C3 (`--use-fewshot`, plus `--use-cot` only if C2 was kept) on top of the best-kept config; compare and keep/reject the same way.
6. Run C4 (`--sample-decoding`, tuning temperature/repetition penalty) on top of the best-kept config; compare and keep/reject the same way.
7. For each model, feed its ordered list of kept/rejected condition results into `build_ladder_rows`, concatenate both models' rows, and pass them to `write_results_md` to produce `performance_eval/analysis_outputs/results.md`.

## Self-Review Notes

- **Spec coverage:** dataset loading + freezing (Task 1), deterministic decoding support (Task 2), every prompt condition in the ladder (Task 3), the two scoring modes the spec calls for (Task 4), the run/resume harness (Task 5), accuracy/CI/McNemar/Holm + chance-baseline reporting (Task 6), and the PACE launch mechanism (Task 7) each map to one task. The adaptive "keep/reject per rung" sequencing is explicitly left as the post-plan runbook above, since the spec itself makes that sequencing data-dependent, not fixable at plan-writing time.
- **Type/signature consistency check:** `build_prompt(question_with_options, n_options, use_cot=, exemplars=)` (Task 3) is the exact signature `run_condition` (Task 5) calls; `extract_answer(text, mode)` / `score_rows(rows, mode)` (Task 4) are the exact names `analyze.py` (Task 6) builds on via already-scored rows; `item_id` is the consistent join key across Tasks 1, 5, and 6.
- **Review Focus coverage:** variable option count → Task 1 Step 2 test + Task 3 Step 1 ten-option test; CoT marker-missing fallback → Task 4 Step 1 tests; few-shot/test leakage → Task 1 Step 6 test; duplicate ids → Task 1 Step 2 test; resume-after-crash → Task 5 Step 1 test.
