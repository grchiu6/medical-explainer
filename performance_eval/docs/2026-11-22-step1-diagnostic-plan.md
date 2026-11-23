# Step 1 Diagnostic (Commercial Ceiling + English Comparison) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out whether the ~60–65% MILU accuracy is limited by the open Indic models, by the Hindi language, or by the questions, after removing government-initiative questions.

**Architecture:** Freeze an exclusion list of government-initiative items, then run a 3 model × 2 language grid (HiMed-8B, Nanda, Gemini 3.8 Flash × Hindi, English) on the same filtered items. English items are machine-translations of the Hindi items and keep the same `item_id` and gold letter, so every comparison is paired. A bonus run reuses the translations as English retrieval queries (the translation-based pipeline of the *Language Bias in Multilingual RAG* paper).

**Tech Stack:** Python, `google-genai` (≥1.73, already installed), existing `performance_eval` runner/scorer/analyzer, PACE (Slurm) for the open models.

**Spec:** this document (Decisions section) + `performance_eval/docs/2026-11-15-mcq-performance-techniques-design.md` (stats rules, NONE rule, greedy decoding) + `performance_eval/docs/2026-11-16-rag-design.md` (bonus task).

## Why (one paragraph for the advisor)

Baseline is 61.3% (HiMed-8B) and 64.6% (Nanda) on 489 MILU Hindi Health & Medicine items; CoT (+3.3 pp) and English-corpus RAG did not give significant gains. Advisor feedback: (a) test a commercial LLM to see whether the target accuracy is achievable at all, (b) test English to see whether Hindi is the cause. The MILU paper notes this subject includes health-related government-initiative questions (schemes, missions, appointments, dated announcements). Those test current-affairs recall, not medical knowledge, so we remove them before drawing conclusions.

## Decisions (locked; veto welcome)

- **D1 — What gets excluded.** An item is excluded if answering it requires knowing about a specific Indian **government scheme, programme, mission, campaign, portal, policy, official survey/report, official appointment, or dated government action/announcement**. Not excluded: disease/drug/anatomy knowledge, and general epidemiology even if it mentions India. Borderline → Grace decides on the hand-check sheet; her label is final.
- **D2 — How items are labelled (DONE 2026-11-22).** Claude read all 489 test items (question + options only, never gold or model answers) and gave each one label in `performance_eval/data/item_labels.csv`: `gov` (47), `non_medical` (31: literature, engineering/physics, celebrity/business trivia, pure logic/arithmetic puzzles, ecology), `current_affairs` (23: dated non-government news, Nobel/WHO appointments), `keep` (388). All 101 non-`keep` items are listed in `performance_eval/data/excluded_items.csv` (current_affairs added at Grace's request, 2026-11-22); **388 items remain**. Single rater (an LLM), no human check; labels are visible with question text so any reader can audit them.
- **D3 — Filter after the split.** Exclusions are applied to the test set *after* `split_test_fewshot`, so existing HiMed/Nanda C0 outputs remain valid and are re-analysed, not re-run. The few-shot pool is unchanged (not used in Step 1).
- **D4 — Commercial model.** `gemini-3.8-flash`, paid tier, temperature 0, lowest thinking level the API accepts, one question per call, same prompt as C0. Hard budget guard of **$2** (Grace's cap) in the wrapper and a billing cap in Google Cloud. Run before **2027-01-01** (prices double). Price constants: $0.75 / 1M input, $3.75 / 1M output incl. thinking.
- **D5 — Translation (no human validator available).** Two independent translations:
  - **Primary: Google Cloud Translation (NMT, v2 Basic), `source=hi, target=en, format=text`.** It is not one of the models being compared, so it cannot "help" any one model (an LLM translator can silently resolve ambiguity using its own knowledge, which would inflate Gemini's English score if Gemini also translates). Question and each option are sent as separate strings in one request, so option structure is preserved. ~60K characters, inside the 500K chars/month free allowance.
  - **Secondary: Gemini 3.8 Flash** (question + options together, JSON out).
  - **Automatic checks instead of a human:** structural flags (Task 4), plus agreement between the two translations (chrF of primary vs secondary, per item); the lowest-agreement 10% are listed in the report.
  - **Sensitivity instead of validation:** run the English condition with both translations. A conclusion counts only if it holds under both. Items flagged `option_count_mismatch`/`empty_option`/`bad_json` in the primary translation are dropped from both languages so comparisons stay paired.
- **D6 — English prompt.** Identical to the Hindi prompt except "in Hindi" → "in English". The Hindi prompt text is not changed (existing outputs stay valid).
- **D7 — Pre-registered comparisons** (two-sided exact McNemar, Bonferroni m = 5, keep α = 0.05), all on the same filtered item set:
  1. gemini/hi vs himed/hi
  2. gemini/hi vs nanda/hi
  3. himed/en vs himed/hi
  4. nanda/en vs nanda/hi
  5. gemini/en vs gemini/hi
- **D8 — How to read the results** (fixed before any run):
  | Outcome | Conclusion | Next |
  |---|---|---|
  | Comparison 1 **and** 2 significant with gemini ≥ 10 pp above the better open model | Open Indic models are the bottleneck | LoRA fine-tuning (translated MedMCQA) or stronger base model |
  | Comparisons 3 **and** 4 significant and positive (≥ 5 pp) | Hindi is the bottleneck | Translate-then-answer pipeline, Hindi training data |
  | Neither | Ceiling is in the questions | Audit wrong answers of all three models |
  Both rows can hold at once; report both.
- **D9 — Branch.** New branch `perf/step1-diagnostic` from `perf/rag` (which already contains `perf/mcq-techniques` except commit `adff02b`; cherry-pick it), in a git worktree so the uncommitted literacy changes on the current checkout are untouched.

## Global Constraints

- All PACE caches/outputs under `~/scratch`; NVIDIA GPUs only.
- Commit email: `grchiu6@gmail.com`.
- Open-model decoding stays greedy (`build_gen_kwargs(False)`); C0 prompt and scorer mode `first_letter` for baseline runs.
- Accuracy = correct / total, NONE counted wrong; NONE rate reported separately.
- Never send gold answers or model answers to the tagger or translator.
- `literacy_eval/` and `models/gemini.py` (used by literacy eval) are not modified.
- Secrets only via `.env` (`GEMINI_API_KEY`); never in config sidecars or logs.

## Review Focus

1. Gemini returns no text (safety block or empty candidate) on a medical question → record `""` (scored NONE), log the finish reason, do not crash. *(Task 2 test)*
2. A run is interrupted and resumed → token usage and cost accumulate across resumes instead of resetting, so the budget guard still holds. *(Task 2 test)*
3. An exclusion-file id matches no loaded item (typo, or the content-hash id changed) → fail loudly, never silently exclude nothing. *(Task 1 test)*
4. Translation changes the number of options, leaves an option empty, or drops the `1. 2. 3.` statement numbering of "consider the following statements" items → flagged for hand-check, not silently used. *(Task 4 test)*
5. Gemini wraps the letter in markdown (`**B**`, `Answer: B`) → scored as B. *(Task 2 test)*

---

### Task 1: Exclusion list support (runner + analyzer)

**Files:**
- Create: `performance_eval/exclusions.py`
- Modify: `performance_eval/run.py` (CLI + config sidecar), `performance_eval/analyze.py` (CLI)
- Test: `tests/test_performance_exclusions.py`

**Interfaces:**
- Produces: `load_exclusions(path: Path) -> dict[str, str]` (item_id → reason; CSV columns `item_id,reason`); `apply_exclusions(items: list[dict], excluded: dict[str, str], key: str = "id") -> list[dict]` (raises `ValueError` listing ids not present in `items`); `--exclude-file PATH` on both `run.py` and `analyze.py`. Run config sidecar gains `"exclude_file"` and `"exclude_sha256"` (None when unused).

- [ ] **Step 0: Set up branch** — `git worktree add ../step1-wt -b perf/step1-diagnostic perf/rag`, then `git cherry-pick adff02b` inside it. Run `pytest tests -q`; expected: all pass.
- [ ] **Step 1: Write failing tests**
  - `test_apply_exclusions_drops_listed_ids`: items with ids a,b,c; excluded {b} → ids [a, c], order kept.
  - `test_apply_exclusions_unknown_id_raises`: excluded {zzz} → `ValueError` whose message contains `zzz`.
  - `test_load_exclusions_requires_columns`: CSV missing `reason` column → `ValueError`.
  - `test_analyze_exclude_file_filters_scored_rows`: two scored CSVs with ids a,b,c; `--exclude-file` listing b → results table n = 2.
- [ ] **Step 2: Run** `pytest tests/test_performance_exclusions.py -v` — expected FAIL (module missing).
- [ ] **Step 3: Implement** `exclusions.py`; in `run.py` apply after `split_test_fewshot` and before `--limit`; in `analyze.py` filter every ladder entry's scored rows before `check_item_ids`. For the analyzer, ids missing from a scored CSV are allowed (scored rows may already be filtered) — only `run.py` uses the strict check.
- [ ] **Step 4: Run** `pytest tests -q` — expected all PASS.
- [ ] **Step 5: Commit** `feat(performance_eval): --exclude-file for item exclusions`.

### Task 2: Gemini 3.8 Flash MCQ wrapper

**Files:**
- Create: `models/gemini_flash.py`
- Modify: `performance_eval/run.py` (`--model` choices, config `model_id`, usage sidecar)
- Test: `tests/test_gemini_flash.py` (fake client via monkeypatching `_get_client`; no network)

**Interfaces:**
- Produces: `MODEL_ID = "gemini-3.8-flash"`; `PRICE_IN_PER_M = 0.75`, `PRICE_OUT_PER_M = 3.75`; `get_answer(prompt: str, gen_kwargs: dict | None = None) -> str` (ignores HF-only keys; `gen_kwargs["thinking_level"]` overrides the default lowest level); `USAGE: dict` with keys `calls, prompt_tokens, output_tokens, thoughts_tokens, cost_usd, blocked`; `load_usage(path: Path) -> None` and `save_usage(path: Path) -> None`; `BudgetExceeded(RuntimeError)` raised *before* a call when `USAGE["cost_usd"] >= float(os.getenv("GEMINI_BUDGET_USD", "2"))`.
- `run.py`: `--model gemini_flash`; config sidecar `"model_id"` = module `MODEL_ID` if present else None; usage at `<raw>.usage.json`, loaded before the run and saved after every call.

- [ ] **Step 1: Write failing tests**
  - `test_request_uses_temperature_zero_and_low_thinking`: fake client captures config → `temperature == 0`, thinking level == module default.
  - `test_usage_accumulates_and_costs`: fake response with usage (100 prompt, 1 output, 50 thoughts) twice → `calls == 2`, `cost_usd == pytest.approx(2*(100*0.75 + 51*3.75)/1e6)`.
  - `test_usage_resumes_from_file`: `save_usage`, reset, `load_usage` → counters restored.
  - `test_budget_guard_blocks_call`: `GEMINI_BUDGET_USD=0.000001`, `USAGE["cost_usd"]=1` → `BudgetExceeded`, fake client not called.
  - `test_blocked_response_returns_empty`: fake response with `text=None` → returns `""`, `USAGE["blocked"] == 1`.
  - `test_markdown_letter_scored`: `score.extract_answer("**B**", "first_letter") == "B"` and `extract_answer("Answer: B", "first_letter") == "B"` (if it fails, fix the scorer, not the wrapper).
- [ ] **Step 2: Run** `pytest tests/test_gemini_flash.py -v` — expected FAIL.
- [ ] **Step 3: Implement.** `types.GenerateContentConfig(temperature=0, thinking_config=types.ThinkingConfig(thinking_level=...))`. Default level: `"minimal"`; if the API rejects it, use `"low"` and change the module default (record the choice in the commit message). Retry 3× with 2/4/8 s backoff on transient errors only; no fixed inter-call sleep (paid tier).
- [ ] **Step 4: Smoke test (real API, ~$0.01):** `python -m performance_eval.run --model gemini_flash --condition smoke --limit 10` (no exclusion file exists yet) → 10 rows, every answer a single letter, `usage.json` shows `thoughts_tokens` small (< 50/call at minimal level). Delete the smoke CSV + sidecars afterwards.
- [ ] **Step 5: Run** `pytest tests -q` — all PASS. **Commit** `feat(models): Gemini 3.8 Flash MCQ wrapper with usage + budget guard`.

### Task 3: ~~Government-initiative tagging~~ — SUPERSEDED by D2 (labels done by hand-reading; files already in `performance_eval/data/`). Only remaining step: commit `item_labels.csv` + `excluded_items.csv`. The design below is kept for reference only; do not build it.

**Files:**
- Create: `performance_eval/gov_filter.py`, `performance_eval/data/excluded_items.csv` (output, committed)
- Test: `tests/test_performance_gov_filter.py`

**Interfaces:**
- Consumes: `models.gemini_flash.get_answer`; `load_milu_hindi_medicine`, `split_test_fewshot` (tags **test** items only, n = 489).
- Produces: `build_gov_prompt(item: dict) -> str` (D1 definition verbatim, ends "Reply with one word: yes or no."); `parse_yes_no(text: str) -> bool | None`; `keyword_hit(item: dict) -> list[str]` (matched keywords); CLI subcommands:
  - `tag` → `outputs/gov_filter/tags.csv` (`item_id, tagger, keywords, raw`), resumable.
  - `sheet` → `outputs/gov_filter/handcheck.csv` (`item_id, question_with_options, tagger, keywords, exclude, note`) with flagged items + 40 random unflagged (seed 42), shuffled so flagged status is not in row order; `exclude` column left blank for Grace.
  - `finalize` → reads the filled sheet; writes `performance_eval/data/excluded_items.csv` (`item_id, reason`) for rows with `exclude=yes`; errors if any `exclude` cell is blank or not yes/no; prints the miss-check result (unflagged items Grace marked yes) and exits non-zero if ≥ 2 (D2 escalation).
- Keyword list (starting set; extend during hand-check, record additions in the commit): `योजना, मिशन, अभियान, कार्यक्रम, सरकार, मंत्रालय, नीति, पोर्टल, आयुष्मान, प्रधानमंत्री, राष्ट्रीय, लॉन्च, नियुक्त, सर्वेक्षण, रिपोर्ट`, plus case-insensitive `scheme, mission, yojana, abhiyan, portal, ministry, launched`.

- [ ] **Step 1: Write failing tests**
  - `test_prompt_ignores_answer`: the same item with `answer_letter="A"` and with `answer_letter="C"` gives byte-identical prompts, and the prompt contains the question text.
  - `test_parse_yes_no`: `"Yes"`→True, `"no."`→False, `"हाँ"`→True, `"maybe"`→None.
  - `test_keyword_hit`: question containing `आयुष्मान भारत योजना` → includes `योजना` and `आयुष्मान`.
  - `test_finalize_rejects_blank_cells` and `test_finalize_flags_misses` (2 unflagged rows marked yes → non-zero exit).
- [ ] **Step 2: Run** — expected FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests -q` — PASS.
- [ ] **Step 5: Real run** `tag` (489 calls, ~$0.10) → `sheet` → **Grace fills `exclude`** → `finalize`. Record in the commit message: n flagged, n excluded, miss-check result. Current keyword pre-scan on the C0 outputs hits 64 / 489, so expect roughly 40–80 exclusions.
- [ ] **Step 6: Commit** `feat(performance_eval): government-initiative exclusion list (n=…)`, including `excluded_gov_items.csv` and the filled hand-check sheet.

### Task 4: Hindi → English translation

**Files:**
- Create: `performance_eval/translate.py`
- Modify: `performance_eval/data_milu.py` (add loader)
- Test: `tests/test_performance_translate.py`

**Interfaces:**
- Produces (primary, D5): `translate_google(items: list[dict], api_key: str) -> list[dict]` — POST to `https://translation.googleapis.com/language/translate/v2` with `q=[question, *options]` per item, `source="hi"`, `target="en"`, `format="text"`; key from `.env` `GOOGLE_TRANSLATE_API_KEY`; output `outputs/milu_en/translations_google.jsonl`. Secondary output `translations_gemini.jsonl`; `agreement(primary, secondary) -> dict[item_id, float]` (chrF via `sacrebleu`). `run.py --translations` accepts either file; condition labels `c0_en_google` / `c0_en_gemini`.
- Produces (secondary): `build_translate_prompt(item: dict) -> str` (asks for JSON `{"question": str, "options": {"A": str, ...}}`, keep numbering/statement lists, keep drug/medicine names, transliterate untranslatable AYUSH terms); `parse_translation(text: str, n_options: int) -> tuple[dict | None, list[str]]` (translation, list of flag strings); CLI `translate` → `outputs/milu_en/translations.jsonl` (one record per item: `item_id, question, options, flags, raw`), resumable; CLI `sheet` → `outputs/milu_en/handcheck.csv` (30 random + all flagged; columns `item_id, hindi, english, ok, note`); `data_milu.load_english_items(translations_path: Path, hindi_items: list[dict], drop_ids: set[str] = frozenset()) -> list[dict]` returning items with **same `id`, `answer_letter`, `n_options`**, English `question`, `options`, `question_with_options` (built with `_format_qwo`).
- Flags: `option_count_mismatch`, `empty_option`, `devanagari_left` (> 20% Devanagari chars in any field), `numbering_lost` (Hindi question has `1.`/`2.` lines and English does not), `bad_json`.

- [ ] **Step 1: Write failing tests** — one per flag above, plus `test_load_english_items_keeps_id_and_gold` and `test_load_english_items_missing_translation_raises`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests -q` — PASS.
- [ ] **Step 5: Real run** `translate` (489 calls, ~$0.25) → `sheet` → **Grace (or a Hindi reader) fills `ok`**. Items with `ok=no` go to `performance_eval/data/bad_translation_items.csv` (`item_id, reason`), which the diagnostic drops from both languages (D5).
- [ ] **Step 6: Commit** `feat(performance_eval): Hindi→English translations of MILU test items`, including the translations file, sheet and bad-translation list.

### Task 5: English prompt variant and `--language` in the runner

**Files:**
- Modify: `performance_eval/prompts.py`, `performance_eval/run.py`
- Test: `tests/test_performance_prompts.py`, `tests/test_performance_run.py`

**Interfaces:**
- Produces: `build_prompt(question_with_options, n_options, use_cot=False, exemplars=None, language: str = "hi") -> str`; `language="en"` replaces "in Hindi" with "in English" in both instructions (D6). `run.py --language {hi,en}` (default `hi`) and `--translations PATH` (required when `en`); output file `{model}_{condition}_raw.csv` under `--output-dir`, default `outputs/milu_hi_med` for hi and `outputs/milu_en_med` for en; sidecar gains `"language"` and `"translations_sha256"`.

- [ ] **Step 1: Write failing tests**
  - `test_hindi_prompt_unchanged`: `build_prompt(q, 4)` equals a stored copy of today's output (byte-identical).
  - `test_english_prompt_says_english`: `"in English"` in prompt and `"in Hindi"` not in it, for both `use_cot` values.
  - `test_run_en_requires_translations`: `main(["--model","himed","--condition","c0_baseline","--language","en"])` → `SystemExit`.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests -q` — PASS.
- [ ] **Step 5: Commit** `feat(performance_eval): English-language MCQ condition`.

### Task 6: Diagnostic report

**Files:**
- Create: `performance_eval/diagnostic.py`
- Test: `tests/test_performance_diagnostic.py`

**Interfaces:**
- Consumes: `analyze.accuracy_summary`, `analyze.compare_conditions`, `analyze.score_raw_csv`, `exclusions.load_exclusions`.
- Produces: CLI `python -m performance_eval.diagnostic --cell MODEL:LANG:RAW_CSV ... --exclude-file F [--exclude-file F2 ...] --family-size 5 --out performance_eval/analysis_outputs/step1_diagnostic.md`. Report contains: (a) n items before/after each exclusion file; (b) a 3×2 accuracy table (95% Wilson CI, NONE rate); (c) the five D7 comparisons (Δ pp, flips, McNemar p, Bonferroni p, significant?); (d) the D8 verdict line computed from (c); (e) sensitivity: the same table on the 489 unfiltered items; (f) Gemini cost from the usage sidecars.
- Errors if the cells do not share an identical item-id set after exclusions.

- [ ] **Step 1: Write failing tests**
  - `test_cells_must_share_items`: two cells with different ids → `SystemExit`.
  - `test_verdict_open_models_bottleneck`: synthetic scored rows where gemini beats both open models by 15 pp with p < 0.01 → verdict contains `open Indic models`.
  - `test_verdict_questions`: all cells equal → verdict contains `questions`.
  - `test_bonferroni_m5`: p = 0.02 → adjusted 0.10, not significant.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests -q` — PASS.
- [ ] **Step 5: Commit** `feat(performance_eval): step-1 diagnostic report`.

### Task 7: Runs and write-up

No new code; operational checklist. Exclusion files: `performance_eval/data/excluded_items.csv` and `performance_eval/data/bad_translation_items.csv`.

- [ ] **Step 1: Gemini Hindi** — `python -m performance_eval.run --model gemini_flash --condition c0_baseline --exclude-file performance_eval/data/excluded_items.csv`. Expected: n = 489 − n_excluded rows; usage cost < $0.20.
- [ ] **Step 2: Gemini English** — same plus `--language en --translations performance_eval/outputs/milu_en/translations.jsonl`.
- [ ] **Step 3: Open models English on PACE** — add `--language`/`--translations` passthrough to `scripts/performance_milu_run.sh`; `sbatch` HiMed and Nanda with `--language en --condition c0_baseline` (HF caches under `~/scratch`). Copy raw CSVs back to `outputs/milu_en_med/`.
- [ ] **Step 4: Existing Hindi runs** — reuse `himed_c0_baseline_raw.csv` and `nanda_c0_baseline_raw.csv` unchanged (D3); exclusions are applied at analysis time.
- [ ] **Step 5: Report** — run `performance_eval.diagnostic` with the six cells and both exclusion files. Check: identical n in all cells; total Gemini cost in the report ≤ $2.
- [ ] **Step 6 (optional; does not fit the $2 cap, needs more budget): Gemini with thinking on** — `--condition c2_thinking` with `thinking_level` raised. Reported as a separate row, **not** part of the D7 family.
- [ ] **Step 7: Update** `2026-11-15-mcq-performance-techniques-design.md` "Non-goals" to note fine-tuning is now in scope if D8 points to the models.
- [ ] **Step 8: Commit** outputs + `step1_diagnostic.md`: `results(performance_eval): step-1 diagnostic`.

### Task 8 (bonus): Translate-then-retrieve RAG

Tests the cheaper fix from *Language Bias in Multilingual RAG* (translate the query to English before retrieval) with no reranker.

**Files:**
- Modify: `performance_eval/rag/retrieve.py`
- Test: `tests/test_rag_retrieve.py` (or the existing RAG retrieve test file)

**Interfaces:**
- Consumes: `data_milu.load_english_items`.
- Produces: `retrieve.py --query-translations PATH`: retrieval queries are built by `build_query` from the **English** item; passages are written under the **same `item_id`**; passages-file metadata gains `"query_language": "en"`. The answering prompt stays Hindi (`--language hi`).

- [ ] **Step 1: Write failing test** `test_query_translations_swap_query_only`: with a fake encoder that records its inputs, the encoded queries are the English texts and the output keys are the Hindi item ids.
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** `pytest tests -q` — PASS.
- [ ] **Step 5: Run on PACE** — `scripts/rag_retrieve.sh` with `--query-translations`, then both open models with `--passages-file` and `--condition rag_k5_enquery`.
- [ ] **Step 6: Compare** with `performance_eval.analyze --exclude-file ... --ladder rag_k5:first_letter rag_k5_enquery:first_letter` per model (this is a new pre-registered comparison: m = 1 per model). Fill the existing retrieval spot-check sheet for 30 items with English queries to see whether retrieved passages got more relevant.
- [ ] **Step 7: Commit** `feat(rag): English-query retrieval (translate-then-retrieve)`.

## Budget summary

| Item | Calls | Est. cost |
|---|---|---|
| Smoke test | 10 | < $0.01 |
| Gov tagging | 489 | ~$0.10 |
| Translation | 489 | ~$0.25 |
| Gemini Hindi + English answers | ~850 | ~$0.15 |
| **Required total** | | **≈ $0.50** |
| Optional thinking-on run | ~425 | $2–4 |
