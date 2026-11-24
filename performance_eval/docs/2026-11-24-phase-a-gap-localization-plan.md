# Phase A: Localize HiMed's Hindi→English Gap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out *which part* of a Hindi MCQ costs HiMed its ~21 pp (63.4% Hindi vs 83.5–84.8% English on the 388 kept MILU items): the answer-option terms, the question stem, or reasoning in Hindi. The answer picks the Phase B training design.

**Architecture:** Three new inference-only conditions reuse the existing runner, translations and scorer: two mixed-language items (Hindi stem + English options, and the reverse) and English chain-of-thought on the untouched Hindi item. A pre-registered report (`gap_report.py`) compares each to its Hindi control and applies a decision rule fixed below. Two CPU-only analyses run on outputs we already have: a labeled table of the items that flip between Hindi and English, and tokenizer fertility.

**Tech Stack:** Python, existing `performance_eval` (run / score / analyze / diagnostic), `transformers` tokenizers (CPU), PACE Slurm for the 5 GPU jobs.

**Spec:** this document (Decisions). Context: `2026-11-22-step1-diagnostic-plan.md` (D5 two-translator rule, exclusions), `analysis_outputs/step1_diagnostic_{google,gemini}.md` (results that motivate this).

**Branch / worktree:** `perf/step1-diagnostic`, worktree `.worktrees/step1`. Push to that branch so the user can pull on PACE.

## Decisions (the spec)

- **A1 — Items.** The same 388 items as step 1: `--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv` on every run and report.
- **A2 — Conditions (HiMed only; Nanda is optional and outside the rule).**

  | label (condition) | stem | options | reasoning | scored | control |
  |---|---|---|---|---|---|
  | `c0_baseline` (exists) | hi | hi | (model's own, Hindi) | first_letter | — |
  | `c0_en_{T}` (exists) | en | en | — | first_letter | — |
  | `c0_mix_qhi_oen_{T}` **new** | hi | en | — | first_letter | `c0_baseline` |
  | `c0_mix_qen_ohi_{T}` **new** | en | hi | — | first_letter | `c0_baseline` |
  | `c2_cot` (exists, 2048 tok) | hi | hi | Hindi CoT | after_marker | — |
  | `c2_cot_en` **new** | hi | hi | English CoT | after_marker | `c2_cot` |

  `{T}` ∈ {google, gemini}; mixed runs are done for both translations (4 jobs) + 1 `c2_cot_en` job = **5 GPU jobs**. `c2_cot_en` uses `--max-new-tokens 2048`, same as `c2_cot`.
  **Amendment A2a (2026-11-24, before any full CoT run):** the 3-item smoke test with HiMed's default system
  message ("…हिंदी में उत्तर दें", answer in Hindi) reasoned in Hindi on 3/3 items despite the English instruction.
  Comparison 3 is therefore run as `c2_cot_en_sysneutral` vs a new control `c2_cot_sysneutral`, both with
  `--system-prompt neutral` ("You are a helpful medical teacher. Follow the given instructions."), so the two
  arms differ only in the reasoning-language instruction. The old `c2_cot` is not used in the rule. 6 GPU jobs.
- **A3 — Prompt wording.** The `--language` value sets the phrase that replaces "in Hindi" in the *question description* only:
  `hi` → "in Hindi"; `en` → "in English"; `hi_q_en_opts` → "in Hindi (answer options in English)"; `en_q_hi_opts` → "in English (answer options in Hindi)".
  `--reasoning-language en` (CoT only) changes only the reasoning phrases: "Think step by step in English." and closing "Reasoning (step by step, in English):"; the question description stays "in Hindi". Existing prompts for `hi`/`en` must stay byte-identical.
- **A4 — Pre-registered comparisons** (per translation T; exact McNemar, two-sided, Bonferroni **m = 3**, α = 0.05):
  1. `c0_mix_qhi_oen_T` vs `c0_baseline` (English options)
  2. `c0_mix_qen_ohi_T` vs `c0_baseline` (English stem)
  3. `c2_cot_en` vs `c2_cot` (English reasoning)

  Gap `G = acc(c0_en_T) − acc(c0_baseline)`. Recovered share `R = Δ / G` (Δ of that comparison).
- **A5 — Decision rule (fixed now, applied mechanically).** A part "explains the gap" when its comparison is significant, Δ > 0 and R ≥ 0.50. Comparison 3 also needs **compliance**: ≥ 80% of `c2_cot_en` responses have reasoning whose letters are ≥ 50% Latin script; otherwise report "English-reasoning instruction not followed; comparison 3 inconclusive" and treat it as not firing. If G < 0.05 the report says "no gap to localize" and applies no rule. Messages and the Phase B direction each points to:
  - 1 fires → "Hindi option terms" → Phase B: bilingual term-level data (Hindi term ↔ English term) in SFT.
  - 2 fires → "Hindi question stem" → Phase B: parallel bilingual SFT on full questions.
  - 3 fires → "reasoning language" → Phase B: SFT on Hindi question → English reasoning → answer.
  - none fires → "no single part explains it; only the fully English item recovers it" → Phase B: cross-lingual self-distillation (English-item teacher → Hindi-item student).
  Several may fire; list all. A conclusion counts only if the google and gemini reports agree (step-1 D5); comparison 3 uses no translation and is identical in both.
- **A6 — Flip table.** Items whose correctness differs between `himed c0_baseline` and `himed c0_en_google` (step 1: 102 English-only-correct, 19 Hindi-only-correct). Each gets ONE primary category from this codebook:
  `terminology` (Hindi medical term rare/Sanskritized/ambiguous; English uses a common term) · `question_form` (negation, "which is NOT", multi-statement, assertion–reason) · `traditional_medicine` (Ayurveda/Yoga/Unani/Indian-specific) · `translation_artifact` (translation changed meaning, dropped or added a cue) · `item_noise` (gold looks wrong or item ambiguous in both languages) · `format` (no parseable answer in one language) · `other`.
  Rater: the implementing agent, single LLM rater, no human check — the summary must say so. Also report how many of those items flip the same way under the gemini translation.
- **A7 — Tokenizer fertility.** fertility = tokens / whitespace-separated words (`add_special_tokens=False`), per item on `question_with_options`, for the HiMed (`FreedomIntelligence/HiMed-8B`) and Nanda (`MBZUAI/Llama-3-Nanda-10B-Chat`, `trust_remote_code=True`) tokenizers, Hindi and English (google). Report: mean fertility per tokenizer × language; HiMed Hindi accuracy by HiMed-Hindi-fertility quartile; and, for comparison, accuracy by Hindi word-count quartile (length confound). Descriptive only, no significance test.
- **A8 — No new dependencies.** `transformers` 5.5.4 and `scipy` 1.18.1 already installed locally.

## Global Constraints

- PACE: all outputs/caches under `/home/hice1/gchiu6/scratch`; NVIDIA GPUs only; commit as `grchiu6@gmail.com` (repo git config already set).
- Every run and report uses the two exclusion files (A1). Paired comparisons require identical item sets (`check_same_items` / `check_item_ids` must pass).
- No edits to the Hindi/English prompt text (A3); `tests/test_performance_prompts.py` existing tests must stay green unchanged.
- Test command for every task: `python3 -m pytest tests/test_performance_*.py tests/test_rag_*.py tests/test_gemini_flash.py -q` (218 pass at plan time).

## Review Focus

1. Mixed item where stem/options come from different sources still renders exactly as `_format_qwo(question, options)` — with `("hi","hi")` it must equal the original Hindi `question_with_options` byte-for-byte (pinned in Task 1).
2. `--reasoning-language en` without `--use-cot`, or mixed language with `--use-cot` / `--use-fewshot` / `--shuffle-options` → argparse error, not a silent run (Tasks 1–2).
3. Resuming an *old* run (sidecar lacks the new key) must not be refused: add `reasoning_language` to the sidecar only when it is set (Task 2).
4. HiMed responses with `## Thinking … ## Response` and CoT responses ending `Answer: X`: the Latin-share check must look only at the reasoning part, and return `None` (not crash) on a response with no letters (Task 3).
5. A cell file missing or G < 0.05 → report says so; no division by zero, no rule applied (Task 3).

---

### Task 1: Mixed-language items (`hi_q_en_opts`, `en_q_hi_opts`)

**Files:**
- Modify: `performance_eval/data_milu.py` (add `load_mixed_items`; `load_english_items` becomes a wrapper)
- Modify: `performance_eval/prompts.py` (`_LANGUAGE_NAMES` → question-description phrases, A3)
- Modify: `performance_eval/run.py` (`--language` choices; validation; item loading; default output dir: anything but `hi` → `milu_en_med`)
- Test: `tests/test_performance_data_milu.py`, `tests/test_performance_prompts.py`, `tests/test_performance_run.py`

**Interfaces:**
- Produces: `load_mixed_items(translations_path, hindi_items: list[dict], question_lang: str, options_lang: str, drop_ids=frozenset()) -> list[dict]` — same validation as `load_english_items` (missing / hard-flag / option-key mismatch raise ValueError); output keeps `id`, `answer_letter`, `n_options`; English parts pass through `_clean`, Hindi parts are the item's own `question`/`options`.
- Produces: `LANGUAGES = ("hi", "en", "hi_q_en_opts", "en_q_hi_opts")` in `prompts.py`; `run.py --language` uses it.

- [ ] **Step 1: Failing tests**
  - `test_mixed_hi_hi_is_identity`: `load_mixed_items(path, items, "hi", "hi")[i]["question_with_options"] == items[i]["question_with_options"]`.
  - `test_mixed_hi_question_en_options`: stem line equals the Hindi question; option lines equal the cleaned English options; `id`/`answer_letter` unchanged.
  - `test_mixed_en_question_hi_options`: the reverse.
  - `test_load_english_items_unchanged`: existing English tests still pass (wrapper calls `load_mixed_items(..., "en", "en", ...)`).
  - Prompts: `build_prompt(q, 4, language="hi_q_en_opts")` contains `"medical question in Hindi (answer options in English)"`; `en_q_hi_opts` contains `"in English (answer options in Hindi)"`; `build_prompt(q, 4, use_cot=True, language="hi_q_en_opts")` raises ValueError.
  - Run: `--language hi_q_en_opts` without `--translations` → SystemExit; with `--use-cot`, `--use-fewshot` or `--shuffle-options` → SystemExit.
- [ ] **Step 2:** Run the three test files; expect the new tests to FAIL (missing function / invalid choice).
- [ ] **Step 3:** Implement. In `run.py`, `test = load_mixed_items(args.translations, test, q, o)` where `(q, o)` come from a dict `{"en": ("en","en"), "hi_q_en_opts": ("hi","en"), "en_q_hi_opts": ("en","hi")}`; keep the existing English refusal messages, extended to the mixed values. The sidecar already records `language` and `translations_sha256`.
- [ ] **Step 4:** Run the full test command; expect all pass.
- [ ] **Step 5:** Commit `feat(performance_eval): mixed-language MCQ items (Hindi stem/English options and reverse)`.

### Task 2: English chain-of-thought on Hindi items (`--reasoning-language`)

**Files:**
- Modify: `performance_eval/prompts.py` (`build_prompt(..., reasoning_language: str | None = None)`)
- Modify: `performance_eval/run.py` (`--reasoning-language {hi,en}`, pass-through in `run_condition`, sidecar)
- Test: `tests/test_performance_prompts.py`, `tests/test_performance_run.py`

**Interfaces:**
- Produces: `build_prompt(question_with_options, n_options, use_cot=False, exemplars=None, passages=None, language="hi", reasoning_language=None) -> str`. `None` = same as before (reasoning in the item language). `run_condition(..., reasoning_language=None)`.

- [ ] **Step 1: Failing tests**
  - `test_cot_english_reasoning_on_hindi`: `p = build_prompt(q, 4, use_cot=True, reasoning_language="en")` → `"medical question in Hindi" in p`, `"Think step by step in English." in p`, `p.endswith("Reasoning (step by step, in English):")`, `"step by step in Hindi" not in p`.
  - `test_cot_reasoning_language_none_unchanged`: `build_prompt(q, 4, use_cot=True) == build_prompt(q, 4, use_cot=True, reasoning_language=None)` and equals the string the current code produces (snapshot it from the current code before editing).
  - `test_reasoning_language_requires_cot`: `build_prompt(q, 4, reasoning_language="en")` raises ValueError; CLI `--reasoning-language en` without `--use-cot` → SystemExit.
  - `test_sidecar_reasoning_language_only_when_set`: the config dict built for a run without the flag has no `reasoning_language` key; with it, `"reasoning_language": "en"`. (Factor the config-dict construction into a small helper if needed to test it.)
- [ ] **Step 2:** Run; expect FAIL.
- [ ] **Step 3:** Implement: replace only the reasoning phrases ("Think step by step in Hindi" and the closing) with the reasoning language; the question description follows `language` (A3).
- [ ] **Step 4:** Full test command; all pass.
- [ ] **Step 5:** Commit `feat(performance_eval): --reasoning-language for English CoT on Hindi items`.

### Task 3: Gap-localization report (`gap_report.py`)

**Files:**
- Create: `performance_eval/gap_report.py`
- Test: `tests/test_performance_gap_report.py`

**Interfaces:**
- Consumes: `score_raw_csv(raw_path, mode, scored_path)`, `accuracy_summary`, `compare_conditions` (analyze.py); `load_exclusions`; `check_same_items` (diagnostic.py).
- Produces:
  - `reasoning_latin_share(response: str) -> float | None` — reasoning part = text before `## Response` if present, else before the last `Answer:`; share = Latin letters / (Latin + Devanagari `ऀ-ॿ` letters); `None` if no letters.
  - `COMPARISONS = [("q_hi_o_en", "hi"), ("q_en_o_hi", "hi"), ("cot_en", "cot_hi")]` (target, control).
  - `run_comparisons(cells: dict[str, list[dict]], family_size=3) -> list[dict]` — like `diagnostic.run_comparisons`, plus `recovered` (= Δ / G, or None) per comparison.
  - `localize(cells, comps, compliance: float | None) -> str` — rule A5, exact messages.
  - CLI: `python -m performance_eval.gap_report --cell NAME:MODE:RAW_CSV ... --exclude-file ... --en-tag T --out PATH`; NAME ∈ {hi, en, q_hi_o_en, q_en_o_hi, cot_hi, cot_en}; missing cells → that comparison "not run".

- [ ] **Step 1: Failing tests**
  - `test_latin_share`: `"## Thinking\nThe liver stores glycogen.\n## Response\nB."` → 1.0; `"यकृत ग्लाइकोजन\nAnswer: B"` → 0.0; `"abc अआ\nAnswer: A"` → 0.6; `"Answer: B"` → None (nothing before the last `Answer:`).
  - `test_recovered_share`: cells hi 60/100, en 80/100, q_hi_o_en 75/100 → comparison 1 `recovered == pytest.approx(0.75)`.
  - `test_rule_options_fire`: with the cells above made significant (use n=200, hi 120, en 170, q_hi_o_en 160, q_en_o_hi 122, cot_hi 125, cot_en 126) → message contains "Hindi option terms" and not "Hindi question stem".
  - `test_rule_none_fires`: mixed cells ≈ hi → message contains "only the fully English item recovers it".
  - `test_cot_noncompliant_inconclusive`: cot_en much better but `compliance=0.5` → contains "inconclusive" and not "reasoning language".
  - `test_no_gap`: en − hi = 0.02 → "no gap to localize".
  - `test_missing_cell_not_run`: drop `cot_en` → comparison 3 skipped, report row says "not run".
- [ ] **Step 2:** Run; FAIL (module missing).
- [ ] **Step 3:** Implement. Report sections: items/exclusions; accuracy table (Wilson CI, none_rate); comparisons table (Δ pp, flips, p, p Bonferroni m=3, significant, recovered %); compliance line (`c2_cot_en`: X% responses mostly-English reasoning; `c2_cot`: Y% for contrast); verdict (A5). Score `cot_*` cells with `after_marker`, others `first_letter` (mode comes from the cell spec).
- [ ] **Step 4:** Full test command; all pass.
- [ ] **Step 5:** Commit `feat(performance_eval): Phase A gap-localization report`.

### Task 4: Flip table, labels and summary

**Files:**
- Create: `performance_eval/flips.py`
- Test: `tests/test_performance_flips.py`
- Output (tracked): `performance_eval/analysis_outputs/phase_a/flips_himed_google.csv`, `performance_eval/analysis_outputs/phase_a/flips_summary.md`

**Interfaces:**
- Produces: `CATEGORIES` (A6 codebook tuple); `flip_rows(hi_scored, en_scored, alt_en_scored=None) -> list[dict]` with columns `item_id, direction (en_only_correct|hi_only_correct), gold, hi_pred, en_pred, same_flip_alt (bool|"" if no alt), hindi_item, english_item, category, note` (category/note empty); `summarize(rows) -> str` (direction × category counts; refuses rows with empty or unknown category: ValueError naming the item_ids).
- CLI: `python -m performance_eval.flips export --hi ... --en ... --alt-en ... --exclude-file ... --out CSV` and `python -m performance_eval.flips summarize --labeled CSV --out MD`.

- [ ] **Step 1: Failing tests**: `test_flip_rows_directions` (3 items: right/wrong, wrong/right, right/right → 2 rows with correct directions); `test_same_flip_alt`; `test_summarize_rejects_unknown_category`; `test_summarize_counts`.
- [ ] **Step 2:** Run; FAIL.
- [ ] **Step 3:** Implement; then export from the existing laptop outputs (`outputs/milu_hi_med/himed_c0_baseline_scored.csv`, `outputs/milu_en_med/himed_c0_en_google_scored.csv`, alt = `..._gemini_scored.csv`, both exclusion files). Expected: 121 rows (102 + 19).
- [ ] **Step 4: Label.** Read every row (Hindi and English item, gold, both predictions) and fill `category` (one value from `CATEGORIES`) and a short `note`. Then run `summarize`. The summary's first line: "Labels: one LLM rater (Claude), no human check." Ruling note for the ledger if a category boundary is unclear.
- [ ] **Step 5:** Full test command; commit `results(performance_eval): labeled HiMed Hindi/English flip table`.

### Task 5: Tokenizer fertility

**Files:**
- Create: `performance_eval/tokenizer_stats.py`
- Test: `tests/test_performance_tokenizer_stats.py` (fake tokenizer object with `.encode(text, add_special_tokens=False)`; no downloads in tests)
- Output (tracked): `performance_eval/analysis_outputs/phase_a/tokenizer_fertility.md`

**Interfaces:**
- Produces: `fertility(tokenizer, text: str) -> float` (0 words → ValueError); `quartile_table(values: list[float], correct: list[bool]) -> list[dict]` (`quartile, lo, hi, n, accuracy`); CLI `--hi-scored --en-raw --exclude-file ... --out`.

- [ ] **Step 1: Failing tests**: `test_fertility` (fake tokenizer that splits on characters: `"ab cd"` → 2.0); `test_fertility_empty_raises`; `test_quartile_table` (8 values, known accuracies per quartile, n sums to 8).
- [ ] **Step 2:** Run; FAIL.
- [ ] **Step 3:** Implement and run on the laptop (downloads two tokenizers only; `HF_TOKEN` from `.env` if Nanda needs auth). Report per A7.
- [ ] **Step 4:** Full test command; commit `results(performance_eval): tokenizer fertility for HiMed/Nanda`.

### Task 6: PACE runbook and push

**Files:**
- Create: `performance_eval/docs/phase_a_pace_runbook.md`
- Modify: `scripts/performance_milu_run.sh` header (add the 5 Phase A examples)

- [ ] **Step 1:** Runbook with: `git pull`; the 5 sbatch commands (below); `scp` of `himed_c0_mix_*_raw.csv*` and `himed_c2_cot_en_raw.csv*` into laptop `performance_eval/outputs/phase_a/` (plus copy the existing `himed_c0_baseline`, `himed_c2_cot`, `himed_c0_en_*` raws there or point cells at their current paths); the two `gap_report` commands (T = google, gemini) writing `analysis_outputs/phase_a/gap_report_$T.md`; optional Nanda variants.

  ```bash
  EX="--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv"
  TR=performance_eval/data/milu_en
  for T in google gemini; do
    sbatch scripts/performance_milu_run.sh himed c0_mix_qhi_oen_$T --language hi_q_en_opts $EX --translations $TR/translations_$T.jsonl
    sbatch scripts/performance_milu_run.sh himed c0_mix_qen_ohi_$T --language en_q_hi_opts $EX --translations $TR/translations_$T.jsonl
  done
  sbatch scripts/performance_milu_run.sh himed c2_cot_en --use-cot --reasoning-language en --max-new-tokens 2048 $EX
  ```
  Note in the runbook: `c2_cot_en` is slower (long reasoning); the 4 h wall time held for `c2_cot`.
- [ ] **Step 2:** Add `test_phase_a_invocations_parse` to `tests/test_performance_run.py`: each of the three new flag sets (`hi_q_en_opts`, `en_q_hi_opts`, `--use-cot --reasoning-language en`) runs `main([...])` end-to-end on 2 items with the existing stub model/data fixtures and writes a 2-row CSV whose sidecar records the right `language` / `reasoning_language`.
- [ ] **Step 3:** Full test command; commit `docs(performance_eval): Phase A PACE runbook`; `git push origin perf/step1-diagnostic`.

### Task 7 (after the PACE jobs; not part of the implementation session)

Copy outputs, run both `gap_report`s, read the A5 verdicts against each other (agree → conclusion; disagree → say so), write `analysis_outputs/phase_a/phase_a_summary.md` (verdict + flip summary + fertility headline + Phase B direction), update project memory, push.
