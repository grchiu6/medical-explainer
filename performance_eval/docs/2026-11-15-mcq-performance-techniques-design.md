# Design: Which techniques improve model answer-correctness on Hindi medical MCQs

Status: APPROVED (2026-11-15) — compute confirmed as PACE cluster
Date: 2026-11-15
Owner: Grace
Home: `performance_eval/` (sibling to `literacy_eval/`; this track is about **answer correctness**, not literacy adaptation)

## Intent and why

The literacy-eval track found accuracy pinned at ~0% for every model on HiMed-Trad. Investigation showed that number is mostly **not** a knowledge failure: it mixes generation artifacts (echo, repetition loops, refusals, truncation), noisy machine-translated reference answers, and a genuine but confounded domain gap (HiMed-Trad tests specialized Ayurveda/Siddha/Unani knowledge the models likely never trained on). Below-random scores are the tell that the *measurement*, not the model's knowledge, drives the floor.

This track asks a cleaner, prior question: **can these models answer a Hindi medical question correctly at all, and which techniques raise that?** We move to a domain the models plausibly know (modern + general medical content, native Hindi) and an instrument that gives an unambiguous right/wrong signal (MCQ with an answer key, deterministic scoring — no LLM judge, which also respects the no-paid-judge-budget constraint).

## Goals

- Measure baseline MCQ accuracy for two models on a better-matched Hindi medical dataset.
- Test, **one technique at a time**, whether each of a set of inference-time techniques raises accuracy, with clean attribution of any gain to a specific technique.
- Produce a reproducible results table + figure a reader can trust: per model × condition accuracy, with a paired significance test between consecutive conditions.

## Non-goals / anti-goals

- **Not** literacy-level adaptation, readability, or register — that stays in `literacy_eval/`.
- **Not** open-ended generation scoring (no judge in this track). Open-ended revisit is deferred.
- **Not** fine-tuning or training. Inference-time techniques only.
- **Anti-goal:** bundling multiple techniques into one run such that a gain can't be attributed. If we can't say *which* technique helped, the experiment has failed its purpose.
- **Anti-goal:** any technique that leaks the answer into the prompt (esp. RAG retrieving from the answer key / explanation field). A leaked result is worse than no result.

## Decisions locked with the user

1. **Instrument: MCQ accuracy** (deterministic, no judge).
2. **Attribution: one-at-a-time, additive (forward-stepwise).** Add exactly one technique on top of the best-so-far config, measure, keep it only if it helps beyond noise.
3. **Models: HiMed-8B and Nanda** — the only two of the five that produce coherent output (HiMed-8B 58/58/60 usable, Nanda 47/57/54; the other three fail for model-intrinsic reasons).
4. **Location:** new code in `performance_eval/`, sibling to `literacy_eval/`.

## Builder-check resolutions (2026-11-15) — my calls, veto welcome

A fresh reviewer read this doc as a builder. Five decisions that materially affect trustworthiness, settled toward the defensible default for a paired-attribution study:

1. **Deterministic decoding for C0–C3 (not just C4).** Both wrappers default to `do_sample=True`, so re-running the *same* condition would flip items right↔wrong from RNG alone — which would pollute the McNemar discordant pairs the whole attribution rests on. **Resolution:** C0–C3 run **greedy** (`do_sample=False`), so every right↔wrong flip between rungs is caused by the one new technique, not sampling. This means `gen_kwargs` is passed from **C0 onward**, not only at C4. C4 is where sampling/temperature re-enters *as a tuned variable*, and any C4 sampling config is evaluated against a fixed seed + a same-condition re-run to estimate the noise floor.
2. **CoT gets enough tokens to emit its answer (correctness floor, not a C4 knob).** Nanda caps at `max_new_tokens=512`; a reasoning trace + the `उत्तर:/Answer:` marker can truncate before the marker, making C2 look falsely bad and getting it wrongly rejected before C4 could help. **Resolution:** raising `max_new_tokens` enough for the marker to appear is a **prerequisite applied at C2**, not reserved decoding tuning. C4 tunes *quality* knobs (repetition penalty, temperature, sampling), not the token budget needed to make the answer parseable.
3. **MILU answer encoding + option order.** (a) The loader must handle MILU storing the gold as an option *index* or *text* (not necessarily a letter) and synthesize A–J labels + the gold-letter mapping; confirmed at load. (b) **Resolution:** evaluate in MILU's **native option order** (standard MMLU practice), and add a **shuffled-subsample robustness check** (fixed seed) to measure position bias rather than reshuffling everything. Few-shot exemplars are letter-balanced so they don't teach a slot bias.
4. **NONE / unparseable rule (one rule, applied everywhere).** **Resolution:** reported accuracy = `correct / total` with NONE counted as wrong (stable denominator across conditions, required for paired comparison); NONE-rate reported **separately** alongside. A NONE→letter transition between rungs **does** count as a flip in the McNemar 2×2 — getting a model to commit to a correct letter is a real technique effect.
5. **Keep/reject rule.** **Resolution:** a rung is kept only if it shows a **positive accuracy delta AND** a two-sided McNemar **p < 0.05** after a **Bonferroni adjustment with a pre-registered family size m** (p_adj = min(1, p × m); default m = 3 per model, for the C2, C3 and C4 comparisons). Holm needs the full family of p-values at once, which a stepwise ladder never has, so the fixed-m Bonferroni bound is used instead. (If you pre-registered a one-sided test or different alpha, say so — it must be fixed before any rung is judged.)

## Dataset

**MILU** (ai4bharat), Hindi, subject = **"Health and Medicine"**.
- Verified from MILU paper (arXiv 2411.02538v2, Table 9): Hindi "Health and Medicine" = **499 items**; "Food Science" = 99 items. Native Hindi (only 115 of 15,450 Hindi items are translated). License **CC BY 4.0**.
- 499 items stands alone for this study (~8× the 60-item literacy sample). **Builder call:** start with "Health and Medicine" (499) only; treat Food Science (99) as optional extra, excluded by default to keep the domain focused.
- Caveat to state in writeup: MILU's health content *mixes* some traditional Ayurveda/Unani with modern/public-health — but it is native Hindi (not garbled MT), which is the key improvement over HiMed-Trad. If a strictly-modern comparison is later wanted, the translated **MMLU-Indic** Hindi clinical subjects (clinical_knowledge, professional_medicine, anatomy) are the fallback (not in scope now).

**Acquisition:** MILU is **gated** on HuggingFace (accept conditions + HF token). Loader lives in `performance_eval/data_milu.py` (or extend `data/download.py`), filters to `language == hi` and `subject == "Health and Medicine"`, and **freezes** two disjoint splits with a fixed seed:
- `test` = the evaluation set (target ~499, minus the exemplar pool).
- `fewshot_pool` = a small held-out pool (e.g. 10 items) exemplars are drawn from; **never** overlaps `test`.

**Option count / random baseline:** confirm the number of options per MILU item at load time (expected 4 → **25% chance baseline**; MMLU-derived format). The baseline number must appear on every results table so "better than chance" is visible.

## Condition ladder (forward-stepwise, one technique per rung)

Each rung = best-kept config so far + exactly one new technique. Measured on the same frozen `test` items (paired). A rung is **kept** only if it beats the prior best by a paired significance test (below); otherwise it is dropped and the next technique is tried on the prior best.

- **C0 — Baseline.** `build_mcq_prompt()` strict single-letter prompt, zero-shot, **greedy decoding** (`do_sample=False`, via `gen_kwargs` — see resolution 1). This is the reference point.
- **C1 — Prompt refinement.** Only if C0 output inspection shows a fixable prompt issue (e.g. wrong-language answers, not emitting a letter). Minimal, documented prompt changes. Skip if C0 is already clean.
- **C2 — Chain-of-thought.** "Reason step by step, then give the final answer as a single letter." Changes output from one token to a reasoning trace + letter.
  - **Scoring change this forces:** the current `extract_letter()` grabs the *first* standalone A–J letter, which breaks under CoT (reasoning mentions options early). CoT conditions need **answer-after-marker / last-letter extraction**: require the model to end with a marker (e.g. `उत्तर: X` / `Answer: X`) and parse the letter at/after the marker; fall back to the last standalone letter. Build this as a scorer mode, not a rewrite.
- **C3 — Few-shot.** Prepend 2–3 solved MCQ exemplars drawn from `fewshot_pool` (disjoint from `test`). Fixed exemplars across both models and all test items (control the variable).
- **C4 — Decoding tuning.** Only relevant once outputs are long (CoT on). Tune the per-model *quality* knobs (repetition penalty, temperature, sampling) to reduce degeneration — NOT the token budget, which is already set at C2 as a correctness floor (resolution 2). Any sampling config is judged against a fixed seed + a same-condition re-run noise floor (resolution 1). **Requires a code change:** `get_answer()` currently hardcodes decoding inside each model wrapper. **Builder call (recommend):** add an optional `gen_kwargs: dict | None = None` parameter to each model's `get_answer()` that overrides the defaults, so `performance_eval` can sweep decoding without forking the wrappers. Keep defaults unchanged so `literacy_eval` is untouched.

**RAG — deferred to week 2, separate sub-design.** Needs (a) an *independent* Hindi medical corpus to retrieve from (NOT the MILU answer/explanation), and (b) a retriever. Scoped only after C0–C4 report back. Its landmine (answer leakage) is why it is not bundled here.

Order rationale: cheapest/most-diagnostic first (prompt, CoT), then few-shot, then decoding (which only bites once CoT makes outputs long). "If appropriate upon initial findings" = each rung is attempted only when the prior result shows a failure mode it addresses.

## What "better" means (stats)

- Same frozen items at every condition → **paired** design.
- Per model × condition: accuracy + 95% CI (Wilson or bootstrap over items).
- Accuracy = `correct / total`, NONE counted as wrong (stable denominator across rungs); **NONE/unparseable rate reported separately** ("committed to a wrong letter" ≠ "gave no letter"). See resolution 4.
- To keep/reject a rung: **positive accuracy delta AND two-sided McNemar p < 0.05** (exact test, counts are modest) between the new condition and the prior best, Bonferroni-adjusted with a pre-registered family size m (default 3 per model). Report the p-value and net items flipped right↔wrong, not just the delta. A NONE→letter transition counts as a flip. See resolutions 4–5.
- Multiple comparisons: a small, pre-registered ladder (≤5 rungs × 2 models), Bonferroni within each model with pre-registered m (default 3: C2, C3, C4), passed as `analyze --family-size`. Holm needs the full family at once, which a stepwise ladder never has.

## Deliverables / outputs

- `performance_eval/outputs/milu_hi_med/<model>_<condition>_raw.csv` — raw generations.
- `..._scored.csv` — per-item parsed letter, correct/skip.
- `performance_eval/analysis_outputs/results.md` — main table: model × condition × accuracy × CI × Δ-vs-prior × McNemar p × NONE-rate, with the chance baseline annotated.
- Main figure: accuracy ladder per model across C0→C4 (shows which rung moved it).

## Proposed module layout (`performance_eval/`)

- `data_milu.py` — download (gated), filter Hindi + Health and Medicine, freeze `test` / `fewshot_pool` splits (seed).
- `prompts.py` — MCQ prompt builders per condition (baseline, CoT, few-shot wrapper). Reuse/import `models/base.py:build_mcq_prompt` where possible.
- `run.py` — for (model, condition) run generations → raw CSV. Imports existing `models/himed.py`, `models/nanda.py`; passes `gen_kwargs` for C4.
- `score.py` — MCQ scoring with two extraction modes (first-letter for C0/C1; after-marker/last-letter for CoT conditions). Reuse `score_mcq.py:extract_letter` as the first-letter mode.
- `analyze.py` — accuracy, CIs, McNemar between consecutive kept configs, build `results.md` + figure.
- `docs/` — this design, the RAG sub-design (later).
- `tests/` (or extend repo `tests/`) — extraction modes on hand-made CoT examples, split disjointness, McNemar on a known 2×2, random-baseline annotation.

## Reused vs new

- **Reuse:** `models/himed.py`, `models/nanda.py` (add optional `gen_kwargs`), `models/base.py:build_mcq_prompt`, `score_mcq.py:extract_letter`, the dated-outputs + resume conventions, `data/download.py` pattern.
- **New:** everything under `performance_eval/`.
- **Untouched:** all of `literacy_eval/` (and model-wrapper default behavior).

## Timeline

- **Week 1:** data loader + frozen splits → C0 baseline (both models) → C1–C4 as warranted → `results.md`. This answers "can they answer at all, and do the cheap techniques help?"
- **Week 2:** RAG sub-design + run, if C0–C4 suggest a knowledge gap retrieval could close.

## Risks / open items

- **MILU gate:** needs HF token + accept conditions on the cluster before any run.
- **Option count unconfirmed:** verify at load (expected 4 → 25% baseline).
- **Domain mix:** MILU health includes some traditional medicine; report it, don't hide it.
- **CoT extraction:** the biggest correctness risk in scoring; must be tested on hand-made CoT outputs before trusting any CoT condition's accuracy.
- **Compute:** PACE cluster (confirmed). `run.py` launches via sbatch, mirroring `scripts/literacy_*_run.sh`; outputs/caches under `~/scratch` per project convention.

## References

- MILU: Verma et al., *MILU: A Multi-task Indic Language Understanding Benchmark*, arXiv:2411.02538. Hugging Face: ai4bharat/MILU (CC BY 4.0).
- MMLU-Indic (fallback, modern clinical): sarvamai/mmlu-indic.
- Existing pipeline: `models/`, `score_mcq.py`, `evaluate.py`.
