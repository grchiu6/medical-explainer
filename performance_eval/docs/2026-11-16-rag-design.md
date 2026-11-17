# Design: Does retrieved medical reference text improve HiMed-8B on Hindi medical MCQs?

Status: APPROVED (2026-11-16)
Owner: Grace
Home: `performance_eval/` (RAG sub-design deferred by `2026-11-15-mcq-performance-techniques-design.md`)

## Intent and why

The MCQ track measures whether inference-time techniques raise HiMed-8B's answer correctness on the frozen MILU Hindi Health & Medicine test set (C0 baseline: **61.3%**, n=489, 4 options, chance 25%). About 190 items are wrong. This sub-study asks one question: **does giving HiMed authoritative medical reference text, retrieved per question, let it answer more of them correctly?** That is a test of whether part of the error is a *knowledge gap* that external knowledge can close.

The comparison is simple: HiMed with vs without retrieved context, on the same items, scored the same way.

## Goals

- Measure HiMed-8B accuracy with retrieval-augmented prompts (RAG) vs the C0 baseline, paired on the same 489 items.
- Separate "relevant knowledge helped" from "any extra context changes behaviour" with a random-passage control.
- Show *where* retrieval helps by breaking results down by question type.
- Leave an auditable record of exactly which passages each item received.

## Non-goals / anti-goals

- **Not** patient-friendly / literacy-adapted retrieval. Parked for later (see Future work).
- **Not** Nanda. Its C0 failure is 100% unparseable output (formatting), which retrieval can't fix. Revisit once approach 1 makes Nanda parseable.
- **Not** tuning retrieval hyper-parameters (k, chunk size, retriever) on the test set. One pre-registered configuration.
- **Not** stacking on CoT/few-shot yet. RAG branches off C0 because approach 1 (the C1–C4 ladder) has not finished. Stacking on the best ladder config is a later run.
- **Anti-goal: answer leakage.** The retrieval query and corpus must never contain the gold answer, the answer letter, or MILU explanation fields. A leaked result is worse than no result.
- **Anti-goal:** a gain we can't attribute to *relevance* (that's what the random control is for).

## Background (literature, 2026-11-16 survey)

- English medical evidence improved non-English medical MCQ accuracy for <30B models (~+8%), while 70B+ models often got worse from knowledge conflict (Yeginbergen et al. 2026, arXiv 2604.20531; 6 languages, no Hindi).
- Native-language-only evidence was weakest for low-resource languages; English (+ native) was best. Authoritative non-English medical text is scarce (same paper).
- Searching across languages beat translate-the-query-first; translating retrieved passages added ~2–3 pts (Ranaldi et al., EACL 2026 Findings, arXiv 2504.03616; includes Hindi via MLQA, general domain).
- Retrievers and generators lean towards English text (Park & Lee, ACL 2025 Findings, arXiv 2502.11175), so retrieval quality must be checked, not assumed.
- No published Hindi medical RAG study found.

## Approaches considered

| | Approach | Trade-off | Decision |
|---|---|---|---|
| **A** | English MedRAG Textbooks + StatPearls, searched cross-lingually with the Hindi question (BGE-M3) | Authoritative, pre-chunked, used in published medical RAG work; no translation. HiMed reads English passages; Indian public-health / AYUSH questions likely poorly covered | **Chosen** (Grace, 2026-11-16) |
| B | A + Wikipedia (English or Hindi) | Covers policy/GK questions; much larger index, less authoritative, higher exam-leakage risk | Follow-up arm, only if the per-type breakdown shows A failing on those questions |
| C | Translate textbooks into Hindi | Expensive; adds translation error before we know RAG helps | Dropped |

Later variant (only if A shows a gain): translate retrieved passages into Hindi before prompting (Ranaldi et al.'s CrossRAG idea).

## Decisions

### Corpus
- **MedRAG/textbooks** — on Hugging Face (`MedRAG/textbooks`, ~125.8k chunks ≤1000 chars, 18 USMLE-oriented textbooks). Fields `id, title, content, contents`.
- **StatPearls** — *not* redistributable. Download from NCBI Bookshelf (`statpearls_NBK430685.tar.gz`) and chunk with MedRAG's `src/data/statpearls.py` (~301k paragraph chunks). Same fields.
- Combined corpus ≈ 427k passages. Text embedded = `contents` (title + content), matching MedRAG.
- Store the built corpus + index under `~/scratch` (project rule), never in the repo. StatPearls text must not be committed.

### Retriever
- **BGE-M3, dense mode** (multilingual, 1024-dim). MedRAG's prebuilt indexes (MedCPT/Contriever/SPECTER) are English-only and can't match Hindi queries, so we embed the corpus ourselves.
- **FAISS exact inner-product index** (`IndexFlatIP`, normalized vectors). ~427k × 1024 float32 ≈ 1.75 GB — fine on scratch; exact search avoids approximate-search noise.
- Retrieval **query = Hindi question + all option texts** (as in MedRAG). The option texts are already shown to the model in every condition, so including them is not leakage. What is gold is *which* option is correct: the query must never include `answer_letter` or anything derived from it. **Leakage test:** the query for an item is identical whatever its `answer_letter` is. (Claude's call, from the builder check.)

### Prompting
- **k = 5** passages (pre-registered, not tuned). ~5 × ~180 tokens ≈ 1k tokens; well within HiMed's (Llama-3.1) context.
- Passages go before the question, in retrieval order, each labelled `[1] Title: … \n …`. Instruction (English, consistent with existing prompts) tells the model the passages may or may not be relevant and to answer with the letter only. Otherwise identical to C0: greedy decoding, same `first_letter` scoring.

### Conditions
| Label | What the model sees | Purpose |
|---|---|---|
| `c0_baseline` | existing run, no passages | reference |
| `rag_rand_k5` | 5 passages drawn uniformly at random from the same corpus (fixed seed, per item) | control: effect of *any* extra reference text |
| `rag_k5` | top-5 BGE-M3 passages | treatment |

### Retrieval as a separate, cached step
Retrieval runs once and writes a JSONL file (one line per item: `item_id`, passage ids, scores, titles, text) for both `rag_k5` and `rag_rand_k5`. `run.py` gets a `--passages-file` flag that reads it. Why: the generation job stays free of the embedding model, reruns are deterministic, and the passages are auditable.

### Retrieval spot-check (gate before the full run)
Before spending GPU time on 489 items, Grace reads the top-5 passages for **20 fixed-seed sampled test items** and marks each item: *relevant passage present* / *partial* / *none*.
- **Proceed** if ≥10/20 have at least one relevant passage.
- **Otherwise** stop and revisit retrieval (e.g. translate the query to English first) — a mostly-irrelevant retriever makes the accuracy result uninterpretable.
- Record the result in the results doc. (Threshold is Claude's call; Grace can change it before looking.)
- The spot-check looks at passages only, never at model answers, so it is not tuning on outcomes. **If the gate fails, the plan stops there and we return to design**; no fallback retriever is built in advance. If retrieval is then changed, the results doc must say so, and the 20 spot-check items are reported separately. (Claude's call.)

### Stats (same machinery as the ladder)
- Paired McNemar exact test, Bonferroni with a **pre-registered family m = 2**:
  1. `rag_k5` vs `c0_baseline` — does RAG help?
  2. `rag_k5` vs `rag_rand_k5` — is the gain from *relevance*?
- RAG "works" when both are positive and adjusted p < 0.05. If (1) holds but (2) doesn't, report it as an extra-context effect, not knowledge.
- `rag_rand_k5` vs `c0_baseline` is reported as descriptive (no test).
- Harm: a tested comparison with Δ < 0 and adjusted p < 0.05 is reported as harm; `rag_k5` significantly worse than `c0_baseline` gives the verdict "RAG hurts (significant)", not "no effect".
- Pre-registered: if a RAG condition's non_letter_rate (share of responses whose Response part is not a bare option letter) exceeds 5%, the first-letter scorer may misread sentence answers; also report after_marker-mode accuracy and hand-audit those rows. `score.py` is not changed.
- Report accuracy + Wilson CI, NONE rate, flips (+/−) as in `results_himed.md`.
- `analyze.py` today compares only consecutive ladder rows and always writes `results_{model}.md` (would overwrite the ladder results). **Implementation (Claude's call, in the plan):** a small `performance_eval/rag/report.py` reuses `analyze.py`'s scoring and McNemar helpers, runs exactly the comparisons above, and writes `results_himed_rag.md`; `analyze.py` and the ladder file are untouched.

### Breakdown by question type (exploratory, no significance test)
Each of the 489 items gets one tag: **clinical/biomedical** (textbook-coverable), **Indian public health / policy / general knowledge**, **yoga / AYUSH / traditional**, **other / non-medical** (MILU's subject label includes off-topic items, e.g. one about वीर रस, a Hindi poetics concept). Report accuracy per type per condition and the flips per type. This is what decides whether Approach B is worth running.
- **Method (Grace, 2026-11-16):** a local model on PACE (default: HiMed-8B; no paid API) assigns each item one tag from the question text; Grace hand-tags a fixed-seed sample of 50 and we report agreement (% and Cohen's κ) next to the breakdown.
- **Timing (Grace):** tagging happens *later*, after the RAG runs, as its own step. It does not block the accuracy comparison.
- **Blindness rule (Grace):** each question's type is decided by reading the question itself. Knowing whether RAG got it right must not influence the label. Concretely: the tagging prompt sees only the question + options; the 50-item hand-check sheet shows only the question + options (no model answers, no correct/incorrect, no condition); tags are written to their own file and only joined to results afterwards.

## Deliverables

- `performance_eval/rag/` — corpus build, embedding/index build, retrieval → JSONL (new).
- `run.py` `--passages-file` + `prompts.py` passage block (extend existing).
- `performance_eval/rag/report.py` (RAG comparisons + per-type breakdown → `results_himed_rag.md`).
- Later step: tagging script (local model → `question_types.csv`) + blind 50-item hand-check sheet + agreement report.
- `scripts/rag_build_index.sh`, `scripts/rag_retrieve.sh` (sbatch, PACE conventions) + reuse `performance_milu_run.sh` for generation.
- Raw/scored CSVs + passages JSONL under `~/scratch/performance_eval_outputs/milu_hi_med/`.
- `performance_eval/analysis_outputs/results_himed_rag.md` + spot-check record + per-type table.
- Tests: leakage guard (query never contains gold), prompt passage block, random control determinism, passages-file/item_id alignment, config sidecar includes retrieval settings.

## Constraints

- PACE cluster, NVIDIA GPUs; all caches/outputs/corpus/index under `~/scratch`.
- No paid API budget.
- Reuse existing resume + config-sidecar conventions; the sidecar must record corpus, retriever, k, passages-file hash, so a resume under different passages is refused.
- `literacy_eval/` and model-wrapper defaults untouched.

## Left to the builder

Exact prompt wording; JSONL schema details; embedding batch size / fp16; how StatPearls download is scripted; test file layout; how the random sampler avoids duplicate passages.

## Open items

1. Risk: HiMed may ignore or be confused by English passages in a Hindi prompt — measured, not assumed; the random control and NONE rate will show it.
2. Risk: StatPearls download URL / script may have changed; verify at build time.

## Future work (parked, not in scope)

- **Literacy-oriented RAG**: retrieval from patient-friendly Hindi health material, tied back to `literacy_eval/`. Grace wants to keep this option open.
- Approach B (Wikipedia arm), translated-passage variant, stacking RAG on the best C1–C4 config, Nanda once parseable.
