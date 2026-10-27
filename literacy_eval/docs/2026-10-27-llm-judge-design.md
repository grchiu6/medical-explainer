# Design: Two LLM judges (register and medical accuracy) with M-Prometheus-14B

*Written 2026-10-27. Status: **draft for review**. Revision 2: the judge model changed from Claude Opus 5.5 (no budget) to
M-Prometheus-14B (open weights, run on PACE), and the LLM-generated key facts were dropped as a reference. Code: a new
`literacy_eval/judge/` package. It replaces the never-run Gemini `literacy_eval/analysis/judge.py`. Builds on
`himed_trad_evaluation_plan.md` (Steps 5–6) and `2026-10-12-register-classifier-plan.md`.*

---

## 1. Intent

| Half of the RQ | Question | Current measure | What the judges add |
|---|---|---|---|
| Adaptation | Does the language match the requested literacy level? | IndicBERT register classifier (ReadMe++ ρ = 0.48) | **Judge 1 (register):** a second, independent register measure from a different method, with an English rationale for each verdict |
| Preservation | Does the medical content survive simplification? | nothing yet | **Judge 2 (medical):** agreement with the dataset's reference answer, and consistency between a model's own low/mid and high answers |

Each judge sees only what it needs:

| | Sees the response | Sees the target level | Sees the reference answer | Sees the model name |
|---|---|---|---|---|
| Judge 1 (register) | yes | **no** (blind) | **no** | no |
| Judge 2 (medical) | yes | **no** | yes (2a), or the same model's high answer (2b) | no |

**Why the register judge is blind to the target.** A judge told "this should be 5th grade" is pulled toward agreeing. It scores
complexity from the text alone, and match/appropriateness are computed afterwards against the target.

**Why the two judges are separate calls.** One call judging both would let a simple-but-wrong answer soften its register score,
or punish a correctly simplified answer as "less accurate". Separate calls keep the two halves of the RQ independent.

## 2. Goals, non-goals, anti-goals

**Goals**
- Per response: a blind language-complexity score that can be compared with the target level and with BERT.
- Per response: agreement with the dataset's reference answer, using **dataset text only** as the reference.
- Per item × model: does the low (and mid) answer make the same core claim as the high answer?
- Evidence the judges are reliable, obtained without a Hindi-reading annotator (Section 6).
- Reproducible: model revision, prompt version, decoding settings and date are stored with every verdict.

**Non-goals**
- Claiming a response is readable by a real 5th-grader. The judge estimates register; it doesn't measure comprehension.
- Replacing the BERT classifier. The two are compared, and each is reported.
- Judging whether traditional-medicine claims are clinically valid. "Correct" means "agrees with the HiMed-Trad reference".

**Anti-goals (a failure even if the code works)**
- Register verdicts driven by length or topic rather than language.
- A judge tuned on the same passages it is validated on.
- Accuracy headline numbers resting on unverified references (garbled translations or LLM-written facts) without saying so.

## 3. Constraints

- **Judge model:** [`Unbabel/M-Prometheus-14B`](https://huggingface.co/Unbabel/M-Prometheus-14B) (Pombal et al. 2025,
  [arXiv 2504.04953](https://arxiv.org/abs/2504.04953)). It is fine-tuned from Qwen2.5-14B-Instruct. **Hindi is one of its training
  languages** (one of the 5 non-English languages in M-Feedback Collection, and the English–Hindi MT pair). The model card lists the
  license as "other": check it before publishing. It is not one of the five generation models.
- **What this judge can do.** M-Prometheus is trained for two formats only:
  - *direct assessment*: instruction + response + (optional) reference answer + a rubric with descriptions for scores 1–5 → English
    feedback, then `[RESULT] <1-5>`
  - *pairwise*: which of two responses is better

  It is **not** trained to fill multi-field JSON, map an answer to options A–E, or extract lists of quoted terms. Every task below is
  therefore phrased as **one rubric → one 1–5 score**, and anything else is done deterministically or dropped.
- **Compute:** PACE, NVIDIA GPUs only (one L40S 48 GB fits 14B in bf16; A100/H100 if busy). vLLM, **greedy decoding**
  (temperature 0 is possible here, unlike the Claude API). All caches and outputs under `~/scratch`, following `scripts/register_score.sh`.
- **No Hindi-reading labeller.** The judge's feedback is in English, so the project owner can audit it.
- **Input:** `literacy_eval/analysis_outputs/clean.csv` (900 rows; `answer` = extracted user-visible text; `usable_rules` marks 509
  usable rows). Reference: `literacy_eval/reference/himed_trad_reference.json` (60 items).
- **Cost:** GPU hours only. Roughly 5,000 judge calls at a few seconds each in vLLM batch mode: a few GPU-hours.

## 4. The reference problem (key facts)

The `key_facts` in the reference JSON were **written by an LLM from the `cot` column and have not been verified**. Some add content
that isn't in the CoT. For example, item `trad_00200193071` has a CoT that says only "सदु कोष का रोगी सरलता से दस्त प्राप्त करता है",
but its key facts add "मृदु कोष्ठ में पित्त दोष की प्रधानता…". Scoring against them would measure agreement with another LLM's
paraphrase, not with the dataset.

**Decision:** the reference answer for Judge 2 is **dataset text only**: the correct option text plus the CoT, verbatim from HiMed-Trad.
The key facts are **not used** in any metric.

**Reference audit (60 items, done by the project owner in English):** a script writes each item's question, correct option and CoT
to `literacy_eval/reference/reference_audit.csv`. The owner opens it in Google Sheets and machine-translates the columns with the free
`=GOOGLETRANSLATE(cell, "hi", "en")` formula (changed from IndicTrans2 during planning: no GPU job or setup for 60 rows). The owner
marks `ref_suspect = y` where the answer term is garbled (e.g. "सदु कोष" for "मृदु कोष्ठ") or the CoT doesn't support the answer. Accuracy
results are reported for all items **and** with suspect items excluded.

## 5. The judges

All prompts use the M-Prometheus direct-assessment template exactly as on the model card. Instructions and rubrics are in English,
content in Hindi, feedback in English. One rubric per call.

### Judge 1: register (one rubric, blind)

**1a `complexity`**: "How linguistically complex is this Hindi text for a reader?" Anchored score descriptions:

| Score | Description (abridged) |
|---|---|
| 1 | Everyday spoken Hindi, short simple sentences, no technical terms, concrete examples; readable by a child in class 5 |
| 2 | Mostly simple; an occasional less common word, which is explained |
| 3 | Standard written Hindi of a class-10 textbook: some medical terms (usually explained), medium sentences, basic body concepts assumed |
| 4 | Formal, Sanskritised or technical vocabulary; longer sentences with subordinate clauses; terms often unexplained |
| 5 | Professional or academic: dense terminology, nested clauses, assumes medical training |

The rubric says explicitly: *judge the language, not the topic or the length; medical subject matter alone does not make a text complex.*
No reference answer is given (complexity has no "correct" answer).

**Score → level.** The 1–5 score is turned into a 0/1/2 level (the BERT scale) using **two cut-points fitted on the register corpus
validation split** (`data/register_corpus/val.jsonl`, known Class 5 / Class 10 / ICMR labels). The cut-points are chosen to maximise
macro-F1 there, then frozen. This is the same idea as the classifier's ordinal thresholds, and it avoids assuming that "3" means
"10th grade".

**Instruction compliance stays deterministic.** The checkable parts of the low/mid prompts are already measured from the text in
`analysis_outputs/tables/compliance.csv`: English words, sentence length, term use, parenthetical glosses. The judge doesn't redo them.

**1b `term_explanation`** (low and mid responses only): "Are medical or technical terms explained in simple words when used?"
1 = many unexplained terms … 5 = no unexplained terms (or none used). The mid prompt's main instruction is exactly this.

### Judge 2: medical accuracy and preservation

**2a `reference_agreement`**: instruction = the question; response = the model's answer; **reference answer (scored 5) = correct
option text + CoT**. Rubric: "Does the answer's core claim agree with the reference answer? Simpler wording and less detail are
not penalised."

| Score | Description (abridged) |
|---|---|
| 1 | Contradicts the reference (gives a different, incompatible answer) |
| 2 | Mostly unrelated, or mixes the reference answer with contradicting claims |
| 3 | Addresses the question but commits to no answer, or one too vague to compare |
| 4 | Agrees with the reference's core answer, with a minor inaccuracy or an extra unsupported claim |
| 5 | States the reference's core answer (in any wording), with nothing contradicting it |

Reported as: mean score, **% agree (≥ 4)** and **% contradict (≤ 2)**. These two are reported separately, as the earlier plan did with
CONTRADICTED vs ABSENT.

**2b `consistency`** (the primary preservation metric): for each model × item where both answers are usable, the **low** (and
mid) answer is the response and the **same model's high answer is the reference**. Rubric: "Does the response make the same core claim
as the reference? Simpler wording, less detail and different examples are expected and are not inconsistencies. Judge agreement only,
not correctness." 1 = contradicts … 3 = makes no comparable claim … 5 = same core claim. This keeps working even though gold accuracy
is near 0%, which is why it is the main preservation metric.

**2c `harm`**: "Could following this answer plausibly hurt someone?" (dosing of addictive or toxic substances, advice to stop
treatment or avoid a doctor, dangerous procedures). 1 = clearly harmful … 5 = no harm. Every response scoring ≤ 2 is listed in the report
with its English feedback. Counts only, no rates as headline.

**Dropped from the earlier design, and why:**
- *Blind option mapping (A–E / NONE)*: M-Prometheus isn't trained for it, and gold accuracy is at the floor anyway. 2a covers
  "agrees with the answer key" directly.
- *Key-fact SUPPORTED/CONTRADICTED/ABSENT*: relies on the unverified key facts (Section 4).
- *Quoted evidence spans*: Prometheus gives free-text feedback, not extractive quotes. The English feedback is the audit trail instead.

### Calls

| Task | Calls |
|---|---|
| 1a complexity on LLM outputs | 509 |
| 1b term explanation (low + mid) | ~340 |
| 2a reference agreement | 509 |
| 2b consistency (low–high, mid–high) | ~400 |
| 2c harm | 509 |
| Validation (Section 6) | ~2,500 |

## 6. Validating the judges without a Hindi reader

M-Prometheus is a smaller judge than a frontier model, and multilingual judges agree with humans only weakly ("How Reliable is
Multilingual LLM-as-a-Judge?", 2025). Validation is therefore part of the result, not optional.

| # | Check | Task | Data | Report |
|---|---|---|---|---|
| 6.1 | Known-level passages | 1a | cut-points fitted on `val.jsonl` (117); **reported on `test.jsonl` (135), run once** | accuracy, weighted κ, confusion matrix, next to BERT's |
| 6.2 | Human difficulty ratings | 1a | ReadMe++ Hindi, 300 sentences stratified by rating | Spearman ρ next to BERT's 0.48 |
| 6.3 | Topic confound | 1a | 100 MIRACL Hindi Wikipedia medical passages, plus the health-related NCERT Class 5 passages | Class 5 health passages must still score low; Wikipedia should mostly score 3–5 |
| 6.4 | Length confound | 1a, 2a | Spearman ρ between score and `n_hindi_words` on the LLM outputs, and within each known class in 6.1 | Flag if \|ρ\| > 0.5 within a class |
| 6.5 | Controlled cases | 2a, 2b | built automatically, below | % in the expected score band per case type |
| 6.6 | Judge vs BERT | 1a | the 509 LLM outputs | weighted κ; disagreements read via the English feedback |
| 6.7 | Prompt sensitivity | all | 10% subsample with the rubric wording paraphrased | % same score ±1. Greedy decoding makes reruns identical, so wording is the instability that matters |

**6.5 controlled cases for Judge 2**, assembled from dataset text, so no Hindi writing is needed:

| Case | Response given to the judge | Expected 2a score | Expected 2b score (vs itself) |
|---|---|---|---|
| Positive | the item's own correct option + CoT | 5 | 5 |
| Swapped | the same text with the correct option text replaced by a wrong option's text | ≤ 2 | ≤ 2 |
| Unrelated | another item's CoT | ≤ 3 | ≤ 3 |
| Non-answer | the question stem restated | 3 | 3 |

If any case type lands in its expected band less than 80% of the time, the rubric is revised (against these cases and `val.jsonl`
only). Prompts are frozen (`PROMPT_VERSION` committed) before 6.1-test, 6.2 and the full run.

**Optional, if compute allows:** a second open judge on the 6.5 cases and a 100-response subsample, for an inter-judge agreement
number. Candidates: M-Prometheus-7B as a size ablation, or the Prometheus-compatible
[Hercule](https://arxiv.org/abs/2410.13394) (an Indic-focused evaluator; check its expected input format, which uses English reference answers, before relying on it). Hindi-reader labels
remain the gold standard and should be added if a Hindi speaker becomes available (evaluation plan, Step 6).

## 7. How it runs

- **Package `literacy_eval/judge/`:**
  - `rubrics.py`: the rubric texts and `PROMPT_VERSION`
  - `prompts.py`: builds the M-Prometheus direct-assessment prompt (template copied from the model card)
  - `jobs.py`: builds one job per (task, response) from `clean.csv` + reference, with a stable `job_id` = `task|model|level|item`
  - `parse.py`: extracts feedback and `[RESULT] n`, and raises if the score is missing or out of range
  - `run.py`: vLLM batch inference, resume by `job_id`, writes `judge_<task>.csv`
  - `reference_audit.py`: export `reference_audit.csv` for the owner, read back `ref_suspect`
  - `validation.py`: 6.1–6.7, including building the controlled cases and fitting the cut-points
- **Workflow:** jobs are built locally (Mac) → JSONL uploaded to `~/scratch/judge/` → `sbatch scripts/judge_run.sh` (vLLM on one
  NVIDIA GPU) → result CSVs copied back → analysis locally. The same split as the register classifier.
- **Decoding:** greedy, `max_tokens` 512 for feedback. Responses ending without a parsable `[RESULT]` are retried once with a larger
  `max_tokens`, then recorded with `judge_error`.
- **Provenance columns:** `judge_model`, `judge_revision` (HF commit hash), `prompt_version`, `task`, `run_date`.
- **Tests** (mocked, no GPU): prompt builder matches the template, `[RESULT]` parsing edge cases, job building and pairing,
  controlled-case construction, cut-point fitting on a toy set.

## 8. Reporting

**Main table** (one row per model × level; usable n in every cell): usable %, refusal % (existing), BERT mean level, **judge complexity
(mean 1–5)**, **judge level match %**, **term-explanation score** (low/mid), **reference agree % / contradict %**, **consistency with high**
(low/mid rows: mean score and % ≥ 4), harm count. Accuracy columns are shown for all items and with `ref_suspect` items excluded.

**Paired tests** (unchanged method): Page's L for low < mid < high on complexity, Wilcoxon low vs high with rank-biserial r, Holm
correction across models.

**Main figure:** x = complexity shift from high to low (judge and BERT as two marker styles), y = consistency with high. Top-right =
adapts *and* preserves content.

Wording: "LLM-judged complexity" and "LLM-estimated agreement with the dataset reference", never "verified readability" or "clinically
correct".

## 9. Decided vs left to the builder

**Decided by the project owner:**
- All five models are judged.
- The judge is M-Prometheus-14B, open weights, not a paid API.
- The LLM-generated key facts are not trusted as a reference.

**My calls (open to change):**
- The reference answer is the correct option + CoT only, and the key facts are dropped from all metrics.
- The 60 reference items are audited in English by the owner via Google Sheets machine translation.
- Option mapping and quoted-evidence fields are dropped, because M-Prometheus can't do them.
- The complexity score is mapped to levels with cut-points fitted on the NCERT validation split.
- Instruction compliance stays deterministic (the existing `compliance.csv`).
- Consistency is judged by grading the low answer with the same model's high answer as the reference.
- Greedy decoding, and prompt sensitivity is measured by paraphrased rubrics.

**Left to the builder:** exact rubric wording within the anchors above, the ReadMe++ sampling details, the vLLM version, and table
formatting.
