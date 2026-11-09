# Judge error analysis: sampling plan and codebook

Goal: check by hand whether M-Prometheus's scores and explanations are right, and find recurring patterns in
(a) the judge's mistakes and (b) the models' answers.

## 1. Sample

`python -m literacy_eval.judge.error_sample` writes `literacy_eval/reference/judge_error_sample.csv`.

- **Population**: the 2,100 primary judge decisions (4 tasks x 5 models x 3 levels x 42 valid items; term
  explanation and consistency only at low/mid). The 18 invalid items are not sampled.
- **Strata**: task x judge score (4 x 5 = 20). **5 decisions per stratum = 100 rows**. Equal allocation is used
  on purpose: rare scores (e.g. reference agreement = 5, only 13 cases) would otherwise never be seen, and
  those are exactly where false positives hide.
- **Within a stratum**: random order (seed 20261108), then round-robin over the 5 models, so no model dominates.
- **Weights**: `weight = stratum_n / 5` (how many decisions each row stands for). `--analyze` uses them, so a
  percentage it prints is an estimate for the whole run, not just for the 100 rows. Unweighted counts are
  printed too; use those for "patterns", the weighted ones for "how often".

## 2. How to review one row

Import the CSV into Google Sheets (File → Import → replace). The `*_en` columns translate themselves.
Per row read, in order:

1. `question_en`, then `answer_en` (and `reference_en` for reference_agreement / consistency).
2. The judge's `score` and `feedback`.
3. For any key term, check the Hindi column, not just the English: Google Translate makes garbled words look
   real ("सदु कोष" → "Sadu Kosh") and real words look wrong ("दुबलापन" → "obesity").
4. Fill the six review columns below. About 2-3 minutes per row; 100 rows ≈ 4 hours, can be split.

What each task's score is supposed to measure (full rubric: `literacy_eval/judge/rubrics.py`):

| task | 1 | 3 | 5 |
|---|---|---|---|
| complexity | class-5 everyday Hindi | class-10 textbook Hindi | professional / academic Hindi |
| term_explanation | many terms, none explained | some explained | all explained, or no terms |
| reference_agreement | contradicts the reference (2 = unrelated) | on topic but no clear answer | states the reference's core answer |
| consistency (vs the model's own high answer) | claims contradict (2 = differ) | no comparable claim | same core claim |

## 3. Review columns

| column | values | meaning |
|---|---|---|
| `human_score` | 1-5 | the score *you* would give under the same rubric |
| `score_verdict` | `correct` / `too_high` / `too_low` / `unsure` | the judge's score vs yours (±0 = correct) |
| `feedback_faithful` | `y` / `partly` / `n` | does the explanation describe what the answer and reference actually say? |
| `judge_error` | one code from 3a | the main reason the judge went wrong (`J0` if it didn't) |
| `answer_error` | one code from 3b | what is wrong with the model's answer (`A0` if nothing) |
| `notes` | free text | one line: the evidence ("feedback says neem, reference says garlic, still 4") |

Use `unsure` and a note when you cannot tell (e.g. translation unclear); these are dropped from the rates.

### 3a. Judge error codes

| code | name | example |
|---|---|---|
| J0 | no error | score and reasoning are right |
| J1 | misread / invented content | feedback claims the answer or reference says something it doesn't ("gold leaf", "angina") |
| J2 | right diagnosis, wrong band | feedback says "unrelated" but gives 1 (contradicts) instead of 2 |
| J3 | too lenient | partial or different answer scored 4-5 |
| J4 | too strict | correct but simpler / shorter answer scored low |
| J5 | feedback contradicts score | explanation argues for one score, number is another |
| J6 | judged the wrong thing | complexity scored on topic or length; agreement scored on style |
| J7 | mishandled a poor response | echo, loop, refusal or non-answer scored as if it were a real answer |
| J8 | other | describe in notes |

### 3b. Model answer codes

| code | name |
|---|---|
| A0 | fine for the task (correct claim / right register) |
| A1 | wrong answer: a different, specific claim |
| A2 | off-topic: talks about something else |
| A3 | vague: on topic but commits to nothing |
| A4 | echo: restates the question or the prompt |
| A5 | repetition loop / degenerate text |
| A6 | refusal |
| A7 | wrong language (not Hindi) |
| A8 | wrong register for the requested level (too complex for 5th grade, too simple for professional) |
| A9 | fabricated remedy or term (invented drug, plant, procedure) |

For complexity / term_explanation rows, A0 vs A8 is the main call; the medical codes still apply if obvious.

### 3c. Decision rules (fixed during the AI-drafted pass, 2026-11-09)

- **J0 = the score is right.** `judge_error` names the cause of a *wrong* score. When the score is right but
  the explanation is flawed, it is still J0 and the flaw goes in `feedback_faithful` (partly / n).
- **reference_agreement bands:** 1 = a different specific answer; 2 = off-topic text, prompt echo, or the
  reference answer mixed with a contradicting claim; 3 = on-topic generic advice or a loop naming no answer.
- **consistency:** if either answer is an echo / loop / non-answer, there is no comparable claim -> 3.
  A 3 from this rule means "not measurable", not "medium agreement".
- **term_explanation:** the rubric is applied literally: a non-answer with no technical terms gets 5. This is a
  rubric gap (a refusal can earn the top score); report such rows separately.

### 3d. How the current labels were made

The sheet was filled by AI agents, not by a person: one agent per task drafted all 25 rows (reading the
Hindi directly), a second agent checked every row independently (`checker_agreed`, `checker_note`), and
disagreements were settled by applying the checker's evidence-backed correction (`adjudication`).
`review_source` says this on every row. It is a draft for human verification: report it as "AI-assisted
error analysis, human-verified on N rows", and start verification with low-`confidence` and
`checker_agreed = no` rows.

## 4. Analysis

`python -m literacy_eval.judge.error_sample --analyze` prints:

- `verdict_by_task`: weighted % correct / too high / too low per task (judge reliability estimate);
- `judge_error_by_task` and `judge_error_counts`: which judge errors dominate where;
- `score_agreement_by_task`: exact and ±1 agreement between judge and human score;
- `feedback_faithful_by_task`: how often the explanation can be trusted;
- `answer_error_by_model`: the models' error profile.

To write up: report the weighted rates with n, then 2-3 patterns, each with one quoted example (sample_id).
Known patterns to confirm or refute from the pilot: J2 on non-answers (gap 1), J1 on garbled terms, J3 on
high reference-agreement scores.
