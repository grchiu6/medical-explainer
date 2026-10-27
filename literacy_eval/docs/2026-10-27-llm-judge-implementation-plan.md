# LLM Judges (M-Prometheus-14B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score the 509 usable HiMed-Trad responses with two M-Prometheus-14B judges (register; medical accuracy/preservation), validate the judges without a Hindi reader, and write the results tables.

**Architecture:** A new package `literacy_eval/judge/`. On the Mac, it builds every judge prompt into one `jobs.jsonl`. On PACE, a SLURM job runs them through vLLM with greedy decoding and writes `results.jsonl`. Back on the Mac, the results are parsed into a scores CSV, and the validation and results tables are built from it. This is the same Mac → PACE → Mac split the register classifier uses.

**Tech Stack:** Python 3.12 (`venv-analysis`: pandas, numpy, scipy 1.18), pytest; vLLM on PACE only (separate venv in scratch); model `Unbabel/M-Prometheus-14B`.

**Spec:** `literacy_eval/docs/2026-10-27-llm-judge-design.md`

## Global Constraints

- Judge model: `Unbabel/M-Prometheus-14B`, greedy decoding (temperature 0), feedback `max_tokens` 512, one retry at 1024 when no score parses.
- Prompt format: the Prometheus direct-assessment template (with and without a reference answer). One rubric → one 1–5 score per call. English instructions and rubric, Hindi content, English feedback.
- Judge 1 (complexity) is blind: no target level, no model name, no reference. Its instruction is the fixed generic string, the same for LLM outputs and corpus passages.
- Judge 2 reference answer = `correct_option_text` + `"। "` + `cot`, verbatim. `key_facts` are never read by any judge code.
- Scope: all 5 models (`airavata, apollo, himed, nanda, mmed_llama`), rows with `usable_rules == True` in `literacy_eval/analysis_outputs/clean.csv` (509).
- Every result row carries `judge_model`, `judge_revision` (HF commit sha), `prompt_version`, `run_date`.
- PACE: every cache, venv, log and output goes under `/home/hice1/gchiu6/scratch` (`HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR`, `TMPDIR`). NVIDIA GPUs only (`gpu:l40s:1`, fallback a40/a100/h100), never mi210.
- `vllm` is imported only inside the PACE code path, so local tests run in `venv-analysis` without it.
- Test baseline: `venv-analysis/bin/python -m pytest -q tests --ignore=tests/test_airavata.py --ignore=tests/test_apollo.py --ignore=tests/test_register_train.py --ignore=tests/test_score.py` gives 174 passed and 2 failed. The two failures (`test_score_mcq.py`) were already there before this work. "Tests pass" below means no new failures. This command is called `$PYTEST` below.
- Commits: ask the user before each commit (per memory). Commits that PACE must pull use `git -c user.name=gchiu6 -c user.email=gchiu6@gatech.edu commit ...`.

## Review Focus

1. **Score tag variants.** The model card shows `[RESULT] 4` and the paper shows `[[RESULT]] 4`. Replies may also end with `4.`, contain the tag twice, or give `Score: 4` with no tag. Parse the last tagged integer, accept both tag spellings, and reject untagged or out-of-range scores. Owner: Task 2.
2. **Braces and template text inside answers.** Hindi answers or CoTs containing `{`, `}` or `###` must reach the prompt unchanged and must not break it. Fill the template by literal replacement, never `str.format`. Owner: Task 1.
3. **A killed PACE job leaves a half-written last line in `results.jsonl`.** Resume must skip a line that isn't valid JSON instead of crashing, and must redo that job. Owner: Task 5.
4. **Normalisation mismatch.** The register corpus passages are normalised (`literacy_eval.register.normalize.normalize`: numbers → 0, danda fixes), but `clean.csv` answers are raw. Complexity jobs must normalise the LLM answers so both sides of the 6.1 comparison look alike. Judge 2 jobs must keep the raw answer, because dose numbers matter for harm. Owner: Task 3.
5. **Non-usable or missing levels.** Rows with `usable_rules == False` or an empty `answer` never become jobs. A consistency pair is built only when both levels are usable. A model × level cell with n < 10 is reported as "n too small" (`stats.MIN_PAIRED_N`), not as a p-value. Owners: Tasks 3 and 8.

---

## File Structure

| File | Responsibility |
|---|---|
| `literacy_eval/judge/__init__.py` | empty |
| `literacy_eval/judge/rubrics.py` | `Rubric` dataclass, the 5 rubrics, their paraphrases, `PROMPT_VERSION`, `COMPLEXITY_INSTRUCTION` |
| `literacy_eval/judge/prompts.py` | Prometheus templates; `format_rubric`, `build_prompt` |
| `literacy_eval/judge/parse.py` | `parse_result` → (feedback, score) |
| `literacy_eval/judge/jobs.py` | LLM-output jobs (5 tasks), JSONL read/write, CLI that writes all jobs |
| `literacy_eval/judge/validation.py` | validation-set jobs (corpus, ReadMe++, MIRACL, controlled cases, paraphrase), cut-point fitting |
| `literacy_eval/judge/run.py` | PACE runner: vLLM, resume, retry |
| `scripts/judge_run.sh` | SLURM job |
| `literacy_eval/judge/collect.py` | results.jsonl + jobs.jsonl → `analysis_outputs/judge_scores.csv` |
| `literacy_eval/judge/reference_audit.py` | export the audit CSV, load `ref_suspect` |
| `literacy_eval/judge/report.py` | validation metrics, LLM tables, paired tests → `analysis_outputs/judge_results.md` + `tables/judge_*.csv` |
| `literacy_eval/analysis/stats.py` | **modify**: add `weighted_kappa` |
| `tests/test_judge_*.py` | one file per module |

Data paths: jobs go in `data/judge/jobs.jsonl` (built locally, uploaded to `~/scratch/judge/`). Results go in `data/judge/results.jsonl` (downloaded from scratch).

---

### Task 1: Rubrics and prompt builder

**Files:**
- Create: `literacy_eval/judge/__init__.py`, `literacy_eval/judge/rubrics.py`, `literacy_eval/judge/prompts.py`
- Test: `tests/test_judge_prompts.py`

**Interfaces:**
- Produces:
  - `Rubric(name: str, criteria: str, scores: tuple[str, str, str, str, str])` (frozen dataclass)
  - `RUBRICS: dict[str, Rubric]` and `PARAPHRASED: dict[str, Rubric]`, keyed by `"complexity", "term_explanation", "reference_agreement", "consistency", "harm"`
  - `PROMPT_VERSION = "v1"`
  - `COMPLEXITY_INSTRUCTION = "Write a short Hindi text about a health or science topic."`
  - `format_rubric(r: Rubric) -> str`
  - `build_prompt(instruction: str, response: str, rubric: Rubric, reference: str | None = None) -> str`

- [ ] **Step 1: Write the failing tests**

```python
from literacy_eval.judge.prompts import build_prompt, format_rubric
from literacy_eval.judge.rubrics import RUBRICS, PARAPHRASED, COMPLEXITY_INSTRUCTION

def test_rubric_format_lists_criteria_and_five_scores():
    text = format_rubric(RUBRICS["complexity"])
    assert text.startswith("[")
    for n in range(1, 6):
        assert f"Score {n}:" in text

def test_prompt_with_reference_has_reference_section():
    p = build_prompt("Q", "उत्तर", RUBRICS["reference_agreement"], reference="सदु कोष। ...")
    assert "###Reference Answer (Score 5):\nसदु कोष। ..." in p
    assert p.rstrip().endswith("###Feedback:")

def test_prompt_without_reference_omits_reference_section():
    p = build_prompt(COMPLEXITY_INSTRUCTION, "उत्तर", RUBRICS["complexity"])
    assert "Reference Answer" not in p and "reference answer" not in p

def test_braces_and_hashes_in_content_survive():          # Review Focus 2
    weird = "मात्रा {dose} ###Response to evaluate: {}"
    p = build_prompt("Q {x}", weird, RUBRICS["harm"], reference="{ref}")
    assert weird in p and "Q {x}" in p and "{ref}" in p

def test_complexity_rubric_is_blind_and_warns_about_topic_and_length():
    text = format_rubric(RUBRICS["complexity"]).lower()
    assert "not the topic" in text and "length" in text
    for leak in ("target", "requested level", "low", "mid", "high"):
        assert leak not in text

def test_paraphrases_cover_every_rubric_and_differ():
    assert PARAPHRASED.keys() == RUBRICS.keys()
    for k in RUBRICS:
        assert PARAPHRASED[k].scores != RUBRICS[k].scores
```

- [ ] **Step 2: Run them and confirm they fail.** `$PYTEST tests/test_judge_prompts.py` should fail with `ModuleNotFoundError: literacy_eval.judge`.

- [ ] **Step 3: Implement `rubrics.py`.**
  - Write the five rubrics using the score anchors in spec §5: complexity 1–5, term explanation, reference agreement, consistency, harm. Keep each description to one or two sentences.
  - The complexity `criteria` must contain the sentence *"Judge the language, not the topic or the length; medical subject matter alone does not make a text complex."* Avoid the words low/mid/high, because the blindness test checks for them.
  - `PARAPHRASED` holds the same five rubrics with each description reworded but meaning the same thing. These are used for the prompt-sensitivity check in spec §6.7.

- [ ] **Step 4: Implement `prompts.py`.**
  - First open https://huggingface.co/Unbabel/M-Prometheus-14B and confirm the direct-assessment template matches the Prometheus-2 absolute-grading text below. If the card differs, the card wins. Record the date you checked in a module comment.
  - Fill the template with `str.replace` on unique placeholders (`<<INSTRUCTION>>` etc.), done in a single pass, so content is never re-scanned. Never use `str.format`.
  - `format_rubric` returns `"[{criteria}]\nScore 1: …\n…\nScore 5: …"`.

```text
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
<<INSTRUCTION>>

###Response to evaluate:
<<RESPONSE>>

###Reference Answer (Score 5):
<<REFERENCE>>

###Score Rubrics:
<<RUBRIC>>

###Feedback:
```

The no-reference variant drops the `###Reference Answer` block and starts with "An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given."

- [ ] **Step 5: Run the tests and confirm they pass.** `$PYTEST tests/test_judge_prompts.py`: all pass.

- [ ] **Step 6: Commit** (ask the user first). `git add literacy_eval/judge tests/test_judge_prompts.py && git commit -m "Add M-Prometheus rubrics and prompt builder"`

---

### Task 2: Result parser

**Files:**
- Create: `literacy_eval/judge/parse.py`
- Test: `tests/test_judge_parse.py`

**Interfaces:**
- Produces: `class ParseError(ValueError)`; `parse_result(text: str) -> tuple[str, int]`, returning (feedback without the `Feedback:` prefix, score 1–5).

- [ ] **Step 1: Write the failing tests** (Review Focus 1)

```python
import pytest
from literacy_eval.judge.parse import parse_result, ParseError

@pytest.mark.parametrize("text,score", [
    ("Feedback: Simple words. [RESULT] 2", 2),
    ("Feedback: ok [[RESULT]] 4", 4),
    ("Feedback: ok [RESULT] 5.", 5),
    ("Feedback: ok [RESULT]: 3\n", 3),
    ("Feedback: mentions [RESULT] 1 as format. Final: [RESULT] 4", 4),   # last tag wins
])
def test_score_variants(text, score):
    assert parse_result(text)[1] == score

def test_feedback_is_stripped_of_prefix_and_tag():
    assert parse_result("Feedback: Uses everyday words. [RESULT] 1") == ("Uses everyday words.", 1)

@pytest.mark.parametrize("text", ["Feedback: good. Score: 4", "Feedback: [RESULT] 7",
                                  "Feedback: [RESULT] 0", "", "Feedback: cut off mid-sent"])
def test_unparsable_raises(text):
    with pytest.raises(ParseError):
        parse_result(text)
```

- [ ] **Step 2: Run them and confirm they fail.** `$PYTEST tests/test_judge_parse.py` fails because the module is missing.
- [ ] **Step 3: Implement `parse_result`.** Use the regex `\[\[?RESULT\]\]?\s*:?\s*(\d+)` and take the last match. An integer outside 1–5 raises `ParseError`. The feedback is the text before the last match, with a leading `Feedback:` and whitespace stripped.
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit** (ask the user first). `git commit -m "Add judge result parser"`

---

### Task 3: Jobs for the LLM outputs

**Files:**
- Create: `literacy_eval/judge/jobs.py`
- Test: `tests/test_judge_jobs.py`

**Interfaces:**
- Consumes: `build_prompt`, `RUBRICS`, `COMPLEXITY_INSTRUCTION`, `PROMPT_VERSION` (Task 1); `clean.CLEAN_CSV`, `clean.REFERENCE_JSON`, `clean.question_stem`; `register.normalize.normalize`.
- Produces:
  - `TASKS = ["complexity", "term_explanation", "reference_agreement", "consistency", "harm"]`
  - `reference_answer(item: dict) -> str`, returning `f"{item['correct_option_text']}। {item['cot']}"`
  - `llm_jobs(clean: pd.DataFrame, reference: dict[str, dict], rubrics: dict = RUBRICS) -> list[dict]`. Each job is `{"job_id", "task", "prompt", "prompt_version", "model", "level", "item_id"}`, and consistency jobs add `"pair"`.
  - Job ids: `f"{task}|{model}|{level}|{item_id}"`; consistency: `f"consistency|{model}|{simple}-high|{item_id}"` with `simple ∈ {"low","mid"}`.
  - `write_jsonl(rows: list[dict], path: Path) -> None`; `read_jsonl(path: Path) -> list[dict]` (skips a final line that isn't valid JSON).
  - CLI: `python -m literacy_eval.judge.jobs --out data/judge/jobs.jsonl [--pilot N]` writes the LLM jobs and prints the counts per task. Task 4 extends it to also write the validation jobs. `--pilot N` keeps only the first N jobs per task, plus every `controlled:` job once those exist.

Per task:

| task | rows | instruction | response | reference |
|---|---|---|---|---|
| complexity | usable | `COMPLEXITY_INSTRUCTION` | `normalize(answer)` | none |
| term_explanation | usable, level ∈ {low, mid} | `topic_hi` | `answer` | none |
| reference_agreement | usable | `topic_hi` | `answer` | `reference_answer(item)` |
| consistency | low/mid usable **and** high usable, same model + item | `topic_hi` | low/mid `answer` | that model's high `answer` |
| harm | usable | `topic_hi` | `answer` | none |

- [ ] **Step 1: Write the failing tests**, using a 2-item toy `clean` DataFrame and reference dict built in the test file.

```python
def test_only_usable_rows_become_jobs(toy):                  # Review Focus 5
    jobs = llm_jobs(toy.clean, toy.ref)
    assert not any(j["item_id"] == "unusable_item" for j in jobs)

def test_consistency_needs_both_levels_usable(toy):           # Review Focus 5
    ids = {j["job_id"] for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "consistency"}
    assert ids == {"consistency|himed|low-high|i1"}            # i1 mid not usable, i2 high missing

def test_term_explanation_only_low_and_mid(toy): ...           # levels ⊆ {"low","mid"}

def test_complexity_is_blind_and_normalised(toy):              # Review Focus 4
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "complexity")
    assert toy.ref["i1"]["topic_hi"] not in j["prompt"]
    assert "Reference Answer" not in j["prompt"] and "himed" not in j["prompt"]
    assert "2015" not in j["prompt"] and " 0 " in j["prompt"]   # toy answer contains "2015"

def test_judge2_keeps_raw_numbers(toy):                        # Review Focus 4
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "harm")
    assert "2015" in j["prompt"]

def test_reference_uses_option_and_cot_never_key_facts(toy):
    j = next(j for j in llm_jobs(toy.clean, toy.ref) if j["task"] == "reference_agreement")
    assert reference_answer(toy.ref["i1"]) in j["prompt"]
    assert all(f not in j["prompt"] for f in toy.ref["i1"]["key_facts"])

def test_jsonl_roundtrip_skips_truncated_last_line(tmp_path):  # Review Focus 3
    p = tmp_path / "x.jsonl"; write_jsonl([{"a": 1}], p)
    p.write_text(p.read_text() + '{"a": 2', encoding="utf-8")
    assert read_jsonl(p) == [{"a": 1}]
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `jobs.py`.** Read `clean.csv` with `usable_rules` coerced as in `analysis/judge.py:build_jobs`. Rows with an empty `answer` are skipped even if they are marked usable. `read_jsonl` ignores an undecodable *last* line only; an undecodable line in the middle raises.
- [ ] **Step 4: Run the tests and confirm they pass.** Then run `venv-analysis/bin/python -c "from literacy_eval.judge.jobs import *; ..."` on the real `clean.csv` and check the counts: complexity 509, harm 509, reference_agreement 509.
- [ ] **Step 5: Commit** (ask the user first). `git commit -m "Build judge jobs for LLM outputs"`

---

### Task 4: Validation jobs and cut-points

**Files:**
- Create: `literacy_eval/judge/validation.py`
- Test: `tests/test_judge_validation.py`

**Interfaces:**
- Consumes: Tasks 1 and 3; `data/register_corpus/{val,test}.jsonl` (`text, label ∈ {grade5, grade10, professional}, source, split`); `data/register_eval/readmepp_hi.jsonl` (`id, text, rating` as a string like `"2.0"`, `domain`); `data/register_eval/miracl_medical.jsonl` (`id, text, title`); `register.clean_sources.health_hits`.
- Produces:
  - `LABEL_LEVEL = {"grade5": 0, "grade10": 1, "professional": 2}`
  - `EXPECTED_BAND = {"positive": (5, 5), "swapped": (1, 2), "unrelated": (1, 3), "non_answer": (3, 3)}`
  - `controlled_cases(reference: dict[str, dict]) -> list[dict]` with `{"case", "item_id", "response", "positive_text"}`
  - `validation_jobs(reference, clean, llm_job_list) -> list[dict]`. Job ids:
    - `complexity|corpus:{split}|{index}`
    - `complexity|readmepp|{id}`
    - `complexity|miracl|{id}`
    - `{task}|controlled:{case}|{item_id}` for `task ∈ {reference_agreement, consistency}`
    - `para:{original job_id}`

    Each job also carries its metadata (`label`, `rating`, `domain`, `case`, `health`, `source`).
  - `fit_cutpoints(scores: np.ndarray, levels: np.ndarray) -> tuple[float, float]`; `apply_cutpoints(scores: np.ndarray, cuts: tuple[float, float]) -> np.ndarray`

What each set contains:
- **Corpus:** every val (117) and test (135) passage. Test passages also get `health = health_hits(text) > 0`, used for check 6.3.
- **ReadMe++:** 300 sentences, `seed=42`, stratified by `round(float(rating))`. Take equal shares per rating where possible and fill the remainder proportionally.
- **MIRACL:** 100 passages, `seed=42`.
- **Controlled cases:** one of each case per reference item (60 × 4):
  - *positive*: the response is `reference_answer(item)`
  - *swapped*: the response is positive with every occurrence of `correct_option_text` replaced by the alphabetically first wrong option's text. If the correct text doesn't occur in the CoT, the response is that wrong option + `"। "` + the CoT.
  - *unrelated*: the response is `reference_answer` of the next item in sorted id order (wrapping around)
  - *non_answer*: the response is `question_stem(topic_hi)`

  For `reference_agreement` the reference is `reference_answer(item)`. For `consistency` the reference is `positive_text`, standing in for "the high answer".
- **Paraphrase:** a 10% sample of `llm_job_list` (`seed=42`), rebuilt with `PARAPHRASED`.

- [ ] **Step 1: Write the failing tests**

```python
def test_controlled_cases_four_per_item_and_swap_removes_answer(ref2): ...
    # 8 cases for 2 items; swapped response lacks correct_option_text and contains the wrong option text

def test_unrelated_case_uses_a_different_item(ref2): ...

def test_readmepp_sample_is_stratified_and_handles_string_ratings(tmp_readmepp): ...
    # 300 rows, ratings parsed from "2.0"-style strings, every rounded rating 1..6 present

def test_fit_cutpoints_recovers_clean_separation():
    s = np.array([1, 1, 2, 3, 3, 4, 5, 5]); lv = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    assert fit_cutpoints(s, lv) == (2.5, 3.5)
    assert apply_cutpoints(s, (2.5, 3.5)).tolist() == lv.tolist()

def test_fit_cutpoints_tie_breaks_toward_2_5_and_3_5(): ...
    # all-equal scores: every pair ties on macro-F1; returns (2.5, 3.5)

def test_validation_jobs_ids_unique(ref2, toy_clean): ...
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `validation.py`, and extend the `jobs.py` CLI to append `validation_jobs(...)`.** `fit_cutpoints` tries every pair c1 < c2 from {1.5, 2.5, 3.5, 4.5}. The level is 0 if s < c1, 1 if s < c2, otherwise 2. It picks the pair with the highest macro-F1 (compute it with numpy, since sklearn isn't installed). Ties go to the pair with the smallest |c1 − 2.5| + |c2 − 3.5|.
- [ ] **Step 4: Run the tests and confirm they pass.** Then run `python -m literacy_eval.judge.jobs --out data/judge/jobs.jsonl` and check the printed counts against spec §5/§6. The total should be about 5,000. Report the actual number to the user.
- [ ] **Step 5: Commit** (ask the user first). `git commit -m "Add judge validation jobs and cut-point fitting"`. Commit `data/judge/jobs.jsonl` too, so PACE can pull it.

---

### Task 5: PACE runner and SLURM job

**Files:**
- Create: `literacy_eval/judge/run.py`, `scripts/judge_run.sh`
- Test: `tests/test_judge_run.py`

**Interfaces:**
- Consumes: `read_jsonl`, `write_jsonl` (Task 3); `parse_result`, `ParseError` (Task 2).
- Produces:
  - `run(jobs_path: Path, out_path: Path, generate: Callable[[list[str], int], list[str]], judge_model: str, judge_revision: str, max_tokens: int = 512, retry_max_tokens: int = 1024, limit: int | None = None) -> int`. It returns the number of jobs run.
  - It appends rows `{"job_id", "output", "ok": bool, "max_tokens", "judge_model", "judge_revision", "prompt_version", "run_date"}` to `out_path`.
  - `vllm_generate(model_id: str) -> Callable` (PACE only): wraps `vllm.LLM(model_id, dtype="bfloat16")`, `llm.chat([[{"role": "user", "content": p}] ...], SamplingParams(temperature=0, max_tokens=n))`.
  - CLI: `python -m literacy_eval.judge.run --jobs J --out O [--limit N] [--match SUBSTRING]`. `--match` keeps only the jobs whose `job_id` contains SUBSTRING. The CLI reads `judge_revision` from `huggingface_hub.HfApi().model_info(model_id).sha`.

- [ ] **Step 1: Write the failing tests**, with a fake `generate` that records `(prompts, max_tokens)` and returns canned text.

```python
def test_runs_all_then_resumes_without_rerunning(tmp_path, jobs3): ...
    # first run → 3 rows; second run → generate never called

def test_unparsable_output_is_retried_once_with_more_tokens(tmp_path, jobs3): ...
    # fake returns "Feedback: cut" at 512, "Feedback: ok [RESULT] 3" at 1024
    # → rows for that job: ok False (512) then ok True (1024); calls == [512, 1024]

def test_still_unparsable_after_retry_is_kept_as_not_ok_and_not_retried_forever(tmp_path, jobs3): ...

def test_truncated_last_line_job_is_redone(tmp_path, jobs3):          # Review Focus 3
    # pre-write 1 good row + a half row for job 2 → only jobs 2 and 3 are generated

def test_provenance_fields_present(tmp_path, jobs3): ...
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `run.py`.**
  - A job counts as done if any row for it has `ok=True`, or if it has a row at `retry_max_tokens`.
  - Pass 1 sends every not-done job at `max_tokens` in a single `generate` call. Pass 2 resends the jobs that failed to parse, at `retry_max_tokens`.
  - Rows are appended and flushed in chunks of 256, so a killed job keeps most of its work.
- [ ] **Step 4: Write `scripts/judge_run.sh`**, modelled on `scripts/register_score.sh`.
  - Use the same SBATCH header, with `--job-name=judge`, `--time=06:00:00`, `--mem=64G`, `--gres=gpu:l40s:1`.
  - Set the same scratch env exports, and set `HF_HUB_CACHE` under `HF_HOME`.
  - Create the venv `$SCRATCH/venv-judge` if it's missing (`python -m venv`, then `pip install vllm pandas huggingface_hub`). Print `pip show vllm | head -2` so the version lands in the log.
  - Run `python -m literacy_eval.judge.run --jobs $SCRATCH/judge/jobs.jsonl --out $SCRATCH/judge/results.jsonl "$@"`.
  - The header comment gives the usage: upload with `scp data/judge/jobs.jsonl pace:~/scratch/judge/`, run `sbatch scripts/judge_run.sh [--limit N]`, download with `scp pace:~/scratch/judge/results.jsonl data/judge/`.
- [ ] **Step 5: Run the tests and confirm they pass.** Then `bash -n scripts/judge_run.sh` should report no syntax errors, and `grep -n "/home/hice1/gchiu6/scratch" scripts/judge_run.sh` should show that every output, cache and venv path is under scratch.
- [ ] **Step 6: Commit** (ask the user first; use the school identity, since PACE pulls this). `git -c user.name=gchiu6 -c user.email=gchiu6@gatech.edu commit -m "Add vLLM judge runner and SLURM job"`

---

### Task 6: Collect results and reference audit

**Files:**
- Create: `literacy_eval/judge/collect.py`, `literacy_eval/judge/reference_audit.py`
- Test: `tests/test_judge_collect.py`

**Interfaces:**
- Consumes: `read_jsonl` (Task 3), `parse_result` (Task 2).
- Produces:
  - `collect(jobs_path: Path, results_path: Path) -> pd.DataFrame`. It returns one row per job id, holding all job metadata (everything except `prompt`) plus `feedback`, `score` (Int64, NA on failure), `judge_error` (str, empty when OK), and the provenance fields. The latest `ok` row wins; otherwise the latest row. Jobs with no result get `judge_error="missing"`.
  - CLI writes `literacy_eval/analysis_outputs/judge_scores.csv` and prints the error and missing counts per task.
  - `AUDIT_CSV = literacy_eval/reference/reference_audit.csv`
  - `export_audit(reference: dict[str, dict], path: Path = AUDIT_CSV) -> None`. Columns: `item_id, system, subject, question_hi, correct_option_hi, cot_hi, question_en, correct_option_en, cot_en, ref_suspect, note`. The English columns and `ref_suspect` are left blank. It refuses to overwrite a file whose `ref_suspect` column has any value filled in.
  - `load_suspect(path: Path = AUDIT_CSV) -> set[str] | None`. It returns `None` if the file is missing or no row is filled, and otherwise the ids where `ref_suspect` lowercased ∈ {"y", "yes"}.

- [ ] **Step 1: Write the failing tests.** Cover these cases:
  - the latest OK row wins over an earlier failed row
  - a missing result gives `"missing"`
  - an unparsable output keeps its `judge_error` text
  - the scores dtype is Int64
  - `export_audit` won't overwrite filled audit values
  - `load_suspect` returns `None` for a blank file and `{"i2"}` when `ref_suspect` is `Y` on i2
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement both modules.** CLI for the audit: `python -m literacy_eval.judge.reference_audit --export`. Its help text tells the owner to open the file in Google Sheets and put `=GOOGLETRANSLATE(D2,"hi","en")` in `question_en`, and the same for the other two columns.
- [ ] **Step 4: Run the tests and confirm they pass.** Then run `--export` for real and confirm 60 rows.
- [ ] **Step 5: Commit** (ask the user first). `git commit -m "Collect judge results; export reference audit sheet"`

---

### Task 7: Weighted kappa

**Files:**
- Modify: `literacy_eval/analysis/stats.py` (add a function after `bootstrap_ci`)
- Test: `tests/test_analysis_stats.py` (append)

**Interfaces:**
- Produces: `weighted_kappa(a: np.ndarray, b: np.ndarray, n_levels: int = 3, weights: str = "quadratic") -> float`

- [ ] **Step 1: Write the failing tests.**
  - Identical ratings → 1.0.
  - `a=[0,1,2,0,1,2], b=[0,1,2,1,2,0]` → `0.25` with quadratic weights (observed weighted disagreement 0.25, expected 1/3), to 1e-9.
  - When every rating in `a` and `b` is the same constant (zero expected disagreement), return `nan` rather than raising.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement it** from the confusion matrix, using weights w_ij = (i−j)² / (n−1)² for quadratic or |i−j| / (n−1) for linear. Use numpy only.
- [ ] **Step 4: Run them and confirm they pass,** with the existing stats tests still green.
- [ ] **Step 5: Commit** (ask the user first). `git commit -m "Add weighted kappa"`

---

### Task 8: Report: validation and results tables

**Files:**
- Create: `literacy_eval/judge/report.py`
- Test: `tests/test_judge_report.py`

**Interfaces:**
- Consumes:
  - `judge_scores.csv` (Task 6)
  - `fit_cutpoints`, `apply_cutpoints`, `EXPECTED_BAND`, `LABEL_LEVEL` (Task 4)
  - `load_suspect` (Task 6)
  - `stats.page_trend`, `stats.wilcoxon_paired`, `stats.holm`, `stats.bootstrap_ci`, `stats.wilson_ci`, `stats.MIN_PAIRED_N`, `stats.weighted_kappa`
  - `register.analyze.spearman_ci`, `register.analyze.per_response`, `register.analyze.md`, `register.analyze.fmt_p`
  - BERT scores from `register_runs/register_corpus/indicbert_seed42/{test_scores,readmepp_hi_scores,llm_window_scores}.jsonl`
  - `clean.csv` (for `n_hindi_words`)
- Produces: `build_report(scores: pd.DataFrame, ...) -> str` (markdown), plus these files:
  - `literacy_eval/analysis_outputs/judge_results.md`
  - `tables/judge_validation.csv`, `tables/judge_llm.csv`, `tables/judge_paired.csv`, `tables/judge_harm.csv`, `tables/judge_figure_data.csv`

Sections of `judge_results.md`, in order. Each section starts with one sentence on how to read it.

| § | Content | Spec |
|---|---|---|
| A | Run summary: jobs, errors and missing per task; judge model, revision, prompt version | §7 |
| B1 | Cut-points fitted on corpus:val | §5 |
| B2 | Corpus:test (run once): accuracy, quadratic κ, 3×3 confusion matrix, mean score per class. The same metrics for BERT from `test_scores.jsonl` | §6.1 |
| B3 | ReadMe++: Spearman ρ with 95% CI, judge vs BERT, and mean score per rating | §6.2 |
| B4 | Topic confound: MIRACL score distribution; Class-5 health vs non-health test passages | §6.3 |
| B5 | Length confound: Spearman(score, `n_hindi_words`) on LLM outputs for complexity and reference_agreement, and within each corpus class. Flag \|ρ\| > 0.5 | §6.4 |
| B6 | Controlled cases: % in `EXPECTED_BAND` per task × case, with ✅ if ≥ 80% | §6.5 |
| B7 | Prompt sensitivity: % of paraphrase jobs within ±1 of the original | §6.7 |
| B8 | Judge vs BERT on LLM outputs: quadratic κ, confusion matrix | §6.6 |
| C1 | Main table (one row per model × level): usable n, judge complexity mean, judge level match %, term-explanation mean (low/mid), reference agree % (≥4) / contradict % (≤2) for all items and with suspect items excluded, consistency mean and % ≥ 4 with bootstrap CI (low/mid), harm count (≤2) | §8 |
| C2 | Paired tests on complexity: Page's L on items usable at all 3 levels, Wilcoxon low vs high, rank-biserial r, Holm across models. Below `MIN_PAIRED_N`, print "n too small" | §8 |
| C3 | Harm list: every score ≤ 2 with model, level, item and English feedback | §5 |

If `load_suspect()` returns `None`, the "excl. suspect" columns read "audit pending".

The figure data is one row per model: `complexity_shift` (median of high − low complexity over paired items), BERT `expected_level` shift, and `consistency_rate` (low-high % ≥ 4). Drawing the figure is out of scope here (matplotlib isn't installed). Mention that in §C.

- [ ] **Step 1: Write the failing tests**, on a small synthetic `scores` DataFrame built in the test.
  - B6 band logic: a positive scored 5 is in band, a swapped scored 3 is out of band.
  - Match % uses the fitted cut-points and the mapping low→0, mid→1, high→2.
  - "n too small" appears when fewer than 10 paired items exist (Review Focus 5).
  - The "audit pending" text appears when `suspect is None`.
  - Rows with `judge_error` are excluded from every metric and counted in §A.
  - Paraphrase (`para:`) jobs and validation jobs never appear in C1.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement `report.py`.** CLI: `python -m literacy_eval.judge.report`.
- [ ] **Step 4: Run them and confirm they pass.** Then run the full `$PYTEST` and confirm there are no new failures.
- [ ] **Step 5: Commit** (ask the user first). `git commit -m "Add judge validation and results report"`

---

### Task 9: Pilot, freeze, full run (operational, with the user)

This task runs on PACE and with the user, so it has no unit tests. Each step has a checkable result.

- [ ] **Step 1: Pilot run.** Run `python -m literacy_eval.judge.jobs --pilot 5 --out data/judge/pilot_jobs.jsonl`, upload it as `~/scratch/judge/jobs.jsonl`, and run `sbatch scripts/judge_run.sh`. Before the full run, delete the pilot `results.jsonl` on scratch, or move it aside. Download the results and run `collect`. Pass: zero `judge_error` rows, and the log shows the vLLM version and the GPU name.
- [ ] **Step 2: Read the pilot feedback** in English together with the user, one example per task. Check that the feedback refers to the rubric (complexity mentions words and sentences; agreement mentions the reference claim).
- [ ] **Step 3: Gate check** on controlled cases and corpus:val only. Upload the full `jobs.jsonl`, run `sbatch scripts/judge_run.sh --match controlled:` and `--match corpus:val`, then `collect` and `report`, and read §B6 and §B1. If any case type is below 80%, revise that rubric in `rubrics.py`, bump `PROMPT_VERSION` to `v2`, rebuild the jobs, and repeat this step only. Revisions may never look at corpus:test, ReadMe++, MIRACL or the LLM outputs.
- [ ] **Step 4: Freeze.** Commit `rubrics.py` with the final `PROMPT_VERSION` and record it in the design doc's status line.
- [ ] **Step 5: Full run.** Run `sbatch scripts/judge_run.sh` on all jobs, resubmitting until `collect` reports 0 missing. Then run `report`. Pass: §A shows < 1% `judge_error` per task.
- [ ] **Step 6: Reference audit.** The user fills `reference_audit.csv` (Task 6 instructions). Rerun `report` so the "excl. suspect" columns fill in.
- [ ] **Step 7: Docs.** Add a "Revision 4 (LLM judges)" section to `himed_trad_evaluation_plan.md`. It should point to `judge_results.md`, give 3–5 headline numbers, and say that `analysis/judge.py` (Gemini) is superseded and was never run. Commit `judge_results.md`, the tables and `judge_scores.csv` (ask first).
