# HiMed Question Extraction — Adding Dataset-Derived Topics

## Background

The current baseline uses 30 curated medical topics (open-ended questions like "What is diabetes?"). These are useful for evaluating **linguistic complexity adaptation** but have no ground-truth answers, making medical accuracy evaluation harder.

Adding a parallel track with questions extracted from the HiMed dataset solves this: each question comes with a correct answer and chain-of-thought reasoning, enabling automatic accuracy scoring alongside complexity analysis.

**Both tracks run through the same pipeline** — same models, same 3 literacy levels, same `run_eval.py`. The only difference is the topic source.

---

## HiMed Dataset Structure (What We Have)

Two JSON files in `data/himed/`:

### `HiMed-West_Bench.json` — 1,784 Western medicine MCQs
```
Keys: question_number, question, options, answer, source
Sources: GPQA-med (249), MMLU-Pro-Biology (717), MMLU-Pro-Health (818)
Format: Hindi MCQs with 4 options (A/B/C/D)
No subject/category field
```

### `HiMed-Trad_Bench.json` — 6,010 Traditional medicine MCQs
```
Keys: question_number, subject, type, question, answer, cot, system, source
Subjects: etiology (1137), medical knowledge (1638), treatment (1902), 
          diagnosis (1037), prognosis (296)
Systems: Ayurveda, Homoeopathy, Naturopathy, Siddha, Unani, Yoga (1000 each), 
         Sowa-Rigpa (10)
Format: Hindi MCQs with 5 options (A-E), each has chain-of-thought (cot)
```

---

## Open Questions

> [!IMPORTANT]
> **Which benchmark file(s) should we use?**
> - **West_Bench only** — Western medicine aligns best with the curated topics (diabetes, TB, etc.) and is what most readers expect from "medical accuracy."
> - **Trad_Bench only** — Traditional Indian medicine (Ayurveda, Unani, etc.) is unique and culturally relevant, but very different from the curated topics.
> - **Both** — Most comprehensive, but doubles the work and the two domains test very different things.
> - My recommendation: **West_Bench only for now** (MMLU-Pro-Health subset specifically — most clinically relevant). Trad_Bench can be added later as a separate experiment if needed.

> [!IMPORTANT]
> **How many questions to extract?**
> - The curated track has 30 topics. For comparability, I'd recommend **30-50 HiMed questions** sampled to cover diverse medical subjects.
> - Alternatively, use **all** MMLU-Pro-Health questions (818) for a large-scale evaluation — but that's 818 × 3 levels × 5 models = 12,270 generations, which is substantial cluster time.
> - My recommendation: **Start with 50 questions**, stratified across sources, to keep the initial run manageable. Scale up later if useful.

> [!IMPORTANT]
> **What prompt format for MCQ questions?**
> The curated topics use an open-ended "explain this" format. For HiMed MCQs, two options:
> - **Option A — Keep open-ended:** Strip the MCQ options and ask the model to *explain* the topic (same as curated). Then score accuracy by checking if the explanation aligns with the correct answer's chain-of-thought. This keeps the two tracks parallel/comparable.
> - **Option B — MCQ format:** Present the full MCQ with options and ask the model to select an answer. This gives cleaner accuracy scoring (exact match) but tests a different capability (MCQ answering vs. explanation generation).
> - **Option C — Both:** Run each question in both formats. More data, more analysis options, but more cluster time.
> - My recommendation: **Option A (open-ended)** for the explanation quality study. If you later want MCQ accuracy, add Option B as a separate experiment.

---

## Proposed Changes

### New File: `literacy_eval/topics_himed.py`

Extracts and formats question stems from the HiMed dataset files.

- Loads `data/himed/HiMed-West_Bench.json`
- Filters to MMLU-Pro-Health source (most clinically relevant)
- Samples N questions (default 50) with deterministic seed for reproducibility
- Converts each MCQ into an open-ended explanation prompt:
  - Original: "निम्नलिखित में से किसके द्वारा पोलियो का उन्मूलन किया जा सकता है?" + options A-D
  - Converted: "पोलियो का उन्मूलन कैसे किया जा सकता है? इसके बारे में विस्तार से समझाएं।"
- Preserves the original question, options, correct answer, and source as metadata for later accuracy scoring
- Returns same dict format as `topics.py`: `{id, topic_hi, topic_en, category}` plus extra fields `{original_question, options, correct_answer, source}`

### Modify: `literacy_eval/run_eval.py`

Add a `--topic-source` CLI argument:
```
--topic-source curated    (default, current behavior — uses topics.py)
--topic-source himed      (new — uses topics_himed.py)
```

The rest of the pipeline stays identical. Output CSVs will be named differently:
- Curated: `airavata_low_raw.csv` (unchanged)
- HiMed: `airavata_low_himed_raw.csv`

### New File: `literacy_eval/himed_extractor.py`

Utility script to preview extracted questions before running the full eval:
```bash
python -m literacy_eval.himed_extractor --count 5 --source MMLU-Pro-Health
```

### Output Schema Extension

HiMed-sourced CSVs will include extra columns:
```
topic_id, topic_hi, topic_en, category, literacy_level, model, model_response,
original_question, options_json, correct_answer, source
```

This lets the analysis phase compare the model's free-form explanation against the known correct answer.

---

## What Does NOT Change

- Curated 30-topic pipeline — completely untouched, runs as before
- Model wrappers (`models/*.py`) — no changes needed
- Prompt templates (`prompts.py`) — same `build_explanation_prompt()` function
- SLURM scripts — same `literacy_full_run.sh` with a new `--topic-source himed` flag

---

## Verification Plan

### Automated
```bash
# Dry run with HiMed topics
python -m literacy_eval.run_eval --model airavata --level low \
  --topic-source himed --dry-run

# Smoke test on cluster (1 topic)
python -m literacy_eval.run_eval --model airavata --level mid \
  --topic-source himed --topic 0 --output-dir $OUTPUT_DIR
```

### Manual
- Inspect extracted questions to confirm they're coherent and medically relevant
- Compare a few model responses against the correct answers to sanity-check
