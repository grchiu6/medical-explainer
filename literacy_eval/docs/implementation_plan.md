# Literacy-Level Baseline Evaluation for Hindi Medical Explanations

## Problem & Motivation

Your research question is: **How can LLMs adapt the linguistic complexity of Hindi medical explanations to different health-literacy levels while preserving medical accuracy?**

Your current baseline evaluation doesn't measure anything related to this. It measures:
- **IndicQA:** extractive QA accuracy (BLEU/ROUGE against short reference answers) — tests general Hindi comprehension, not medical explanation or register adaptation
- **HiMed MCQ:** letter-matching accuracy — tests medical *knowledge* retrieval, not the ability to *explain* medical concepts at a given complexity level

Neither of these captures the core phenomenon you're studying: **can a model generate a medical explanation in Hindi that is linguistically appropriate for a specified literacy level?**

## What the New Baseline Should Actually Measure

The right "baseline" for your research question is: **given a medical topic and a target literacy level, can the model produce a Hindi explanation that matches the requested register?**

This is a *generation* task, not a QA-accuracy task. You're not checking if the model knows the right answer — you're checking if it can *explain* a medical concept *at the right linguistic level*.

> [!IMPORTANT]
> You correctly identified that **analyzing whether the output is appropriate/accurate is out of scope for now**. So the goal of this baseline is purely to **collect model outputs** — prompt models at different literacy levels on medical topics and save what they produce. Evaluation of those outputs (readability scoring, human judgment, etc.) comes later.

## My Input on Your Approach

Your instinct — "give the model a literacy level and a medical question, see if it can produce an explanation at that level" — is exactly right. Here's how I'd refine it:

### 1. Don't use a dataset "based on" a literacy level — use the **prompt** to specify the level

You don't need different datasets per literacy level. You need **one set of medical topics/questions** and then prompt the model multiple times with different literacy-level instructions. The variable you're testing is the *prompt instruction*, not the *input data*.

### 2. Use the HiMed questions you already have as your medical topic source

HiMed-Trad and HiMed-West already contain Hindi medical questions across a wide range of subjects (etiology, pharmacology, anatomy, etc.). You can repurpose the **question stems** (ignoring the MCQ options) as medical topics for the model to explain. This saves you from needing a new dataset.

### 3. Define 3 literacy levels with clear Hindi-language-appropriate register descriptions

| Level ID | Label | Description in Prompt | Target Audience |
|----------|-------|----------------------|-----------------|
| `low` | 5th-grade / बुनियादी | Simple everyday Hindi, no medical jargon, short sentences | Someone with primary school education |
| `mid` | 10th-grade / मध्यम | Some medical terms with definitions, moderate sentence complexity | High-school educated person |
| `high` | Professional / विशेषज्ञ | Technical medical terminology, complex sentence structure, assumes medical knowledge | Healthcare professional or medical student |

### 4. The experimental design: a 3×3 (or 3×N) matrix

For each **(model × literacy level × medical topic)** combination, you collect one output. This gives you a matrix like:

| | Low (5th grade) | Mid (10th grade) | High (Professional) |
|---|---|---|---|
| **Gemini** | output | output | output |
| **Airavata** | output | output | output |
| **Apollo** | output | output | output |

This is your baseline data. Later analysis will measure: do the outputs actually differ across levels? Do they preserve medical content? Which model adapts best?

---

## User Review Required

> [!IMPORTANT]
> **Literacy level definitions:** The 3 levels (low/mid/high) above are a starting point. Do you have specific levels in mind from your literature review? Some health-literacy frameworks use 5+ levels. I'd recommend starting with 3 for manageable output volume.

> [!IMPORTANT]
> **Medical topic source:** I propose extracting question stems from HiMed (which you already have). Alternative: use a curated list of ~20-30 common medical conditions (diabetes/मधुमेह, hypertension/उच्च रक्तचाप, tuberculosis/क्षय रोग, etc.). The curated list would be simpler and more focused. Which do you prefer?

> [!IMPORTANT]
> **Sample size for this baseline:** Running 3 models × 3 levels × N topics produces 9N outputs. With N=30 topics, that's 270 API calls. With N=50, that's 450. What feels right for an initial collection? I'd recommend **30 topics** as a good starting point.

> [!WARNING]
> **Models that require GPU (Airavata, Apollo):** These still need HPC cluster access. The new pipeline will work identically to the old one for model integration — the only thing changing is the prompt and the output schema. Are you running Gemini-only for now, or do you have cluster access?

## Open Questions

1. **Language of the prompt instructions:** Should the literacy-level instruction be in English (e.g., "Explain this medical concept as if to a 5th grader") or Hindi (e.g., "इस चिकित्सा अवधारणा को ऐसे समझाएं जैसे आप किसी पांचवीं कक्षा के बच्चे को समझा रहे हों")? I'd recommend **English instructions** (consistent with NLP research convention and your existing pipeline) with the expected **output in Hindi**.

2. **Output format:** Should the model's response be completely free-form, or should we ask it to structure the explanation (e.g., "explain in 3-5 sentences")? Free-form gives more natural data; structured makes later analysis easier. I'd recommend **free-form** for the baseline — you want to see what the model naturally does.

3. **Do you want to keep the old evaluation pipeline intact** (as a separate thing) or replace it? I recommend keeping it — it still demonstrates general Hindi QA capability, which is useful context in a paper even if it's not the core contribution.

---

## Proposed Changes

### New Evaluation Module (new directory alongside existing pipeline)

The new code lives in a new subdirectory to keep the existing baseline intact.

---

#### [NEW] [literacy_eval/](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/)

New package directory for the literacy-level evaluation pipeline.

#### [NEW] [`__init__.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/__init__.py)

Empty init file to make `literacy_eval/` a Python package.

#### [NEW] [`prompts.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/prompts.py)

Core prompt engineering module. Contains:
- `LITERACY_LEVELS` dict mapping level IDs (`low`, `mid`, `high`) to detailed prompt instructions
- `build_explanation_prompt(medical_topic: str, literacy_level: str) -> str` — constructs the full prompt that instructs the model to explain a medical topic at the specified literacy level
- Prompt template designed so:
  - Instructions are in English (for consistent model behavior)
  - The model is told to respond **entirely in Hindi**
  - The literacy level is described with concrete linguistic constraints (vocabulary complexity, sentence length, jargon usage)

Example prompt structure:
```
You are a medical educator. Explain the following medical concept in Hindi.

Target audience: A person with only primary school education (5th grade level).
Language requirements:
- Use only simple, everyday Hindi words (बोलचाल की भाषा)
- Avoid all medical jargon and technical terms
- Use short, simple sentences (5-10 words each)
- If you must mention a medical term, immediately explain it in simple words
- Use examples from daily life to explain concepts

Medical topic: {topic}

Explanation in Hindi:
```

#### [NEW] [`topics.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/topics.py)

Curated list of 30 medical topics as a Python data structure. Each topic is a dict with:
- `id`: short identifier (e.g., `diabetes`)
- `topic_hi`: the topic/question in Hindi (e.g., `मधुमेह (डायबिटीज़) क्या है और यह कैसे होता है?`)
- `topic_en`: English translation for reference (not sent to model)
- `category`: broad medical category (e.g., `endocrine`, `infectious`, `cardiovascular`)

Topics will cover common conditions relevant to Indian public health: diabetes, hypertension, tuberculosis, dengue, malaria, anemia, asthma, diarrhea, COVID-19, typhoid, jaundice, cholera, pneumonia, kidney stones, arthritis, thyroid disorders, heart attack, stroke, HIV/AIDS, hepatitis, etc.

#### [NEW] [`run_eval.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/run_eval.py)

Main evaluation runner (analogous to the existing `evaluate.py`). Features:
- CLI: `python -m literacy_eval.run_eval --model gemini --level low --sample 5`
- Also supports `--level all` to run all 3 levels sequentially
- Also supports `--topic diabetes` to run a single topic (for testing)
- Loads topics from `topics.py`
- Builds prompts using `prompts.py`
- Calls the **existing model wrappers** from `models/` (reuses `models/gemini.py`, `models/airavata.py`, etc.)
- Saves outputs to `literacy_eval/outputs/{model}_{level}_raw.csv`
- CSV schema: `topic_id, topic_hi, literacy_level, model_response`
- Incremental saving + resume support (same pattern as existing `evaluate.py`)

#### [NEW] [`outputs/`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/outputs/)

Directory for literacy evaluation outputs. Example files after running:
- `gemini_low_raw.csv`
- `gemini_mid_raw.csv`
- `gemini_high_raw.csv`
- `airavata_low_raw.csv`
- etc.

---

### Minimal Changes to Existing Code

#### [MODIFY] [`models/base.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/models/base.py)

No changes needed. The new pipeline builds its own prompts in `literacy_eval/prompts.py` and passes the finished prompt string directly to the model's `get_answer(prompt)` function. The existing `build_prompt()` and `build_mcq_prompt()` remain untouched for the old pipeline.

#### [MODIFY] [`evaluate.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/evaluate.py)

No changes needed. The new pipeline has its own runner (`literacy_eval/run_eval.py`).

---

## Verification Plan

### Automated Tests

```bash
# Test that the prompt builder produces correct prompts for each level
python -m pytest literacy_eval/tests/test_prompts.py -v

# Test that topics load correctly
python -m pytest literacy_eval/tests/test_topics.py -v

# Dry-run: generate prompts for 2 topics × 3 levels without calling any model
python -m literacy_eval.run_eval --model gemini --level all --sample 2 --dry-run
```

### Manual Verification

1. Run `--sample 1 --level low` with Gemini and manually inspect the output — does it look like a simple Hindi explanation?
2. Run `--sample 1 --level high` with Gemini and compare — is the language noticeably more technical?
3. Verify the CSV output schema has the right columns and data
4. Verify resume support by interrupting a run and restarting

### Sanity Check

After collecting outputs for at least one model across all 3 levels on 5+ topics, visually compare the outputs across levels for the same topic. If the model is doing its job at all, you should see obvious differences in vocabulary complexity and sentence structure. This visual comparison will be the first informal validation that the baseline setup is working.

---

## What This Gives You (Research-Wise)

After running this pipeline, you'll have a structured dataset of:
- **N topics × 3 levels × M models** = Hindi medical explanations at varying requested complexity levels

This is your **baseline collection**. It answers: "What do current off-the-shelf LLMs produce when asked to adapt Hindi medical explanations to different literacy levels?"

Future analysis steps (out of scope now, but this data enables them):
1. **Readability analysis:** Apply Hindi readability metrics to quantify whether models actually change linguistic complexity across levels
2. **Medical accuracy check:** Have medical experts verify factual correctness is preserved across simplification levels  
3. **Fine-tuning comparison:** Fine-tune a model specifically for this task and compare its outputs to these baselines
4. **Human evaluation:** Have native Hindi speakers at different education levels rate the explanations for understandability
