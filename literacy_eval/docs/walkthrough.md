# Walkthrough: Literacy-Level Baseline Evaluation Pipeline

## What Was Built

A new `literacy_eval/` module that collects Hindi medical explanations from LLMs at 3 different literacy levels — directly testing the core research question about register adaptation.

### New Files

| File | Purpose |
|------|---------|
| [`literacy_eval/__init__.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/__init__.py) | Package init |
| [`literacy_eval/__main__.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/__main__.py) | Convenience entry point (`python -m literacy_eval`) |
| [`literacy_eval/topics.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/topics.py) | 30 curated medical topics covering 12 categories |
| [`literacy_eval/prompts.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/prompts.py) | 3 literacy levels with Hindi instructions + prompt builder |
| [`literacy_eval/run_eval.py`](file:///Users/gracechiu/Desktop/Research/Baseline%20Evaluation/literacy_eval/run_eval.py) | Main runner with CLI, incremental save, and resume support |

### No changes to existing files
The old baseline pipeline is completely untouched. The new module reuses the existing model wrappers (`models/airavata.py`, `models/apollo.py`) via dynamic import.

---

## The 3 Literacy Levels

| Level | Target Audience | Key Linguistic Constraints |
|-------|----------------|---------------------------|
| **`low`** (कक्षा 5) | Primary-school-educated, village setting | No jargon, no English words, short sentences (5-10 words), everyday examples |
| **`mid`** (कक्षा 10) | High-school-educated | Some medical terms with definitions, moderate sentences, structured (cause→symptom→treatment) |
| **`high`** (विशेषज्ञ) | Medical student / health worker | Technical terminology, pathophysiology, clinical criteria, complex sentences |

All instructions are in **Hindi** to reflect the real-world use case.

---

## How to Run

```bash
# Quick test — dry run shows prompts without calling any model
python3 -m literacy_eval.run_eval --model airavata --level low --sample 2 --dry-run

# Run one topic at one level (good for verifying model output)
python3 -m literacy_eval.run_eval --model airavata --level low --topic diabetes

# Full run — one model, one level, all 30 topics
python3 -m literacy_eval.run_eval --model airavata --level low

# Full run — one model, all 3 levels (90 total outputs)
python3 -m literacy_eval.run_eval --model airavata --level all

# Both models, all levels (run separately, 180 total outputs)
python3 -m literacy_eval.run_eval --model airavata --level all
python3 -m literacy_eval.run_eval --model apollo --level all
```

### Output Location

Results go to `literacy_eval/outputs/`:
- `airavata_low_raw.csv`
- `airavata_mid_raw.csv`
- `airavata_high_raw.csv`
- `apollo_low_raw.csv`
- etc.

### CSV Schema

| Column | Description |
|--------|-------------|
| `topic_id` | Short identifier (e.g., `diabetes`) |
| `topic_hi` | Hindi question that was asked |
| `topic_en` | English translation (reference only) |
| `category` | Medical category (e.g., `endocrine`) |
| `literacy_level` | Which level was requested (`low`/`mid`/`high`) |
| `prompt` | The full prompt sent to the model |
| `model_response` | What the model returned |

---

## Verification

All tested and verified:
- ✅ Dry run mode prints correct prompts for all 3 levels
- ✅ `--level all` iterates through high → low → mid correctly
- ✅ `--topic` flag filters to a single topic
- ✅ `--sample` flag limits the number of topics
- ✅ Resume support skips already-answered topics
- ✅ Help text is clear and complete

---

## What's Next

1. **Run the actual evaluation** on the HPC cluster with Airavata and Apollo
2. **Visually inspect** outputs — do the levels actually look different for the same topic?
3. **Later:** Apply readability metrics / human evaluation to quantify register adaptation quality
4. **Later:** Fine-tune a model specifically for this task and compare against these baselines
