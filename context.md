# Research Context: Hindi Health-Literacy-Adapted Medical Explanations

## Research Question

**How can LLMs adapt the linguistic complexity of Hindi medical explanations to different health-literacy levels while preserving medical accuracy?**

## Current Phase: Baseline Evaluation

Collecting and evaluating how current off-the-shelf models perform at this task *before* any fine-tuning. The baseline answers: "What do existing LLMs produce when asked to explain Hindi medical topics at varying literacy levels?"

---

## Project Architecture

### Two Evaluation Pipelines

| Pipeline | Purpose | Location | Status |
|----------|---------|----------|--------|
| **Original QA pipeline** | General Hindi QA accuracy (IndicQA) + MCQ accuracy (HiMed) | `evaluate.py`, `score.py`, `score_mcq.py` | Complete — Gemini, Airavata, Apollo evaluated |
| **Literacy-level pipeline** | Register-adapted Hindi medical explanations at 3 literacy levels | `literacy_eval/` | Outputs collected, analysis pending |

The original pipeline tests *knowledge retrieval* (can the model pick the right answer?). The literacy-level pipeline tests *explanation generation* (can the model explain a topic at the right linguistic register?) — this is the core research contribution.

### Literacy Levels

| Level | Target Audience | Key Constraints |
|-------|----------------|-----------------|
| `low` (कक्षा 5) | Primary-school-educated, village setting | No jargon, no English, short sentences, everyday examples |
| `mid` (कक्षा 10) | High-school-educated | Some medical terms with definitions, structured explanation |
| `high` (विशेषज्ञ) | Medical student / health worker | Technical terminology, pathophysiology, clinical criteria |

Prompt instructions are in Hindi (design decision: reflects real-world usage scenario).

### Models Evaluated

| Model | Type | Specialization | GPU Req |
|-------|------|---------------|---------|
| **Airavata** (ai4bharat) | Instruction-tuned | Hindi language | ~14GB |
| **Apollo-7B** (FreedomIntelligence) | Instruction-tuned | Medical, 6 languages | ~18GB |
| **HiMed-8B** (FreedomIntelligence) | Instruction-tuned (Llama 3.1) | Hindi + Medical | ~16GB |
| **Nanda-10B** (MBZUAI) | Instruction-tuned (Llama 3) | Hindi bilingual | ~20GB |
| **MMed-Llama-3-8B** (Henrychur) | Base model (NOT instruction-tuned) | Medical, 6 languages | ~16GB |

Key research contrasts:
- Hindi-specialist (Airavata, Nanda) vs. medical-specialist (Apollo, HiMed, MMed-Llama)
- Instruction-tuned (all except MMed-Llama) vs. base model (MMed-Llama) — isolates the effect of instruction tuning
- HiMed-8B is the strongest candidate (combines Hindi + medical specialization)

### Topic Sources

| Source | Count | Description |
|--------|-------|-------------|
| **Curated** (`topics.py`) | 30 | Hand-picked common Indian public health conditions across 12 medical categories |
| **HiMed** (`topics_himed.py`) | 50 | MCQ stems from MMLU-Pro-Health (HiMed-West_Bench), converted to open-ended explanation topics |

The curated track is what has been evaluated so far. The HiMed track is implemented but not yet run.

### Data Collected

- **450 responses** (5 models × 3 levels × 30 curated topics) — with Nanda high at 21/30
- All outputs in `literacy_eval/outputs/{model}_{level}_raw.csv`
- No HiMed-track outputs collected yet

### Known Issues from Initial Analysis (documented in `literacy_eval/docs/output_analysis.md`)

- **Airavata**: Prompt echo / repetition at low level (12 near-failures at low vs 3 at high)
- **Apollo**: Repetition loops in 16-22/30 responses per level (greedy decoding issue)
- **Neither early model showed clear register adaptation** — a key baseline finding

---

## Codebase Structure

```
├── evaluate.py              # Original QA pipeline runner
├── score.py                 # BLEU + ROUGE-L scoring (extractive QA)
├── score_mcq.py             # MCQ accuracy scoring
├── results_table.py         # Aggregate scored CSVs into comparison table
├── models/
│   ├── base.py              # Shared prompt templates
│   ├── gemini.py            # Gemini API wrapper
│   ├── airavata.py          # Airavata local inference
│   ├── apollo.py            # Apollo-7B local inference
│   ├── himed.py             # HiMed-8B local inference
│   ├── nanda.py             # Nanda-10B local inference
│   └── mmed_llama.py        # MMed-Llama base model (few-shot)
├── data/
│   ├── loader.py            # IndicQA dataset loader
│   ├── himed_loader.py      # HiMed MCQ dataset loader
│   ├── indicqa/             # IndicQA Hindi JSON
│   └── himed/               # HiMed benchmark JSONs
├── literacy_eval/
│   ├── prompts.py           # 3 literacy levels + prompt builder
│   ├── topics.py            # 30 curated medical topics
│   ├── topics_himed.py      # HiMed question extraction (50 MMLU-Pro-Health)
│   ├── run_eval.py          # Main runner (CLI, resume support)
│   ├── outputs/             # Collected model responses
│   └── docs/                # Design docs, analysis notes
├── scripts/
│   ├── literacy_full_run.sh     # SLURM full evaluation job
│   ├── literacy_smoke_test.sh   # SLURM quick verification
│   ├── run_local_model.sh       # Local model runner
│   └── submit_all.sh            # Submit all SLURM jobs
├── outputs/                 # Original pipeline outputs
└── tests/                   # Test suite
```

## Infrastructure

- **HPC cluster**: Georgia Tech PACE cluster with SLURM, GPU nodes (A100/V100)
- **API models**: Gemini via Google GenAI API (excluded from literacy eval by design)
- **Local models**: Run on cluster GPUs via HuggingFace transformers
- Model weights cached at `$SCRATCH/hf_cache` on the cluster

---

## Open Questions (Current)

1. **Topic sourcing**: Should the baseline use curated topics, dataset-derived topics (HiMed), or both? (See `topics_himed.py` — already implemented but not run)
2. **Automated evaluation metrics**: How to measure register adaptation and medical accuracy without human reviewers?
3. **Nanda completion**: 9 responses still missing from nanda_high_raw.csv
