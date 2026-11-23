# Step 1 on PACE: English runs + translate-then-retrieve RAG

Plan: `2026-11-22-step1-diagnostic-plan.md`. The Gemini runs and both translations are already done locally;
PACE only runs the two open models (HiMed, Nanda). All commands run from the repo root on PACE.

## 0. Get the branch

```bash
cd <your repo on PACE>
git fetch origin
git checkout perf/step1-diagnostic
git pull
```

Check that the data files arrived (all three must exist):

```bash
ls performance_eval/data/excluded_items.csv \
   performance_eval/data/bad_translation_items.csv \
   performance_eval/data/milu_en/translations_google.jsonl \
   performance_eval/data/milu_en/translations_gemini.jsonl
```

## 1. English runs (open models)

Four jobs: 2 models × 2 translations. The Hindi `c0_baseline` runs you already have are reused; nothing to re-run.

```bash
EX="--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv"
TR=performance_eval/data/milu_en

sbatch scripts/performance_milu_run.sh himed c0_en_google --language en $EX --translations $TR/translations_google.jsonl
sbatch scripts/performance_milu_run.sh nanda c0_en_google --language en $EX --translations $TR/translations_google.jsonl
sbatch scripts/performance_milu_run.sh himed c0_en_gemini --language en $EX --translations $TR/translations_gemini.jsonl
sbatch scripts/performance_milu_run.sh nanda c0_en_gemini --language en $EX --translations $TR/translations_gemini.jsonl
```

Each job's log ends with `Rows: N`; N should equal 388 minus the number of rows in `bad_translation_items.csv`.

## 2. Translate-then-retrieve RAG

Step 2a retrieves passages using the **English** translation as the search query (same BGE-M3 index and corpus as your
existing `rag_k5` run). Step 2b answers the **Hindi** question with those passages, exactly like `rag_k5`, so the only
thing that changes is the query language.

```bash
# 2a. retrieval (needs .venv-rag and the index at $SCRATCH/rag/index_bge_m3, same as before)
RID=$(sbatch --parsable scripts/rag_retrieve.sh \
      --query-translations performance_eval/data/milu_en/translations_google.jsonl \
      --exclude-file performance_eval/data/excluded_items.csv)

# 2b. answering; waits for 2a to finish successfully
P=/home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/passages_rag_k5_enquery.jsonl
sbatch --dependency=afterok:$RID scripts/performance_milu_run.sh himed rag_k5_enquery \
       --exclude-file performance_eval/data/excluded_items.csv --passages-file $P
sbatch --dependency=afterok:$RID scripts/performance_milu_run.sh nanda rag_k5_enquery \
       --exclude-file performance_eval/data/excluded_items.csv --passages-file $P
```

Compare with the old Hindi-query RAG run on the same 388 items (one comparison per model, so `--family-size 1`).
Nanda has no `rag_k5` run yet; if you want its comparison too, first run
`sbatch scripts/performance_milu_run.sh nanda rag_k5 --passages-file /home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/passages_rag_k5.jsonl`.

```bash
R=/home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med
python -m performance_eval.analyze --model himed --family-size 1 --raw-dir $R --out-dir performance_eval/analysis_outputs/rag_enquery \
       --exclude-file performance_eval/data/excluded_items.csv \
       --ladder rag_k5:first_letter rag_k5_enquery:first_letter
```

The row for `rag_k5_enquery` gives the change vs Hindi-query RAG (Δ, flips, McNemar p). For reference, HiMed on these
388 items: no RAG 63.4%, Hindi-query RAG 56.4%.

## 3. Bring the results back and build the report

The Gemini outputs live on the laptop, the open-model outputs on PACE. Copy the PACE ones down (run on the laptop):

```bash
cd ~/"Desktop/Research/Baseline Evaluation/.worktrees/step1"
mkdir -p performance_eval/outputs/milu_en_med
scp '<pace-login>:/home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/*_c0_en_*_raw.csv*' performance_eval/outputs/milu_en_med/
```

Then (laptop) — once per translation, so conclusions can be checked under both:

```bash
H=performance_eval/outputs/milu_hi_med; E=performance_eval/outputs/milu_en_med
X="--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv"
for T in google gemini; do
  python -m performance_eval.diagnostic $X --en-tag $T --out performance_eval/analysis_outputs/step1_diagnostic_$T.md \
    --cell himed:hi:$H/himed_c0_baseline_raw.csv --cell nanda:hi:$H/nanda_c0_baseline_raw.csv \
    --cell gemini_flash:hi:$H/gemini_flash_c0_baseline_raw.csv \
    --cell himed:en:$E/himed_c0_en_${T}_raw.csv --cell nanda:en:$E/nanda_c0_en_${T}_raw.csv \
    --cell gemini_flash:en:$E/gemini_flash_c0_en_${T}_raw.csv
done
```

A conclusion counts only if both reports agree (plan D5).
