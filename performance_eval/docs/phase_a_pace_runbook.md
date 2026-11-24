# Phase A on PACE: where is HiMed's Hindi→English gap?

Plan: `2026-11-24-phase-a-gap-localization-plan.md`. Five HiMed GPU jobs, then two reports on the laptop.
The Hindi `c0_baseline`, Hindi `c2_cot` and English `c0_en_{google,gemini}` runs already exist and are reused.

| job | condition label | stem | options | reasoning | compared with |
|---|---|---|---|---|---|
| 1 | `c0_mix_qhi_oen_google` | Hindi | English (google) | — | `c0_baseline` |
| 2 | `c0_mix_qen_ohi_google` | English (google) | Hindi | — | `c0_baseline` |
| 3 | `c0_mix_qhi_oen_gemini` | Hindi | English (gemini) | — | `c0_baseline` |
| 4 | `c0_mix_qen_ohi_gemini` | English (gemini) | Hindi | — | `c0_baseline` |
| 5 | `c2_cot_en` | Hindi | Hindi | English (instructed) | `c2_cot` |

## 0. Get the branch (PACE, repo root)

```bash
git fetch origin
git checkout perf/step1-diagnostic
git pull
ls performance_eval/data/excluded_items.csv performance_eval/data/bad_translation_items.csv \
   performance_eval/data/milu_en/translations_google.jsonl performance_eval/data/milu_en/translations_gemini.jsonl
```

## 1. Optional smoke test (≈5 min, mostly model loading)

Checks the new flags on 3 items before spending GPU hours. Uses throwaway labels; delete the CSVs afterwards.

```bash
EX="--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv"
TR=performance_eval/data/milu_en
sbatch scripts/performance_milu_run.sh himed smoke_mix --language hi_q_en_opts $EX --translations $TR/translations_google.jsonl --limit 3
sbatch scripts/performance_milu_run.sh himed smoke_cot_en --use-cot --reasoning-language en --max-new-tokens 2048 $EX --limit 3
```

Look at the `model_answer` column of `smoke_cot_en`: the reasoning should be in English. If HiMed still reasons
in Hindi, comparison 3 will come out "inconclusive" (that is a result, not a bug).
Then: `rm /home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/himed_smoke_*`.

## 2. The five jobs

```bash
EX="--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv"
TR=performance_eval/data/milu_en
for T in google gemini; do
  sbatch scripts/performance_milu_run.sh himed c0_mix_qhi_oen_$T --language hi_q_en_opts $EX --translations $TR/translations_$T.jsonl
  sbatch scripts/performance_milu_run.sh himed c0_mix_qen_ohi_$T --language en_q_hi_opts $EX --translations $TR/translations_$T.jsonl
done
sbatch scripts/performance_milu_run.sh himed c2_cot_en --use-cot --reasoning-language en --max-new-tokens 2048 $EX
```

Each log (`/home/hice1/gchiu6/scratch/logs/perf-milu-*.out`) ends with `Rows: 388`.

**Timing.** The existing HiMed runs generated ~80–90k tokens per 388–489 items (≈160–240 tokens per answer,
because HiMed writes `## Thinking … ## Response` even when asked for one letter). At typical single-stream
speeds on a PACE GPU that is roughly 45–75 min per mixed-language job. `c2_cot_en` is the slowest:
English reasoning may be longer than HiMed's short Hindi CoT (mean 178 tokens), so allow 1–2 h. The 4 h
wall time that held for `c2_cot` is enough. If a job is killed, resubmit the same command; it resumes.

## 3. Copy outputs to the laptop

On PACE the script writes every run to `.../performance_eval_outputs/milu_hi_med/`. Run on the laptop:

```bash
cd ~/"Desktop/Research/Baseline Evaluation/.worktrees/step1"
mkdir -p performance_eval/outputs/phase_a
scp '<pace-login>:/home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/himed_c0_mix_*_raw.csv*' performance_eval/outputs/phase_a/
scp '<pace-login>:/home/hice1/gchiu6/scratch/performance_eval_outputs/milu_hi_med/himed_c2_cot_en_raw.csv*' performance_eval/outputs/phase_a/
```

## 4. Build the two reports (laptop)

The existing control runs stay where they are; the report scores every raw CSV and filters to the 388 items.

```bash
H=performance_eval/outputs/milu_hi_med; E=performance_eval/outputs/milu_en_med; P=performance_eval/outputs/phase_a
X="--exclude-file performance_eval/data/excluded_items.csv --exclude-file performance_eval/data/bad_translation_items.csv"
for T in google gemini; do
  python -m performance_eval.gap_report $X --en-tag $T --out performance_eval/analysis_outputs/phase_a/gap_report_$T.md \
    --cell hi:first_letter:$H/himed_c0_baseline_raw.csv \
    --cell en:first_letter:$E/himed_c0_en_${T}_raw.csv \
    --cell q_hi_o_en:first_letter:$P/himed_c0_mix_qhi_oen_${T}_raw.csv \
    --cell q_en_o_hi:first_letter:$P/himed_c0_mix_qen_ohi_${T}_raw.csv \
    --cell cot_hi:after_marker:$H/himed_c2_cot_raw.csv \
    --cell cot_en:after_marker:$P/himed_c2_cot_en_raw.csv
done
```

## 5. Reading the reports

Each report has: accuracy per setup; three comparisons (Δ, flips, McNemar p with Bonferroni m = 3, and the
share of the Hindi→English gap G each recovers); the English-reasoning compliance line; and the verdict.

A part "explains the gap" when its comparison is significant, positive, and recovers ≥ 50% of G
(comparison 3 also needs ≥ 80% of `c2_cot_en` answers to actually reason in English). The verdict names
the Phase B direction:

| fires | meaning | Phase B |
|---|---|---|
| 1 (English options) | Hindi option terms | bilingual term-level data in SFT |
| 2 (English stem) | Hindi question stem | parallel bilingual SFT on full questions |
| 3 (English reasoning) | reasoning language | SFT: Hindi question → English reasoning → answer |
| none | only the fully English item recovers it | cross-lingual self-distillation |

A conclusion counts only if `gap_report_google.md` and `gap_report_gemini.md` agree. Comparison 3 uses no
translation, so it is the same in both.

## Optional: Nanda

Not part of the decision rule. Same commands with `nanda` in place of `himed` (Nanda's CoT budget was 768:
`--max-new-tokens 768`), and Nanda's raws in the report cells.
