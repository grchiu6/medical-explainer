# Phase A: gap localization (HiMed)

- excluded by `excluded_items.csv`: 101 items
- excluded by `bad_translation_items.csv`: 0 items
- hi (first_letter): 489 rows in `himed_c0_baseline_raw.csv`, 388 after exclusions
- en (first_letter): 388 rows in `himed_c0_en_google_raw.csv`, 388 after exclusions
- q_hi_o_en (first_letter): 388 rows in `himed_c0_mix_qhi_oen_google_raw.csv`, 388 after exclusions
- q_en_o_hi (first_letter): 388 rows in `himed_c0_mix_qen_ohi_google_raw.csv`, 388 after exclusions
- English translation: google

Gap G = acc(en) − acc(hi) = +20.1 pp

## Accuracy (95% Wilson CI)

| cell | setup | n | accuracy | 95% CI | none_rate |
|---|---|---|---|---|---|
| hi | c0_baseline (hi stem, hi options) | 388 | 63.4% | 58.5%–68.0% | 1.5% |
| en | c0_en (en stem, en options) | 388 | 83.5% | 79.5%–86.9% | 1.3% |
| q_hi_o_en | c0_mix_qhi_oen (hi stem, en options) | 388 | 77.1% | 72.6%–81.0% | 1.5% |
| q_en_o_hi | c0_mix_qen_ohi (en stem, hi options) | 388 | 73.5% | 68.8%–77.6% | 4.4% |
| cot_hi | c2_cot (hi item, Hindi CoT) | — | — | — | not run |
| cot_en | c2_cot_en (hi item, English CoT) | — | — | — | not run |

## Pre-registered comparisons (exact McNemar, Bonferroni m = 3, alpha = 0.05)

| # | comparison | Δ (pp) | flips (newly correct / newly wrong) | McNemar p | p (Bonferroni) | significant? | recovered share of G |
|---|---|---|---|---|---|---|---|
| 1 | q_hi_o_en vs hi | +13.7 | +79 / −26 | 2.166e-07 | 6.498e-07 | yes (better) | 67.9% |
| 2 | q_en_o_hi vs hi | +10.1 | +71 / −32 | 0.0001532 | 0.0004597 | yes (better) | 50.0% |
| 3 | cot_en vs cot_hi | — | — | — | — | not run | — |

## English-reasoning compliance (share of responses whose reasoning letters are ≥ 50% Latin)

—

## Verdict (rule fixed before the runs; plan A5)

Comparison 1 fires: Hindi option terms explains the gap (Δ +13.7 pp, recovers 68% of G = 20.1 pp) → Phase B: bilingual term-level data (Hindi term ↔ English term) in SFT. Comparison 2 fires: Hindi question stem explains the gap (Δ +10.1 pp, recovers 50% of G = 20.1 pp) → Phase B: parallel bilingual SFT on full questions. (Not run: cot_en vs cot_hi.)

A conclusion counts only if the google and gemini reports agree (step-1 D5); comparison 3 uses no translation and is identical in both.
