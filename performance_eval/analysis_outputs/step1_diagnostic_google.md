# Step 1 diagnostic

- excluded by `excluded_items.csv`: 101 items
- excluded by `bad_translation_items.csv`: 0 items
- himed/hi: 489 rows in `himed_c0_baseline_raw.csv`, 388 after exclusions
- nanda/hi: 489 rows in `nanda_c0_baseline_raw.csv`, 388 after exclusions
- gemini_flash/hi: 388 rows in `gemini_flash_c0_baseline_raw.csv`, 388 after exclusions
- himed/en: 388 rows in `himed_c0_en_google_raw.csv`, 388 after exclusions
- gemini_flash/en: 388 rows in `gemini_flash_c0_en_google_raw.csv`, 388 after exclusions
- nanda/en: 388 rows in `nanda_c0_en_google_raw.csv`, 388 after exclusions
- English translation: google

## Accuracy (95% Wilson CI)

| model | language | n | accuracy | 95% CI | none_rate |
|---|---|---|---|---|---|
| himed | hi | 388 | 63.4% | 58.5%–68.0% | 1.5% |
| nanda | hi | 388 | 70.6% | 65.9%–74.9% | 0.0% |
| gemini_flash | hi | 388 | 97.4% | 95.3%–98.6% | 0.0% |
| himed | en | 388 | 83.5% | 79.5%–86.9% | 1.3% |
| gemini_flash | en | 388 | 95.6% | 93.1%–97.2% | 0.0% |
| nanda | en | 388 | 76.5% | 72.1%–80.5% | 0.0% |

## Pre-registered comparisons (exact McNemar, Bonferroni m = 5, alpha = 0.05)

| # | comparison | Δ (pp) | flips (newly correct / newly wrong) | McNemar p | p (Bonferroni) | significant? |
|---|---|---|---|---|---|---|
| 1 | gemini_flash/hi vs himed/hi | +34.0 | +135 / −3 | 2.515e-36 | 1.257e-35 | yes (better) |
| 2 | gemini_flash/hi vs nanda/hi | +26.8 | +108 / −4 | 2.483e-27 | 1.241e-26 | yes (better) |
| 3 | himed/en vs himed/hi | +20.1 | +97 / −19 | 8.63e-14 | 4.315e-13 | yes (better) |
| 4 | nanda/en vs nanda/hi | +5.9 | +45 / −22 | 0.006741 | 0.03371 | yes (better) |
| 5 | gemini_flash/en vs gemini_flash/hi | -1.8 | +1 / −8 | 0.03906 | 0.1953 | no |

## Verdict (rule fixed before the runs; plan D8)

The open Indic models are the bottleneck: Gemini is 26.8 pp above the better open model in Hindi (comparisons 1 and 2 significant). Next: fine-tuning (e.g. translated MedMCQA) or a stronger base model. Hindi is the bottleneck: both open models score >= 5 pp higher on the English translations (comparisons 3 and 4 significant). Next: translate-then-answer pipeline, more Hindi medical training data.

## Gemini cost

- gemini_flash/hi: 388 calls, $0.1644 (34722 thinking tokens, 0 blocked)
- gemini_flash/en: 388 calls, $0.1435 (29917 thinking tokens, 0 blocked)
