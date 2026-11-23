# MCQ accuracy by condition

Chance baseline: 25.0% (1 / 4 options, the most common option count in the data). Majority-letter baseline: always answering C would score 29.6%.

Δ vs prior (in percentage points, pp) and McNemar p compare each row to the row above it (paired on n_paired items; every row covers the same item_ids). Flips = items newly correct / newly wrong relative to the row above (+N / −M; net = N − M). p (Bonferroni, m=2) = min(1, p × 2), with m the pre-registered number of comparisons per model. kept? = yes when Δ > 0 and the adjusted p < 0.05. Accuracy CIs are 95% Wilson intervals. none_rate is the share of answers with no parseable letter (counted as incorrect). no_marker_rate (CoT / after_marker rows only) is the share with no explicit Answer marker, scored from the last letter instead; a high value suggests truncated reasoning. — means not applicable (first row has no prior).

| model | condition | n | n_paired | accuracy (95% CI) | none_rate | no_marker_rate | Δ vs prior (pp) | flips (newly correct / newly wrong) | McNemar p | p (Bonferroni, m=2) | kept? | chance baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| himed | c0_baseline | 388 | — | 63.4% (58.5%–68.0%) | 1.5% | — | — | — | — | — | — | 25.0% |
| himed | rag_k5 | 388 | 388 | 56.4% (51.5%–61.3%) | 18.6% | — | -7.0 pp | +61 / −88 | 0.0328 | 0.0656 | no | 25.0% |
| himed | rag_k5_enquery | 388 | 388 | 57.7% (52.8%–62.5%) | 17.5% | — | +1.3 pp | +51 / −46 | 0.6849 | 1.0000 | no | 25.0% |
