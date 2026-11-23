# MCQ accuracy by condition

Chance baseline: 25.0% (1 / 4 options, the most common option count in the data). Majority-letter baseline: always answering C would score 29.6%.

Δ vs prior (in percentage points, pp) and McNemar p compare each row to the row above it (paired on n_paired items; every row covers the same item_ids). Flips = items newly correct / newly wrong relative to the row above (+N / −M; net = N − M). p (Bonferroni, m=1) = min(1, p × 1), with m the pre-registered number of comparisons per model. kept? = yes when Δ > 0 and the adjusted p < 0.05. Accuracy CIs are 95% Wilson intervals. none_rate is the share of answers with no parseable letter (counted as incorrect). no_marker_rate (CoT / after_marker rows only) is the share with no explicit Answer marker, scored from the last letter instead; a high value suggests truncated reasoning. — means not applicable (first row has no prior).

| model | condition | n | n_paired | accuracy (95% CI) | none_rate | no_marker_rate | Δ vs prior (pp) | flips (newly correct / newly wrong) | McNemar p | p (Bonferroni, m=1) | kept? | chance baseline |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nanda | c0_baseline | 388 | — | 70.6% (65.9%–74.9%) | 0.0% | — | — | — | — | — | — | 25.0% |
| nanda | rag_k5_enquery | 388 | 388 | 70.6% (65.9%–74.9%) | 0.3% | — | +0.0 pp | +39 / −39 | 1.0000 | 1.0000 | no | 25.0% |
