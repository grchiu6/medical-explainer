# HiMed-Trad literacy evaluation results

Suspect reference items: 0

## usability

| model | level | n | refusal | non_answer | echo | degenerate | non_hindi | too_short | incoherent_judge | truncated | usable | usable_pct | usable_ci | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| airavata | low | 60 | 0 | 1 | 48 | 3 | 0 | 1 | 0 | 26 | 7 | 11.7 | 6-22 | rules only (upper bound) |
| airavata | mid | 60 | 0 | 2 | 45 | 2 | 0 | 1 | 0 | 24 | 10 | 16.7 | 9-28 | rules only (upper bound) |
| airavata | high | 60 | 0 | 1 | 35 | 5 | 0 | 0 | 0 | 17 | 19 | 31.7 | 21-44 | rules only (upper bound) |
| apollo | low | 60 | 0 | 0 | 2 | 26 | 0 | 0 | 0 | 48 | 32 | 53.3 | 41-65 | rules only (upper bound) |
| apollo | mid | 60 | 0 | 0 | 1 | 39 | 0 | 0 | 0 | 56 | 20 | 33.3 | 23-46 | rules only (upper bound) |
| apollo | high | 60 | 0 | 0 | 2 | 44 | 0 | 0 | 0 | 57 | 14 | 23.3 | 14-35 | rules only (upper bound) |
| himed | low | 60 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 58 | 96.7 | 89-99 | rules only (upper bound) |
| himed | mid | 60 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 58 | 96.7 | 89-99 | rules only (upper bound) |
| himed | high | 60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 60 | 100 | 94-100 | rules only (upper bound) |
| nanda | low | 60 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 47 | 78.3 | 66-87 | rules only (upper bound) |
| nanda | mid | 60 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 57 | 95 | 86-98 | rules only (upper bound) |
| nanda | high | 60 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 17 | 54 | 90 | 80-95 | rules only (upper bound) |
| mmed_llama | low | 60 | 0 | 1 | 0 | 33 | 3 | 0 | 0 | 26 | 23 | 38.3 | 27-51 | rules only (upper bound) |
| mmed_llama | mid | 60 | 0 | 0 | 0 | 35 | 4 | 0 | 0 | 32 | 21 | 35 | 24-48 | rules only (upper bound) |
| mmed_llama | high | 60 | 0 | 0 | 0 | 19 | 12 | 0 | 0 | 29 | 29 | 48.3 | 36-61 | rules only (upper bound) |

## usability_tests

| model | outcome | n_items | low_pct | mid_pct | high_pct | cochran_Q | cochran_p | mcnemar_low_only | mcnemar_high_only | mcnemar_p |
|---|---|---|---|---|---|---|---|---|---|---|
| airavata | usable | 60 | 11.7 | 16.7 | 31.7 | 11.1 | 0.00381 | 3 | 15 | 0.00754 |
| airavata | refusal | 60 | 0 | 0 | 0 | nan | nan | 0 | 0 | 1 |
| apollo | usable | 60 | 53.3 | 33.3 | 23.3 | 12.3 | 0.00214 | 24 | 6 | 0.00143 |
| apollo | refusal | 60 | 0 | 0 | 0 | nan | nan | 0 | 0 | 1 |
| himed | usable | 60 | 96.7 | 96.7 | 100 | 2 | 0.368 | 0 | 2 | 0.5 |
| himed | refusal | 60 | 0 | 0 | 0 | nan | nan | 0 | 0 | 1 |
| nanda | usable | 60 | 78.3 | 95 | 90 | 7.9 | 0.0193 | 4 | 11 | 0.118 |
| nanda | refusal | 60 | 21.7 | 5 | 10 | 7.9 | 0.0193 | 11 | 4 | 0.118 |
| mmed_llama | usable | 60 | 38.3 | 35 | 48.3 | 2.36 | 0.307 | 9 | 15 | 0.307 |
| mmed_llama | refusal | 60 | 0 | 0 | 0 | nan | nan | 0 | 0 | 1 |

## adaptation

| model | feature | expected | n_items | median_low | median_mid | median_high | page_p | wilcoxon_p | rank_biserial | share_expected_direction | share_monotonic | page_p_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| airavata | asl | up | 3 | 17.5 | 14 | 14.4 | nan | nan | nan | 0.667 | 0 | nan |
| airavata | short_sentence_share | down | 3 | 0.25 | 0.273 | 0 | nan | nan | nan | 0.667 | 0.333 | nan |
| airavata | term_rate | up | 3 | 0 | 0.0067 | 0 | nan | nan | nan | 0 | 0 | nan |
| airavata | rare_rate | up | 3 | 0.0571 | 0.0571 | 0.05 | nan | nan | nan | 0.333 | 0 | nan |
| airavata | awl | up | 3 | 3.37 | 3.52 | 3.56 | nan | nan | nan | 1 | 0.667 | nan |
| airavata | psw_rate | up | 3 | 0.34 | 0.356 | 0.387 | nan | nan | nan | 0.667 | 0 | nan |
| airavata | juk_rate | up | 3 | 0.171 | 0.228 | 0.225 | nan | nan | nan | 1 | 0 | nan |
| airavata | latin_rate | up | 3 | 0 | 0 | 0 | nan | nan | nan | 0 | 0 | nan |
| airavata | rh1 | up | 3 | 5.22 | 5.51 | 5.67 | nan | nan | nan | 1 | 0.667 | nan |
| airavata | rh2 | up | 3 | 4.92 | 5.15 | 5.2 | nan | nan | nan | 1 | 0.667 | nan |
| airavata | gloss_rate | none | 3 | 0 | 0 | 0 | nan | nan | nan | 0 | nan | nan |
| airavata | copy_rate | none | 3 | 0.0816 | 0.159 | 0.0702 | nan | nan | nan | 0 | nan | nan |
| airavata | n_words | none | 3 | 70 | 140 | 100 | nan | nan | nan | 0.667 | nan | nan |
| apollo | asl | up | 3 | 17.5 | 16.4 | 25.8 | nan | nan | nan | 1 | 0.333 | nan |
| apollo | short_sentence_share | down | 3 | 0.125 | 0.111 | 0 | nan | nan | nan | 1 | 0.333 | nan |
| apollo | term_rate | up | 3 | 0 | 0.007 | 0.0297 | nan | nan | nan | 0.667 | 0 | nan |
| apollo | rare_rate | up | 3 | 0 | 0.0328 | 0.0396 | nan | nan | nan | 1 | 0.667 | nan |
| apollo | awl | up | 3 | 3.48 | 3.36 | 3.53 | nan | nan | nan | 0.667 | 0.333 | nan |
| apollo | psw_rate | up | 3 | 0.352 | 0.369 | 0.393 | nan | nan | nan | 0.667 | 0.667 | nan |
| apollo | juk_rate | up | 3 | 0.174 | 0.211 | 0.2 | nan | nan | nan | 1 | 0 | nan |
| apollo | latin_rate | up | 3 | 0 | 0 | 0 | nan | nan | nan | 0.333 | 0 | nan |
| apollo | rh1 | up | 3 | 5.46 | 5.23 | 5.69 | nan | nan | nan | 1 | 0.333 | nan |
| apollo | rh2 | up | 3 | 5.06 | 4.92 | 5.12 | nan | nan | nan | 0.667 | 0.333 | nan |
| apollo | gloss_rate | none | 3 | 0 | 0 | 1.55 | nan | nan | nan | 0.667 | nan | nan |
| apollo | copy_rate | none | 3 | 0.16 | 0.173 | 0.184 | nan | nan | nan | 0.667 | nan | nan |
| apollo | n_words | none | 3 | 140 | 122 | 129 | nan | nan | nan | 0.667 | nan | nan |
| himed | asl | up | 56 | 13 | 20 | 24 | 0 | 0 | 0.951 | 0.893 | 0.5 | 0 |
| himed | short_sentence_share | down | 56 | 0 | 0 | 0 | 0.0853 | 0.0006 | 0.81 | 0.286 | 0 | 0.286 |
| himed | term_rate | up | 56 | 0 | 0 | 0.0083 | 0.0402 | 0.136 | 0.312 | 0.375 | 0.0714 | 0.241 |
| himed | rare_rate | up | 56 | 0.04 | 0.0642 | 0.0747 | 0.0054 | 0.0435 | 0.316 | 0.643 | 0.214 | 0.0429 |
| himed | awl | up | 56 | 3.62 | 3.56 | 3.68 | 0.0715 | 0.0493 | 0.302 | 0.625 | 0.125 | 0.286 |
| himed | psw_rate | up | 56 | 0.4 | 0.411 | 0.423 | 0.198 | 0.838 | 0.0313 | 0.536 | 0.214 | 0.286 |
| himed | juk_rate | up | 56 | 0.143 | 0.163 | 0.197 | 0.0017 | 0.0085 | 0.408 | 0.643 | 0.321 | 0.0153 |
| himed | latin_rate | up | 56 | 0 | 0 | 0 | 0.0117 | 0.0004 | 1 | 0.286 | 0.0179 | 0.0817 |
| himed | rh1 | up | 56 | 5.84 | 5.71 | 5.94 | 0.11 | 0.0913 | 0.259 | 0.607 | 0.125 | 0.286 |
| himed | rh2 | up | 56 | 5.23 | 5.19 | 5.34 | 0.0445 | 0.0176 | 0.365 | 0.625 | 0.143 | 0.241 |
| himed | gloss_rate | none | 56 | 0 | 0 | 0 | 0.0091 | 0.0003 | 1 | 0.304 | nan | nan |
| himed | copy_rate | none | 56 | 0.1 | 0.157 | 0.14 | 0.0782 | 0.133 | 0.235 | 0.571 | nan | nan |
| himed | n_words | none | 56 | 14 | 29 | 65.5 | 0 | 0 | 1 | 0.982 | nan | nan |
| nanda | asl | up | 40 | 19.8 | 21.5 | 21 | 0.0588 | 0.219 | 0.223 | 0.575 | 0.275 | 0.235 |
| nanda | short_sentence_share | down | 40 | 0.16 | 0.0557 | 0.0909 | 0.0146 | 0.0072 | 0.521 | 0.625 | 0.125 | 0.0731 |
| nanda | term_rate | up | 40 | 0 | 0 | 0 | 0.269 | 0.661 | 0.107 | 0.325 | 0 | 0.808 |
| nanda | rare_rate | up | 40 | 0.0267 | 0.0386 | 0.0452 | 0.0002 | 0.0003 | 0.634 | 0.725 | 0.4 | 0.0021 |
| nanda | awl | up | 40 | 3.84 | 3.95 | 3.91 | 0.0009 | 0.0003 | 0.627 | 0.75 | 0.35 | 0.007 |
| nanda | psw_rate | up | 40 | 0.434 | 0.442 | 0.438 | 0.327 | 0.314 | 0.185 | 0.6 | 0.1 | 0.808 |
| nanda | juk_rate | up | 40 | 0.226 | 0.239 | 0.249 | 0.0004 | 0.0056 | 0.495 | 0.75 | 0.325 | 0.0036 |
| nanda | latin_rate | up | 40 | 0 | 0.0012 | 0 | 0.369 | 0.875 | 0.0513 | 0.175 | 0.05 | 0.808 |
| nanda | rh1 | up | 40 | 6.32 | 6.55 | 6.47 | 0.0051 | 0.0011 | 0.576 | 0.725 | 0.275 | 0.0304 |
| nanda | rh2 | up | 40 | 5.58 | 5.73 | 5.71 | 0.0009 | 0.0004 | 0.62 | 0.75 | 0.35 | 0.007 |
| nanda | gloss_rate | none | 40 | 0.166 | 1.06 | 0.985 | 0.0009 | 0.002 | 0.576 | 0.675 | nan | nan |
| nanda | copy_rate | none | 40 | 0.0124 | 0.0112 | 0.0084 | 0.891 | 0.294 | -0.198 | 0.375 | nan | nan |
| nanda | n_words | none | 40 | 194 | 284 | 322 | 0.0003 | 0.0003 | 0.652 | 0.7 | nan | nan |
| mmed_llama | asl | up | 3 | 14 | 31 | 16 | nan | nan | nan | 0.667 | 0 | nan |
| mmed_llama | short_sentence_share | down | 3 | 0.2 | 0 | 0 | nan | nan | nan | 0.667 | 0 | nan |
| mmed_llama | term_rate | up | 3 | 0 | 0 | 0 | nan | nan | nan | 0 | 0 | nan |
| mmed_llama | rare_rate | up | 3 | 0.0143 | 0.0741 | 0.0769 | nan | nan | nan | 0.667 | 0 | nan |
| mmed_llama | awl | up | 3 | 3.2 | 3.57 | 3.25 | nan | nan | nan | 0.667 | 0 | nan |
| mmed_llama | psw_rate | up | 3 | 0.35 | 0.333 | 0.294 | nan | nan | nan | 0.333 | 0 | nan |
| mmed_llama | juk_rate | up | 3 | 0.05 | 0.3 | 0.308 | nan | nan | nan | 1 | 0.667 | nan |
| mmed_llama | latin_rate | up | 3 | 0 | 0.206 | 0 | nan | nan | nan | 0.333 | 0 | nan |
| mmed_llama | rh1 | up | 3 | 4.86 | 5.63 | 4.99 | nan | nan | nan | 0.333 | 0 | nan |
| mmed_llama | rh2 | up | 3 | 4.59 | 5.25 | 4.76 | nan | nan | nan | 1 | 0 | nan |
| mmed_llama | gloss_rate | none | 3 | 0 | 2.94 | 6.25 | nan | nan | nan | 0.667 | nan | nan |
| mmed_llama | copy_rate | none | 3 | 0.107 | 0 | 0 | nan | nan | nan | 0 | nan | nan |
| mmed_llama | n_words | none | 3 | 40 | 31 | 16 | nan | nan | nan | 0 | nan | nan |

## compliance

| model | level | n_usable | no_latin_pct | mean_short_sentence_share | mostly_short_pct | mean_term_rate | uses_terms_pct | mean_gloss_rate | has_gloss_pct |
|---|---|---|---|---|---|---|---|---|---|
| airavata | low | 7 | 100 | 0.181 | 14.3 | 0.0082 | 42.9 | 0.11 | 14.3 |
| airavata | mid | 10 | 100 | 0.256 | 20 | 0.0088 | 30 | 0 | 0 |
| airavata | high | 19 | 94.7 | 0.202 | 21.1 | 0.0357 | 47.4 | 0.18 | 10.5 |
| apollo | low | 32 | 100 | 0.09 | 3.1 | 0.0163 | 43.8 | 1.34 | 9.4 |
| apollo | mid | 20 | 100 | 0.127 | 5 | 0.0126 | 55 | 0.04 | 5 |
| apollo | high | 14 | 42.9 | 0.014 | 0 | 0.0177 | 50 | 3.72 | 64.3 |
| himed | low | 58 | 100 | 0.247 | 25.9 | 0.0146 | 19 | 0 | 0 |
| himed | mid | 58 | 93.1 | 0.023 | 1.7 | 0.0304 | 50 | 0.65 | 12.1 |
| himed | high | 60 | 71.7 | 0.046 | 3.3 | 0.0199 | 53.3 | 1.52 | 30 |
| nanda | low | 47 | 87.2 | 0.149 | 2.1 | 0.002 | 31.9 | 0.59 | 48.9 |
| nanda | mid | 57 | 42.1 | 0.091 | 1.8 | 0.0018 | 36.8 | 1.32 | 82.5 |
| nanda | high | 54 | 83.3 | 0.097 | 0 | 0.003 | 48.1 | 1.61 | 81.5 |
| mmed_llama | low | 23 | 78.3 | 0.125 | 21.7 | 0.0033 | 8.7 | 3 | 69.6 |
| mmed_llama | mid | 21 | 19 | 0.072 | 9.5 | 0.002 | 9.5 | 3.94 | 71.4 |
| mmed_llama | high | 29 | 24.1 | 0.109 | 3.4 | 0.0145 | 24.1 | 7.16 | 79.3 |
