# Tokenizer fertility (HiMed vs Nanda)

Items: 388 (388-item set after both exclusion files). Fertility = tokens / whitespace words on question_with_options, add_special_tokens=False. English = google translation. Descriptive only, no significance test.

## Mean fertility (median in parentheses)

| tokenizer | Hindi | English | Hindi / English |
|---|---|---|---|
| HiMed | 2.94 (2.88) | 1.65 (1.60) | 1.79 |
| Nanda | 1.69 (1.63) | 1.65 (1.60) | 1.03 |

## HiMed Hindi accuracy by quartile

### By HiMed-tokenizer Hindi fertility

| quartile | range | n | HiMed Hindi accuracy |
|---|---|---|---|
| Q1 | 2.06–2.68 | 97 | 64.9% |
| Q2 | 2.69–2.88 | 97 | 62.9% |
| Q3 | 2.88–3.12 | 97 | 64.9% |
| Q4 | 3.12–4.54 | 97 | 60.8% |

### By Hindi word count (length confound check)

| quartile | range | n | HiMed Hindi accuracy |
|---|---|---|---|
| Q1 | 12–19 | 97 | 60.8% |
| Q2 | 19–23 | 97 | 68.0% |
| Q3 | 23–30 | 97 | 68.0% |
| Q4 | 30–93 | 97 | 56.7% |

If accuracy falls with fertility but not with word count, fragmentation (not just length) tracks errors; if both fall together, length is a confound.

## Reading

HiMed's tokenizer (Llama 3.1 vocabulary) splits Hindi into ~1.8× as many tokens per word as English;
Nanda's extended vocabulary brings Hindi to parity (1.03×). Within HiMed, though, Hindi accuracy is
nearly flat across fertility quartiles (64.9 / 62.9 / 64.9 / 60.8%), a 4 pp spread that is smaller
than the spread across word-count quartiles. Heavily fragmented questions are not clearly the ones
HiMed gets wrong, so tokenizer fragmentation is unlikely to be the main driver of the 20 pp gap.
