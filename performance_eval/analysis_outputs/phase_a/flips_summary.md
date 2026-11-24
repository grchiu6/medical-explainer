Labels: one LLM rater (Claude), no human check.

# HiMed Hindi/English flip table: summary

Items whose correctness differs between `himed c0_baseline` (Hindi) and `himed c0_en_google` (English): 116 (97 English-only correct, 19 Hindi-only correct).

| category | en_only_correct | hi_only_correct | total |
|---|---|---|---|
| terminology | 21 | 0 | 21 |
| question_form | 12 | 2 | 14 |
| traditional_medicine | 1 | 0 | 1 |
| translation_artifact | 3 | 3 | 6 |
| item_noise | 4 | 1 | 5 |
| format | 5 | 3 | 8 |
| other | 51 | 10 | 61 |
| total | 97 | 19 | 116 |

## Same flip under the gemini translation

97 of 116 flipped items flip the same way when the English item comes from the gemini translation (English-only correct: 88 of 97; Hindi-only correct: 9 of 19).

## Codebook

- `terminology`: Hindi medical term rare/Sanskritized/ambiguous; English uses a common term
- `question_form`: negation, 'which is NOT', multi-statement, assertion–reason
- `traditional_medicine`: Ayurveda/Yoga/Unani/Indian-specific
- `translation_artifact`: translation changed meaning, dropped or added a cue
- `item_noise`: gold looks wrong or item ambiguous in both languages
- `format`: no parseable answer in one language
- `other`: none of the above

## Rater notes

- Primary category = the feature that most plausibly explains the flip; when two apply equally, the
  order format → item_noise → translation_artifact → traditional_medicine → question_form →
  terminology → other breaks the tie (secondary features are in `note`).
- `terminology` is used only when a Sanskritized or native Hindi term carries the answer
  (e.g. अस्थिसुषिरता, विषाणु, सूखा रोग). Devanagari transliterations of English terms
  (स्कर्वी, एलिसा, names of people and drugs) are `other` with "translit" in the note:
  26 of the 61 `other` items have such a note. About 25 `other` items show no distinctive Hindi feature
  ("plain recall"), so the flip is not explained by anything visible in the item.
- Reading: in this table the most common *visible* cause of English-only correctness is
  Sanskritized terminology (21 of 97), then negated question forms (12). Most flips (51 of 97) have no
  single visible cause, which points to general Hindi comprehension rather than one fixable part.
  The Phase A runs (English options; English stem; English reasoning) test this directly.
- Hindi-only-correct flips are less stable: only 9 of 19 reproduce with the gemini translation
  (vs 88 of 97 English-only flips), so many of them look like noise around a coin flip.
