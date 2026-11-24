# Phase A summary: where HiMed's Hindi→English gap comes from

Plan: `performance_eval/docs/2026-11-24-phase-a-gap-localization-plan.md` (rule A5, fixed before the runs).
Items: the same 388 MILU Health-and-Medicine items as step 1. Model: HiMed-8B, greedy decoding.
Full reports: `gap_report_google.md`, `gap_report_gemini.md`.

## Verdict

**Hindi option terms explain most of the gap.** Giving HiMed English answer options with the Hindi question
recovers about two thirds of the Hindi→English gap, under both translations:

| setup | google | gemini |
|---|---|---|
| Hindi question, Hindi options (baseline) | 63.4% | 63.4% |
| Hindi question, **English options** | 77.1% | 77.6% |
| English question, Hindi options | 73.5% | 70.6% |
| English question, English options | 83.5% | 84.8% |
| gap G (English − Hindi) | 20.1 pp | 21.4 pp |

| comparison (vs Hindi baseline) | google: Δ, share of G, p (Bonferroni m = 3) | gemini | fires in both? |
|---|---|---|---|
| 1. English options | +13.7 pp, 68%, p = 6.5e-7 | +14.2 pp, 66%, p = 1.1e-7 | **yes** |
| 2. English question stem | +10.1 pp, 50%, p = 4.6e-4 | +7.2 pp, 34%, p = 0.023 | no (google only, exactly at the 50% line) |
| 3. English reasoning | not run | not run | inconclusive |

- **Comparison 1 counts** (both translations agree, step-1 D5). Phase B direction per A5: bilingual
  term-level data (Hindi medical term ↔ English term) in SFT.
- **Comparison 2 does not count.** It is significant and positive in both, but it recovers ≥ 50% of the gap
  only with the google translation, and then only at exactly 50% (39 of 78 items). The English stem helps,
  but less than the English options. Its runs also have more unparseable answers (4.4% / 6.4% vs 1.5%),
  so part of its shortfall may be answer format, not understanding.
- **Comparison 3 is inconclusive.** In 3-item smoke tests HiMed reasoned in Hindi despite "Think step by step in
  English", with both its default Hindi system message and a neutral one (amendment A2a). By rule A5 the
  instruction was not followed, so the two CoT jobs were not run. That HiMed cannot be prompted into English
  reasoning is itself a finding.
- The two parts don't add up neatly. Options alone recover ~67% and the stem alone 34–50%, which is more than
  100% together. They overlap: many items are fixed by either change.

## Supporting analyses (no GPU)

- **Flip table** (`flips_summary.md`; one LLM rater, no human check): of the 97 items HiMed gets right only in
  English (google), 21 hinge on a Sanskritized or native Hindi medical term, 12 on negated question forms,
  and 51 show no single visible cause. This fits the verdict: terminology is the largest visible cause, but
  not the whole story.
- **Tokenizer fertility** (`tokenizer_fertility.md`): HiMed's tokenizer uses ~1.8× more tokens per word for Hindi
  than for English, but its Hindi accuracy is nearly flat across fertility quartiles (64.9 → 60.8%).
  Fragmentation is unlikely to be the main driver.

## Caveats

- Comparison 1's English options come from machine translation, so they may also be *clearer* than the Hindi
  originals, not just in a better-known language (some MILU Hindi options are Sanskritized or misspelled).
  For Phase B the remedy is the same either way: teach the Hindi↔English term mapping.
- One model (HiMed), one dataset (388 general-knowledge-style medical MCQs). Nanda was not run (optional in the plan).

## Next (Phase B)

Fine-tune with bilingual term-level data: pairs of Hindi medical terms (including Sanskritized forms such as
अस्थिसुषिरता, विषाणु, प्रतिजैविक) with their English equivalents, mixed into SFT. Re-test on the same 388 items
with the Hindi baseline prompt; success = a significant gain that closes a large share of the 20–21 pp gap.
