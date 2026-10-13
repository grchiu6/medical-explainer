# Register Classifier: Plan and Process

*Started 2026-10-12. Code: `literacy_eval/register/`. This document is the reference for **what** we are building,
**why** each step exists, and **who does what**. A chronological search log is in the appendix.*

This replaces the deterministic adaptation scores in `baseline_metrics_guide.md` (ASL, AWL, RH1/RH2, `term_rate`,
`rare_rate`, …). **Those scores are no longer used.** The output-cleaning step from that pipeline (`analysis/clean.py`:
answer extraction and usability flags) is kept, because the classifier needs clean input.

---

## 1. Purpose

**Research question:** can LLMs adapt Hindi medical explanations to a reader's literacy level? We asked 5 models to
explain 60 HiMed-Trad questions at 3 levels (`low` = 5th grade, `mid` = 10th grade, `high` = medical professional),
which gives 900 responses.

**The measurement problem:** to say whether a `low` answer really *is* 5th-grade Hindi, we need a reliable judge of
register. Hindi readability formulas exist, but they were fitted on 16 passages (Sinha et al. 2012), and we have no
Hindi-reading annotator.

**The approach:** train a classifier on real Hindi text whose level is known, then apply it to the LLM outputs.

```
  NCERT Class 5  ─┐                                                              ┌─ 900 LLM outputs
  NCERT Class 10 ─┼─▶ OCR ─▶ clean ─▶ normalize ─▶ chunk ─▶ fine-tune BERT ──▶ score ◀─ clean ─▶ normalize
  ICMR Patrika   ─┘          (same normalizer for both sides)                    └─▶ does predicted level
                                                                                      match requested level?
```

If a model adapts, its `low` answers should be classified as 5th grade, `mid` as 10th, and `high` as professional.

## Status (2026-10-13)

| Stage | Status | Result |
|---|---|---|
| Download (`sources`) | ✅ done | 64 NCERT chapter PDFs (Class 5, 9, 10) and 115 ICMR issues; 2,207 pages |
| OCR (`ocr`) | ✅ done | 2,207 / 2,207 pages, 0 failures (~2 h on the Mac) |
| Clean + normalize + chunk (`build_corpus`) | ✅ done, **2 variants** | **A `class10`: 237 passages per class** (train 153 / val 39 / test 45). **B `class9_10`: 353 per class** (train 222 / val 74 / test 57) |
| Shortcut check | ✅ done | A: length-only 0.39, sentence-length 0.62, word naive Bayes 0.99. B: 0.39 / 0.62 / 0.98. Top words are language, not source markers |
| LLM outputs (`prepare_outputs`) | ✅ done | 509 usable responses → 650 windows (usable counts match `results.md`) |
| Fine-tuning (`train`) | ✅ **run 2 done** | Run 1 had a threshold bug (section 4.10). Run 2: IndicBERT 0.985 (Class 10 corpus) / 0.936 (Class 9+10), MuRIL 0.953 (Class 9+10), MuRIL on Class 10 **failed** (stuck at chance). **IndicBERT on the Class 10 corpus is the primary model** |
| Outside validation sets | ⏳ built, **needs scoring on PACE** | ReadMe++ Hindi (1,486 rated sentences) and Hindi Wikipedia medical (600 passages) in `data/register_eval/`. COIL-D is gated: request access on Hugging Face |
| Analysis (`analyze`) | ✅ runs; validation section pending | `literacy_eval/analysis_outputs/register_results.md` |

**Two corpus variants, because the smallest class sets every class's size** (classes are kept equal-sized to avoid bias):

| | A `class10` | B `class9_10` |
|---|---|---|
| "10th grade" text | NCERT Class 10 biology (5 chapters) | Class 10 biology + **Class 9 biology** (`ihsc1` chapters 2 cell, 3 tissues, 11 reproduction, 12 diversity of living things) |
| Smallest class | Class 10: 237 passages | Class 5: 353 passages |
| Passages per class (train / val / test) | 237 (153 / 39 / 45) | 353 (222 / 74 / 57) |
| Trade-off | purest grade label, little data | +49% data, but the class spans two grades |

We train on both and compare (section 7). Directories: `data/register_corpus/` (A) and `data/register_corpus_9_10/` (B).

## 2. Decisions at a glance

| Topic | Decision | Why |
|---|---|---|
| 5th-grade text | NCERT Class 5 environmental-studies books, both editions: `ehev1` हमारा अद्भुत संसार (new) and `ehap1` आस-पास (old) | Real 5th-grade Hindi covering food, water, the body and illness. Class 5 is the smallest class, so both editions are used. |
| 10th-grade text | NCERT Class 10 विज्ञान (Science) `jhsc1`, **biology/health chapters only**. Variant B adds Class 9 `ihsc1` biology chapters | Real 10th-grade Hindi. Chemistry and physics chapters would make "10th grade" partly mean "chemistry". |
| Professional text | **ICMR Patrika**, the Hindi periodical of the Indian Council of Medical Research: 115 issues, 2007 to ~2025, 1,239 pages | The only free, verified source of medical Hindi written by researchers (see appendix for the search). |
| Text extraction | **OCR every page the same way** (Tesseract `tessdata_best`, `hin+eng`, 300 dpi), even where a PDF has text | ICMR PDFs have no usable text, so they must be OCR'd. OCR'ing only ICMR would make "has OCR errors" a cue for "professional". |
| Normalization | **One shared `normalize()`** for training text **and** LLM outputs | Any formatting difference between sources is a shortcut the classifier could learn instead of register (section 4.4). |
| Passage length | Same length distribution for every class, matched to the LLM outputs | Otherwise "long passage = professional" is a shortcut. |
| Class balance | Down-sample every class to the size of the smallest | Prevents the classifier from favouring the biggest class. |
| Splits | 70 / 15 / 15 **by chapter (NCERT) or issue (ICMR)** | Passages from one chapter on both sides of the split would inflate test scores. |
| Encoders | MuRIL (`google/muril-base-cased`) and IndicBERTv2-MLM (`ai4bharat/IndicBERTv2-MLM-only`); keep the better one | Both are pretrained on Hindi. IndicBERTv2-**SS** was dropped: "SS" is script standardisation, which converts other Indic scripts to Devanagari and does nothing special for Hindi-English mixing. |
| Output | Ordinal: 5th < 10th < professional, giving a probability per class plus a continuous "expected level" | Lets us test ordering (low < mid < high), not just exact matches. |
| GPU (PACE) | `l40s` or `a40`. **Not `mi210`**: AMD, won't run the CUDA build of PyTorch | A BERT-base model fine-tunes in minutes on any of these. |

## 3. Data sources

### 3.1 NCERT (5th and 10th grade)

| Book | Code | Pages | Raw text layer | Why OCR anyway |
|---|---|---|---|---|
| Class 5 हमारा अद्भुत संसार (new edition) | `ehev1` | 200 | Unicode, but conjuncts broken (`सांस््ककृ तिक`) | Same pipeline as ICMR |
| Class 5 आस-पास (old edition) | `ehap1` | 216 | Legacy Kruti-Dev font, extracts as garbage (`pkVrs&pkVrs`) | No usable text |
| Class 10 विज्ञान | `jhsc1` | 256 | Unicode, some vowel signs misplaced (`गण\nु धर्म`) | Same pipeline as ICMR |

Download URL: `https://ncert.nic.in/textbook/pdf/{code}dd.zip` (whole book, one PDF per chapter).

Why NCERT **science/EVS** and not NCERT **Hindi language** books: the language books are stories and poems. If one
class is stories and another is medicine, the classifier learns topic instead of difficulty.

### 3.2 ICMR Patrika (professional)

- **What it is:** ICMR scientists summarising their research in formal, Sanskrit-heavy Hindi. Topics in a 10-issue
  sample: hypertension, tribal seroprevalence, plasmid-mediated drug resistance, TB diagnosis in HIV patients, iodine
  deficiency, occupational health. About 800 words per page, so far more text than we need.
- **Archive:** <https://www.icmr.gov.in/hindi-publication-icmr-patrika>. The page links some issues twice under
  different names, so issues are de-duplicated by file hash.
- **OCR quality:** Hindi words come out almost entirely correct. Typical errors are digits (`2015`→`20:5`) and italic
  words turned into English junk (`creed`, `ceed`). The normalizer removes both (section 4.4).

### 3.3 Limitations of these sources

- **No traditional-medicine content.** HiMed-Trad asks Ayurveda/Unani/Siddha questions, while ICMR is Western and
  public health. Hindi Ayurveda theses (Shodhganga) could be added later as a second professional source.
- **ICMR is research summaries**, a little less technical than clinical textbooks, and some articles explain
  terms ("कमेंसल अर्थात …").
- **Source confound.** Grades 5 and 10 come from NCERT; professional comes from ICMR. The classifier could learn
  "NCERT vs ICMR" instead of register. Section 4 removes formatting differences, and section 5's outside test sets
  check whether it worked.

## 4. Pipeline

Each stage is a module in `literacy_eval/register/`, run as `python -m literacy_eval.register.<stage>` from the repo root.
Raw data and the built corpus live in `data/register_raw/` and `data/register_corpus/`. Both are **gitignored**: the
text is copyrighted, and everything can be rebuilt by rerunning the stages.

### 4.1 `sources`: download
Downloads the 3 NCERT books, every ICMR issue (de-duplicated by MD5 hash) and Tesseract's Hindi model. Writes
`data/register_raw/manifest.csv` with one row per PDF: source, label, group (chapter or issue), file, md5, pages.
**Result:** 49 NCERT chapter PDFs (416 Class 5 pages, 256 Class 10 pages) and 115 ICMR issues (1,239 pages).

### 4.2 `ocr`: same OCR for every page
Renders each page at 300 dpi in grayscale and runs Tesseract (`hin+eng`, page segmentation mode 1, which handles
ICMR's two-column layout). Writes `data/register_raw/ocr/<pdf>/pNNN.txt` plus `.tsv`, which holds word confidences.
It runs in parallel, and a killed run can simply be restarted because finished pages are skipped. About 20 minutes on
a 12-core Mac. It is CPU-only, so it doesn't need PACE.

### 4.3 `clean_sources`: remove what isn't prose (source-specific)
This stage only **removes** material. It never rewrites the language of what remains.

| Source | Removed | Why |
|---|---|---|
| All | Low-confidence OCR lines, page numbers, 1–2 character lines; words hyphenated across a line break are rejoined | OCR debris |
| NCERT | Print furniture (`Chapter 1.indd`, codes like `0536CH01`), running heads, figure captions | Not prose |
| NCERT Class 5 | Activity and exercise blocks: bullet lines (`+ • ●`), questions addressed to the child, imperative cues (`बताओ / लिखो / चर्चा करो / पता करो`) | Keeps the class about register, not "textbook exercise" genre |
| NCERT Class 5 | **Teacher-facing notes** ("यह इकाई कक्षा 5 के विद्यार्थियों को …") and front/back matter | Adult register inside a Class 5 book; would contaminate the label |
| NCERT Class 5 + 10 | **Commands and questions to the student:** polite `-इए` (`देखिए, कीजिए, बनाइए`) and familiar `-ओ` (`देखो, लगाओ`) imperatives, questions with `तुम/आप/आपको…`. Not removed: `लिए` ("for"), `चाहिए` ("should"), `किए/दिए` (past forms) | Exercise genre, not register. Side effect: Class 5 training text contains almost no commands, while health advice to a reader naturally does ("डॉक्टर से मिलिए"). |
| NCERT Class 10 | Chemistry and physics chapters (keep biology/health) | Topic control |
| ICMR | Masthead (`आई सी एम आर पत्रिका`, month/year), editorial and contents pages, reference lists, non-medical pages | Not research prose, or not medical |
| ICMR | **Event-report pages:** 3 or more of `बैठक, सम्मेलन, संगोष्ठी, कार्यशाला, उद्घाटन, मुख्य अतिथि, अध्यक्षता, स्वागत, धन्यवाद ज्ञापन`. Deliberately *not* `सम्पन्न`/`आयोजित`, because research text says "अध्ययन सम्पन्न किया गया" ("the study was conducted") | Speeches, meetings, conference listings (spot-checked: 464 pages) |
| ICMR | **Sentences naming the publisher or a person:** `आई सी एम आर`, `परिषद`, `डॉ`, `महानिदेशक`, `सचिव` | Source markers. Found by the shortcut check: before this rule, ICMR's spelled-out name was among the top "professional" words |

**Bug found and fixed (2026-10-13):** the imperative rules used the whole Devanagari Unicode block as a "word
boundary", but that block includes the danda `।`. So a command at the end of a sentence ("…धो लीजिए।", the most common
place) never matched and survived as training text. Fixed with a boundary class that excludes `।` and `॥`, plus a
regression test. Exercise-sentence drops rose from 990 to 1,191 (variant A) and 1,628 (variant B, which adds Class 9
lab procedures). All counts and numbers in this document are from after the fix.

Every rule records how many lines it dropped, and these counts appear in the corpus report so we can check that no
rule removes too much. Current counts (variant A): 1,191 exercise sentences, 1,852 ICMR source-marker sentences, 464 event pages,
95 teacher pages.

### 4.4 `normalize`: the shared normalizer
One function, `normalize(text)`, used on **every** training passage **and every** LLM output. Each rule removes a
difference that comes from the source, not from the language:

| Rule | Example | Bias it prevents |
|---|---|---|
| Unicode NFC; remove nukta, ZWJ/ZWNJ, soft hyphen | `ज़रूरी` → `जरूरी` | OCR often misses the nukta; LLMs write it. "Has nuktas" would mark LLM text. |
| All sentence ends → `।` | `हैं|` → `हैं।`; `करें।.` → `करें।` | OCR reads `।` as `|`; HiMed writes `।.` |
| Every number → `0` | `वर्ष 20:5 में` → `वर्ष 0 में` | OCR garbles digits, and ICMR is full of statistics: "many numbers" would become "professional" |
| Latin-script words removed | `सभी ceed आगामी` → `सभी आगामी` | ICMR's OCR junk and LLMs' English terms. Kept, they would teach "English = professional". |
| Markdown, bullets, `[12]` references, quote marks, stray symbols removed | `- **पहला** कारण` → `पहला कारण` | LLM formatting vs print layout |
| Line breaks and spacing collapsed into running text | | OCR line breaks vs markdown newlines |

**English use is still measured,** just not by the classifier: `prepare_outputs` records each response's share of
Latin-script words *before* normalization, as a separate descriptive column. The `low` prompt forbids English, so
this remains a reportable finding.

Tested in `tests/test_register_normalize.py` with real OCR and LLM strings; `normalize()` is idempotent.

### 4.5 `chunk`: passages, de-duplication, balance, splits
1. **Sentences:** split on `।?!` (reusing `analysis/features.py:split_sentences`).
2. **Passages:** pack whole sentences up to a target word count. **Targets are drawn from the same distribution
   for every class**, matched to the lengths of the usable LLM outputs. Passage length then carries no class signal,
   while sentence length, a real register signal, is left alone.
3. **Filters:** drop passages that are mostly non-Devanagari or too short.
4. **De-duplication:** drop repeated sentences and passages. The two Class 5 editions share some text, and ICMR
   sometimes reprints articles.
5. **Balance:** down-sample every class to the smallest class. ICMR passages are drawn from as many issues and years
   as possible.
6. **Splits:** 70 / 15 / 15 by group, so no group appears in two splits. A group is a chapter (Class 5), an issue
   (ICMR), or a **4-page block of a chapter (Class 10)**. Class 10 has only 5 biology chapters, so splitting by whole
   chapter would leave a single chapter, and a single topic, as the test set. The cost is that one chapter's blocks
   can sit in different splits, which leaks a little shared vocabulary.

**Current corpus** (before balancing; `corpus_report.md` has the full tables):

| class | passages (A / B) | groups (A / B) | median words | mean words per sentence | numbers (share of words) |
|---|---|---|---|---|---|
| grade5 | 353 / 353 | 31 / 31 | 77 | 14.4 | 0.8% |
| grade10 | 237 / 511 | 19 / 40 | 65 / 77 | 17.6 / 18.1 | 0.5% / 1.1% |
| professional | 3,309 / 3,295 | 115 / 115 | 92 / 96 | 26.3 / 26.5 | 2.3% |

After balancing: A has 237 passages per class, B has 353. Words per sentence rise from 14 → 18 → 26, the expected
register signal. **Known small residual:** after balancing, ICMR passages come out a little longer than the others
(B: median 112 words vs 77 for Class 5), because ICMR's long sentences overshoot the length targets. The
length-only baseline (4.7) picks this up weakly (0.39, chance 0.33).

### 4.6 `build_corpus`: run 4.3–4.5 and report
`--variant class10` (default) writes `data/register_corpus/`; `--variant class9_10` writes `data/register_corpus_9_10/`.
Each gets `{train,val,test}.jsonl` (fields: `text`, `label`, `source`, `group`) and `corpus_report.md`, which contains:
- counts per class at each stage
- passage-length histograms per class (they should look the same)
- per-rule drop counts
- 10 random passages per class, to read with a translator

### 4.7 Shortcut check (before any BERT)
Train two cheap classifiers on the corpus:
- one using **only passage length**
- one using **character n-grams** (short letter sequences)

If either is near-perfect, a leftover artifact (for example a masthead string) is giving the class away. Fix the
cleaning before spending GPU time. (Implemented with numpy only, because `pip` is broken in `venv`: Homebrew Python
3.14's `pyexpat` can't load the system `libexpat`.)

| Classifier | A `class10` | B `class9_10` | Reading (chance = 0.33) |
|---|---|---|---|
| Passage length only | 0.39 | 0.39 | Close to chance, weak leftover length difference (see 4.5). It never predicts grade10, so it acts as a 2-way split |
| Mean sentence length only | 0.62 | 0.62 | The simple baseline BERT must clearly beat |
| Word naive Bayes | 0.99 | 0.98 | Vocabulary differs by class. Top words are language: `स्वास्थ्य, व्यापकता, रोगवाहक` / `कोशिका, ऊतक, जनन` / `मैं, घर, गाँव, पानी` |

Test sets are small (135 and 171 passages), so differences of a few points are noise.

**Caution:** the word baseline is high partly because of **topic**: Class 10 = biology, ICMR = public health,
Class 5 = everyday life and environment. BERT will exploit that too. Only the outside test sets (section 5) can show
whether the classifier also recognises register on unfamiliar topics.

Report: `data/register_corpus/shortcut_report.md`.

### 4.8 `prepare_outputs`: the 900 LLM responses
Reuses `analysis/clean.py`: `extract_answer` keeps only HiMed-8B's text after `## Response`, and `flag_response` marks
refusals, question echoes, repetition loops and non-Hindi text. **Only usable responses are scored.** Unusable ones are
reported as a separate outcome, as before. Usable responses then go through the same `normalize()`; responses longer
than a passage are split into passage-sized windows and their scores averaged.

Outputs: `data/register_corpus/llm_responses.csv` (all 900, with `usable_rules`, `failure_type`, `latin_share`,
`n_windows`) and `llm_windows.jsonl` (650 windows from the 509 usable responses).

### 4.9 Train (PACE GPU): `train.py` + `scripts/register_train.sh`
- **Ordinal head (CORAL):** the encoder produces one score `s` per passage; two learned, ordered thresholds give
  `P(level > 5th)` and `P(level > 10th)`. The loss is binary cross-entropy on those two questions, so calling a 10th-grade
  passage "professional" costs less than calling it "5th grade". `expected_level = P(>5th) + P(>10th)` runs from 0 to 2.
- **Settings:** 5 epochs, batch 16, max 320 tokens. Two learning rates: **2e-5 for the pretrained encoder** and
  **1e-3 for the new score layer and the two thresholds** (`--head-lr`), which start from scratch and must move far.
  The thresholds start 3.0 apart (at -1.5 and +1.5). The best epoch is chosen by val macro-F1.
- **Outputs** (`~/scratch/register_runs/<encoder>_seed42/`): `metrics.json`, `test_scores.jsonl`,
  `llm_window_scores.jsonl` (bring this back to the Mac), and `best.pt`.
- Tested: `tests/test_register_train.py` (ordinal maths) and a local end-to-end smoke run with a tiny model.

### 4.10 Reading the training outputs, and what run 1 showed

Each run writes three files to `register_runs/<corpus>/<encoder>_seed42/`:

| File | What it holds | How to read it |
|---|---|---|
| `metrics.json` | `history`: one row per **epoch** (one pass over the training data) with the training `loss` and the validation accuracy / macro-F1. `test`: the final score on held-out data, plus `confusion_gold_x_pred` | **Loss** is the training error; it should fall. **val** numbers show whether the model works on data it didn't train on; they should rise, then flatten. If loss falls but val doesn't rise, the model is memorising (overfitting). The **confusion matrix** has one row per true class and one column per predicted class; the diagonal is correct. |
| `test_scores.jsonl` | one row per held-out passage: `p_grade5/p_grade10/p_professional` (probabilities summing to 1), `expected_level` (0 = 5th, 2 = professional), `predicted` | Look at the *average* `expected_level` per true class: a healthy model gives ≈0 / ≈1 / ≈2. |
| `llm_window_scores.jsonl` | the same scores for the LLM answers (usable ones, in windows) | The thing we actually care about. Average `expected_level` per model × requested level should rise from `low` to `high` if the model adapts. |

**Metrics.** *Accuracy* = share correct. *Macro-F1* averages the per-class F1 scores, so a class that is never predicted
drags it down (chance for 3 balanced classes: accuracy 0.33).

**Run 1 (2026-10-13), and why it was a bug in our code, not a finding.** All four runs scored accuracy ≈0.66 with the
confusion matrix's middle column all zeros: 10th grade was *never* predicted. Two checks showed why:
- The highest `p_grade10` any passage ever got was exactly **0.245** in all four runs.
- With the ordinal thresholds a gap `g` apart, the best a passage can score for the middle class is
  `sigmoid(g/2) − sigmoid(−g/2)`. That is 0.245 at g = 1 (where I initialised them) and exceeds the 1/3 needed to win only
  when g > 1.39. The thresholds were trained at the encoder's tiny 2e-5, so they never left g = 1.

So the middle class was *structurally unable* to win. Evidence the rest worked: IndicBERT's average `expected_level`
was 0.06 / 0.98 / 1.94 for the three true classes, a clean ordering. MuRIL, in contrast, barely learned (training loss
nearly flat, near-identical scores for every passage); it needs the rerun with the fixed learning rates before judging.
**Fix:** a separate larger learning rate for the head and thresholds, initial gap 3.0, two regression tests, and a
synthetic end-to-end check (middle class now predicted, 100% on an easy 3-class toy set). Rerun all four jobs.

## 5. Validation

| Check | Data | Pass condition |
|---|---|---|
| Held-out chapters and issues | `test.jsonl` | Accuracy, macro-F1 and confusion matrix; clearly better than the sentence-length baseline (0.62) |
| **Human difficulty ratings** | **ReadMe++ Hindi**: 1,491 sentences rated 1–6 (A1–C2) by human annotators across 16 domains (MIT license; clone in `../readme/`). Built: 1,486 rows | Classifier scores rank sentences like the humans do (Spearman ρ clearly above 0). **Our stand-in for a Hindi annotator.** Caveat: single sentences, median 16 words, shorter than training passages |
| Encyclopedia medical text | Hindi Wikipedia medical passages (MIRACL `hi`, 2022 dump): 600 passages from 259 disease/treatment articles (biographies, hospitals, films and astrology filtered out) | Mostly 10th / professional, rarely 5th |
| Lay health text | COIL-D Health v2 (public-health messages, CC BY 4.0) | Mostly 5th / 10th. **Gated on Hugging Face**: needs an access request |

These sources were never seen in training, so they test whether the classifier learned **register** or just
**NCERT vs ICMR**.

**How they're built and scored:** `python -m literacy_eval.register.validation_sets` writes `data/register_eval/*.jsonl`
(same `normalize()` as everything else). Trained weights stay on PACE, so `scripts/register_score.sh` scores them there
with every trained model (`literacy_eval/register/score.py`, no retraining). `analyze.py` then reports them in section B
of `register_results.md`.

## 6. Scoring the 900 outputs

For each model:
- **Confusion matrix:** requested level vs predicted level.
- **Match rate:** share of usable responses classified at the requested level.
- **Ordering:** for each question, does the expected level rise from `low` to `mid` to `high`? Tested with the Page
  trend test across the three levels, and Wilcoxon for `low` vs `high`.
- **Alongside:** usable-response rate and English (Latin-script) share per level. Neither comes from the classifier.

Implemented in `literacy_eval/register/analyze.py` (run with `venv-analysis`, which has scipy). Each response's score is
the word-weighted average of its windows. Paired tests reuse `literacy_eval/analysis/stats.py`. Output:
`literacy_eval/analysis_outputs/register_results.md`, with A (training check), B (outside validation), C (LLM outputs
for the primary run) and a robustness table across all passing runs.

## 7. Action items

### For you (Grace)
| # | Task | Where | Why |
|---|---|---|---|
| 1 | Run the 4 training jobs (steps below): 2 encoders × 2 corpus variants | Mac + PACE | Compares MuRIL vs IndicBERTv2, and "Class 10 only" vs "Class 9 + 10" |
| 2 | `git clone https://github.com/tareknaous/readme` and check that the Hindi data is included and what the license is | anywhere | Validation against human difficulty ratings (section 5) |
| 3 | *(Optional)* Ask a Hindi reader to skim the 10 sample passages per class in `data/register_corpus*/corpus_report.md` | — | Catches cleaning mistakes the rules miss |

**Everything on PACE goes to scratch** (`/home/hice1/gchiu6/scratch`, the only place with space). The SLURM script sets
`HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR` and `TMPDIR` there, and writes logs and
results there. The one thing it can't cover is your own login-node commands, which is why step 1 sets `HF_HOME` first.

**Running the training jobs:**
```bash
# 1. On PACE, once. Set HF_HOME FIRST (otherwise the download fills your home directory).
export HF_HOME=/home/hice1/gchiu6/scratch/hf_cache
mkdir -p ~/scratch/{logs,hf_cache,register_corpus,register_corpus_9_10,register_eval,register_runs}
huggingface-cli download google/muril-base-cased
huggingface-cli download ai4bharat/IndicBERTv2-MLM-only
#   -> if either asks for access/login, tell me which one.

# 2. On the Mac, from the repo root: copy both corpora up. This uses plain ssh + tar, so it works even
#    when scp doesn't. (llm_windows.jsonl is shared, so it is copied into both folders on PACE.)
H=gchiu6@login-ice.pace.gatech.edu
(cd data && COPYFILE_DISABLE=1 tar czf - register_corpus/{train,val,test,llm_windows}.jsonl \
                                          register_corpus_9_10/{train,val,test}.jsonl) \
  | ssh $H 'tar xzf - -C ~/scratch && cp ~/scratch/register_corpus/llm_windows.jsonl ~/scratch/register_corpus_9_10/'
ssh $H 'ls ~/scratch/register_corpus ~/scratch/register_corpus_9_10'      # check: 4 files in each
#   (optional, when you have it) ssh $H 'cat > ~/scratch/register_eval/readmepp_hi.jsonl' < readmepp_hi.jsonl

# 3. On PACE, in the repo, once the new code is there (git pull):
sbatch --job-name=reg-muril-c10   scripts/register_train.sh muril     register_corpus
sbatch --job-name=reg-indic-c10   scripts/register_train.sh indicbert register_corpus
sbatch --job-name=reg-muril-c910  scripts/register_train.sh muril     register_corpus_9_10
sbatch --job-name=reg-indic-c910  scripts/register_train.sh indicbert register_corpus_9_10
squeue -u gchiu6                                   # watch them; logs: ~/scratch/logs/register-<jobname>-<id>.out

# 4. When all four finish (a few minutes each), on the Mac: copy results back, WITHOUT the ~1 GB weight files
ssh $H 'cd ~/scratch && tar czf - --exclude=best.pt register_runs' | tar xzf - -C .      # -> ./register_runs/
```

### If `scp` / `rsync` fail

The commands above avoid them. If `ssh` itself works, so do these. Notes:
- Your Mac has OpenSSH 10, whose `scp` uses the SFTP protocol by default. If PACE's login node doesn't offer SFTP,
  `scp` fails but `scp -O ...` (the older protocol) often works. This is a guess, not confirmed; the error message
  would tell us.
- If each new `ssh` asks for a password or Duo approval, reuse one login for all the commands by adding
  `-o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h:%p -o ControlPersist=15m` to your first `ssh`.
- Another route is the PACE web portal (Open OnDemand), which has a file-upload page, if your account has it.

The `best.pt` weight files (~1 GB each) stay on PACE; delete them from scratch when you're done. When step 4 is done,
send me `register_runs/*/*/metrics.json` and the `llm_window_scores.jsonl` files (or just tell me they're in
`register_runs/`).

GPU: the script asks for `l40s`. If those are busy, change `--gres=gpu:l40s:1` in the script to `a40`, `a100` or `h100`.
**Never `mi210`** (AMD).

**Scoring the outside validation sets** (after the 4 training jobs; keep the `best.pt` files on PACE until this is done):
```bash
# 1. On the Mac, from the repo root: upload the two validation files
H=gchiu6@login-ice.pace.gatech.edu
(cd data && COPYFILE_DISABLE=1 tar czf - register_eval/readmepp_hi.jsonl register_eval/miracl_medical.jsonl) \
  | ssh $H 'mkdir -p ~/scratch/register_eval && tar xzf - -C ~/scratch && ls ~/scratch/register_eval'

# 2. On PACE, in the repo, after git pull:
sbatch scripts/register_score.sh          # scores both files with every trained model (a few minutes)

# 3. On the Mac: bring everything back (same command as before) and rebuild the report
ssh $H 'cd ~/scratch && tar czf - --exclude=best.pt register_runs' | tar xzf - -C .
venv-analysis/bin/python -m literacy_eval.register.analyze
```

### For Claude
- Done: `sources`, `ocr`, `normalize`, `clean_sources`, `chunk`, `build_corpus`, `shortcut_check`,
  `prepare_outputs`, `train.py`, `scripts/register_train.sh`, and 44 unit tests (`tests/test_register_*.py`).
- Next: prepare the outside validation sets (COIL-D Health, MIRACL medical passages, ReadMe++ once downloaded), then
  analyse the scores (section 6).

**Rebuilding locally** (from the repo root, with `venv`):
```bash
python -m literacy_eval.register.sources          # download (skips existing files)
python -m literacy_eval.register.ocr --workers 10 # ~2 h, resumable
python -m literacy_eval.register.build_corpus --variant class10     # -> data/register_corpus/
python -m literacy_eval.register.build_corpus --variant class9_10   # -> data/register_corpus_9_10/
python -m literacy_eval.register.shortcut_check [--corpus-dir data/register_corpus_9_10]
python -m literacy_eval.register.prepare_outputs  # -> llm_responses.csv + llm_windows.jsonl
python -m pytest tests/test_register_*.py
```

## 8. Threats to validity

1. **Source confound** (3.3): mitigated by the shared normalizer and length matching, and checked with the outside
   test sets.
2. **ICMR ≠ clinical textbook:** "professional" here means *medical-research Hindi for an educated audience*. Say so
   when reporting results.
3. **No traditional-medicine text in training,** while the questions are traditional medicine. Ayurveda vocabulary may
   look unfamiliar to the classifier.
4. **OCR noise:** the same pipeline everywhere keeps it even across classes, but it isn't zero.
5. **Topic is confounded with class:** everyday life (Class 5), biology (Class 10), public health (ICMR). The
   classifier may partly learn topic. Checked with the outside sets; the LLM answers are all medical, so watch for
   "medical topic → professional".
6. **Small training set:** 163 training passages per class. Report variation across seeds if results are close.
7. **Imperatives removed from Class 5** (exercise filter), while health advice uses imperatives. A `low` answer full
   of "…कीजिए" may look less like Class 5 than it should.
8. **Reward hacking later:** if a later phase fine-tunes a generator to hit a target level, **don't evaluate it with
   the same classifier used to train it**. The generator would learn to fool that classifier, and the scores would
   improve without the text getting easier.

---

## Appendix A: datasets considered and rejected

| Dataset | Why not |
|---|---|
| ASER (PrathamOrg) | Audio of children reading aloud, labelled per child; passages are grade 1–2 level. |
| HindiRC | Grades 2–5 only; no license stated. |
| StoryWeaver | Reading levels are not school grades; mostly stories. |
| MP Hindi MBBS textbooks (and UP / Uttarakhand / Chhattisgarh) | Ideal, but no official download; would need a request to MP's medical education department or GMC Bhopal. |
| Hindi-medium BAMS (Ayurveda) textbooks | Good fit for HiMed-Trad, but paid and copyrighted, and would need scanning. Possible later addition, e.g. via a university library. |
| HiMed-Trad / HiMed-West corpora | HiMed-8B was trained on them, so the classifier would learn that model's style as "professional". |
| MIRACL `hi` (Hindi Wikipedia) | ~870 medical articles, but encyclopedic or casual register, not professional. **Used as a test set instead.** |
| Synthetic Indian Clinical Notes (Mendeley) | English (some Hinglish), synthetic. |
| MedMCQA | English. |
| CSTT *Vigyan Garima Sindhu* | General science, not medical; garbled text layer. |
| "Hindi Health Dataset" (Kaggle), "Zenodo Healthcare Dataset", "MEDIQ", "Clinical Medicine (Hindi)" by Anil K. Tripathi | Suggested by Gemini; none could be found. Probably invented. |
| Peer-reviewed Hindi medical journals | None found; Indian medical journals publish in English. |

## Appendix B: decision log

- **2026-10-11:** Idea: fine-tune a Hindi BERT on texts of known level to score register. Dataset survey; ASER and
  HindiRC rejected; NCERT science/EVS chosen for grades 5 and 10.
- **2026-10-12:** Deterministic readability scores dropped. Professional-source search (MBBS, BAMS, HiMed corpora,
  MIRACL, ICMR). ICMR Patrika chosen after a hands-on check (download, OCR, 10-issue content sample). NCERT text
  layers found broken, so everything is OCR'd. Shared normalizer, length matching and group splits adopted.
  Pipeline built; full OCR run; corpus built. The shortcut check found ICMR source markers (its own name, `डॉ`,
  meeting reports), and these were removed. Class 10 split by 4-page blocks.
- **2026-10-13 (later):** Run 2 succeeded for 3 of 4 runs; IndicBERT chosen as primary. Built the outside validation sets
  (ReadMe++, MIRACL medical), score-only mode + `register_score.sh`, and `analyze.py`.
- **2026-10-13:** Class 9 biology added as corpus variant B (`class9_10`) so we can compare. Chapters were identified by
  OCR of each chapter's first page, because the 2025 book renumbered everything. Found and fixed the danda word-boundary
  bug in the imperative filter (section 4.3). SLURM script rewritten so every cache, temp file, log and result is on
  scratch, and to take `<encoder> <corpus>`.
