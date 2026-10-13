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

## Status (2026-10-12)

| Stage | Status | Result |
|---|---|---|
| Download (`sources`) | ✅ done | 49 NCERT chapter PDFs, 115 ICMR issues; 1,911 pages |
| OCR (`ocr`) | ✅ done | 1,911 / 1,911 pages, 0 failures (~1.5 h on the Mac) |
| Clean + normalize + chunk (`build_corpus`) | ✅ done | **237 passages per class** (balanced): train 489 / val 111 / test 111 |
| Shortcut check | ✅ done | length-only 0.34 (≈ chance ✓), sentence-length 0.57, word naive Bayes 0.91 (top words now real language) |
| LLM outputs (`prepare_outputs`) | ✅ done | 509 usable responses → 650 windows (usable counts match `results.md`) |
| Fine-tuning (`train`) | ⏳ ready, **needs you on PACE** | code smoke-tested locally with a tiny model |
| Outside validation sets | ⏳ to do | ReadMe++ (needs your download), COIL-D, MIRACL medical |

**Bottleneck:** Class 10 has only 237 passages (5 biology chapters), and it sets the size of every class. Option, not yet
decided: add the biology chapters of NCERT Class 9 विज्ञान and call the class "grades 9–10" (action item 1 in section 7).

## 2. Decisions at a glance

| Topic | Decision | Why |
|---|---|---|
| 5th-grade text | NCERT Class 5 environmental-studies books, both editions: `ehev1` हमारा अद्भुत संसार (new) and `ehap1` आस-पास (old) | Real 5th-grade Hindi covering food, water, the body and illness. Class 5 is the smallest class, so both editions are used. |
| 10th-grade text | NCERT Class 10 विज्ञान (Science) `jhsc1`, **biology/health chapters only** | Real 10th-grade Hindi. Chemistry and physics chapters would make "10th grade" partly mean "chemistry". |
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

Every rule records how many lines it dropped, and these counts appear in the corpus report so we can check that no
rule removes too much. Current counts: 990 exercise sentences, 1,852 ICMR source-marker sentences, 464 event pages,
95 teacher pages, 1,937 low-confidence paragraphs, 3,337 short paragraphs.

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

**Current corpus:**

| | passages before balancing | groups | median words | mean words per sentence | numbers (share of words) |
|---|---|---|---|---|---|
| grade5 | 370 | 31 | 73 | 14.3 | 0.6% |
| grade10 | 237 | 19 | 84 | 17.8 | 0.5% |
| professional | 3,286 | 115 | — | ~26 | ~2.3% |

After balancing: 237 per class. Passage-length distributions match across classes (medians 72–84 words). Words per
sentence rise from 14 → 18 → 26, the expected register signal.

### 4.6 `build_corpus`: run 4.3–4.5 and report
Writes `data/register_corpus/{train,val,test}.jsonl` (fields: `text`, `label`, `source`, `group`) and
`data/register_corpus/corpus_report.md`, which contains:
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

| Classifier | Test accuracy (chance 0.33) | Reading |
|---|---|---|
| Passage length only | 0.34 | ✓ length matching works |
| Mean sentence length only | 0.57 | the simple baseline BERT must clearly beat; mostly confuses 5th with 10th |
| Word naive Bayes | 0.91 | vocabulary differs by class, as expected. Top words are now language (`स्वास्थ्य, व्यापकता, रोगवाहक` / `कोशिका, जनन, रुधिर` / `मैं, जमीन, गाँव`) |

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
- **Settings:** 5 epochs, batch 16, learning rate 2e-5, max 320 tokens. The best epoch is chosen by val macro-F1.
- **Outputs** (`~/scratch/register_runs/<encoder>_seed42/`): `metrics.json`, `test_scores.jsonl`,
  `llm_window_scores.jsonl` (bring this back to the Mac), and `best.pt`.
- Tested: `tests/test_register_train.py` (ordinal maths) and a local end-to-end smoke run with a tiny model.

## 5. Validation

| Check | Data | Pass condition |
|---|---|---|
| Held-out chapters and issues | `test.jsonl` | Accuracy, macro-F1 (each class counts equally), confusion matrix; clearly better than the length-only baseline |
| Human difficulty ratings | **ReadMe++ Hindi**: ~1,500 sentences rated A1–C2 by humans ([github.com/tareknaous/readme](https://github.com/tareknaous/readme)) | Classifier scores rank sentences like the humans do (Spearman ρ clearly above 0). **This is our stand-in for a Hindi annotator.** |
| Lay health text | COIL-D Health v2 (public-health messages, CC BY 4.0) | Mostly 5th / 10th |
| Encyclopedia medical text | Hindi Wikipedia medical passages (MIRACL `hi`, filtered) | 10th / professional, rarely 5th |

The last three sources were never seen in training, so they test whether the classifier learned **register** or
just **NCERT vs ICMR**.

## 6. Scoring the 900 outputs

For each model:
- **Confusion matrix:** requested level vs predicted level.
- **Match rate:** share of usable responses classified at the requested level.
- **Ordering:** for each question, does the expected level rise from `low` to `mid` to `high`? Tested with the Page
  trend test across the three levels, and Wilcoxon for `low` vs `high`.
- **Alongside:** usable-response rate and English (Latin-script) share per level. Neither comes from the classifier.

## 7. Action items

### For you (Grace)
| # | Task | Where | Why |
|---|---|---|---|
| 1 | Decide: keep "10th grade" = Class 10 biology only (237 passages per class), or add NCERT Class 9 biology chapters as "grades 9–10" (more data, slightly wider class) | — | Class 10 limits the size of every class |
| 2 | Run the training job (steps below) | Mac + PACE | GPU fine-tuning |
| 3 | `git clone https://github.com/tareknaous/readme` and check that the Hindi data is included and what the license is | anywhere | Validation against human difficulty ratings (section 5) |
| 4 | *(Optional)* Ask a Hindi reader to skim the 10 sample passages per class in `data/register_corpus/corpus_report.md` | — | Catches cleaning mistakes the rules miss |

**Running the training job:**
```bash
# 1. On PACE (once): make the folder; check the encoders download without access approval
mkdir -p ~/scratch/register_corpus
huggingface-cli download google/muril-base-cased
huggingface-cli download ai4bharat/IndicBERTv2-MLM-only

# 2. On the Mac, from the repo root: copy the corpus up
scp data/register_corpus/{train,val,test,llm_windows}.jsonl \
    gchiu6@login-ice.pace.gatech.edu:/home/hice1/gchiu6/scratch/register_corpus/

# 3. On PACE, in the repo (after committing + git pull, so literacy_eval/register/ is there):
sbatch scripts/register_train.sh muril
sbatch scripts/register_train.sh indicbert

# 4. When both finish: send me metrics.json for each, and copy back the scores
scp gchiu6@login-ice.pace.gatech.edu:/home/hice1/gchiu6/scratch/register_runs/*/{metrics.json,llm_window_scores.jsonl} .
```
GPU: the script asks for `l40s`. If those are busy, change `--gres=gpu:l40s:1` to `a40`, `a100` or `h100`. **Never
`mi210`** (AMD).

### For Claude
- Done: `sources`, `ocr`, `normalize`, `clean_sources`, `chunk`, `build_corpus`, `shortcut_check`,
  `prepare_outputs`, `train.py`, `scripts/register_train.sh`, and 43 unit tests (`tests/test_register_*.py`).
- Next: prepare the outside validation sets (COIL-D Health, MIRACL medical passages, ReadMe++ once downloaded), then
  analyse the scores (section 6).

**Rebuilding locally** (from the repo root, with `venv`):
```bash
python -m literacy_eval.register.sources          # download (skips existing files)
python -m literacy_eval.register.ocr --workers 10 # ~1.5 h, resumable
python -m literacy_eval.register.build_corpus     # -> data/register_corpus/*.jsonl + corpus_report.md
python -m literacy_eval.register.shortcut_check   # -> shortcut_report.md
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
