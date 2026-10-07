# Plan: Ground-truth-anchored evaluation of literacy-level explanations (HiMed-Trad track)

## Context

The literacy pipeline (`literacy_eval/`) turns HiMed MCQs into open-ended "explain this" prompts at 3 levels
(low / mid / high). The HiMed-West run is done for 4 models: Apollo, HiMed-8B, Nanda and MMed-Llama, each with 50 items × 3 levels.
Airavata's files are empty and will be rerun later. The outputs have no scoring yet. The proposed next step was to keep the MCQ
ground truth as an anchor: build a reference table of the correct concept and key facts, then score (A) factual accuracy
and retention, (B) readability and (C) vocabulary rarity.

**What checking the data showed (it changes that proposal):**
- **HiMed-West has no rationale field.** Every one of the 1,784 items has only `question_number, question, options, answer, source`.
  **HiMed-Trad has a `cot` field** on all 6,010 items: 1–3 sentences (median ~150 chars) explaining why the correct option is right.
  **Decision (user): switch the accuracy track to HiMed-Trad** and use its CoT as the source for key facts.
- The West prompts are often broken. For example, `"…निम्नलिखित में से कौन प्रबंधन में सबसे उपयुक्त अगला कदम है?"` still asks
  "which of the following" after the options were removed, and `"एक परिस्थिति में…"` is a broken regex strip. Trad items
  need a proper answerability filter and a human-reviewed rewrite of each question stem.
- Outputs need cleaning before any metric is computed. HiMed-8B prints `## Thinking … ## Response` (48 of 50 responses), MMed-Llama often
  answers in English, and Apollo falls into repetition loops.
- The pasted proposal **does** use an LLM judge for accuracy (Step 2). Readability and vocabulary are deterministic. The judge is
  still needed, because exact string matching fails on machine-translated Hindi (e.g. फोलिएट vs फॉलेट, or answers in English).

## What changes from the pasted proposal (review)

| Proposal | Keep / change | Why |
|---|---|---|
| Reference table with target concept + key facts | **Keep**, keyed by `item_id` only | The prompts for each level are already stored in the output CSVs, so don't copy them into the table |
| Key facts "extracted from rationale" | **Change**: split the Trad CoT into sentences, then review by hand | West has no rationale. The Trad CoT is real dataset text, so the reference isn't invented by us |
| "Target identification" | **Make it concrete**: a blind **option-mapping** judge. Given the response and options A–E, which option does it support, or NONE? | Uses the dataset's own answer key, is more objective than a free-form judgment, and gives a clean accuracy score per level |
| Fact coverage YES/NO | **Extend** to SUPPORTED / CONTRADICTED / ABSENT | Contradictions (wrong medical claims) are the safety-relevant signal for simplified text |
| HRS → "CBSE grade" | **Change**: report raw features plus relative scores, and make no grade claim | I don't know of a validated mapping from Hindi readability to CBSE grades. The claim to test is the *ordering* low < mid < high |
| Shabd word rarity | **Start with `wordfreq`** (supports Hindi, pip-installable) and swap in Shabd if it can be obtained | Don't block on getting hold of a dataset |
| (not in proposal) | **Add** judge validation on a hand-labelled sample | Reviewers will ask how reliable the judge is |
| (not in proposal) | **Add** a prompt-term copy rate | The technical wording of the question can carry over into "low" outputs. This is a confound worth measuring |

**Deferred (record in `context.md`):**
- MCQ knowledge ceiling. Trad MCQ results already exist for Airavata and Apollo (`outputs/9:4:2026/*_himed_trad_scored.csv`), but there are none for
  HiMed-8B, Nanda or MMed-Llama. Those need to be run later. For reference, Airavata and Apollo score 8% on the 50 West items, below the 10% random baseline.
- Airavata literacy runs (West files are empty, Trad not yet run).
- Existing West outputs: keep them. They can still get readability and vocabulary metrics at no extra cost, but they won't be used for accuracy.

## Implementation steps

### Step 1: Build a frozen Trad reference set — DONE (60 items)
**New file:** `literacy_eval/topics_himed_trad.py` (mirrors `topics_himed.py`).
- Load `data/himed/HiMed-Trad_Bench.json`. Parse the options out of the `question` text; both the `\nA. …` and inline ` A. … B. …` formats occur.
- Answerability filter: drop negation or exception stems (नहीं है / सिवाय / अतिरिक्त, ~255 items), stems that make no sense
  without options (fill-in statements that don't end in `?`), and Sowa-Rigpa (only 10 items).
- **Stratified sample**: 6 systems × 5 subjects × 2 = **60 items**, `seed=42`. A fallback redistributes quota when a cell is
  thin; for example, Siddha has only 19 prognosis items.
- Draft fields per item:
  `item_id, system, subject, original_question, options{A..E}, correct_answer, correct_option_text, cot,
  topic_hi_draft` (deterministic clean: strip options, "निम्नलिखित में से", "पारंपरिक चिकित्सा में"),
  `key_facts_draft` (CoT split on `।`, dropping any restated "सही विकल्प है…" sentence).
- Write `literacy_eval/reference/himed_trad_review.csv` for **human review**. The reviewer edits `topic_hi_final`
  (an open-ended question that must not reveal the answer) and `key_facts_final` (2–3 facts), and can set `exclude`.
- Command `python -m literacy_eval.topics_himed_trad --finalize` turns the reviewed CSV into
  **`literacy_eval/reference/himed_trad_reference.json`**. This frozen file is the single source of truth: every model
  sees identical prompts and every scorer reads identical facts.

### Step 2: Generate outputs on the cluster — DONE (900 responses, `outputs/himed_trad_open/`)
- `literacy_eval/run_eval.py`: add `himed_trad` to `--topic-source` choices. Load the reference JSON and use output suffix `_himed_trad`.
  Reuse the existing `FIELDNAMES` (`options_json`, `correct_answer`) and set `category = f"{system}/{subject}"`.
- New `scripts/literacy_himed_trad_run.sh`: a copy of `scripts/literacy_himed_run.sh` that passes `--topic-source himed_trad`.
- Run all 5 models (Airavata, Apollo, HiMed-8B, Nanda, MMed-Llama) × 3 levels × 60 items = 900 responses. 


---

## Revision 2 (2026-09-29): what the outputs showed, and how the analysis changes

### What the 900 responses actually look like

| Model | Main failure | Numbers (per level, out of 60) | Usable for analysis? |
|---|---|---|---|
| Airavata | **Echoes the question back** instead of answering | echo 19 / 20 / 27 (low/mid/high) | Mostly no |
| Apollo | **Repetition loops**, meaningless text (e.g. "पेट में दस्त (gas) के कारण पेट में दस्त (distention)…") | repeated-4-gram share >0.3 in 26 / 39 / 46 | Mostly no |
| MMed-Llama | Base model: number lists, Hindi-English mix, invented words | Latin-script share rises with level (median 1% → 18%) | Mostly no |
| HiMed-8B | Clean, and **the only model with clear adaptation**. But the answer after `## Response` is only 1–3 sentences, and at **high, 19/60 never reach `## Response`** because the Thinking block uses all 512 tokens | response median 71 / 139 / 252 chars | Yes (high needs a rerun) |
| Nanda | Fluent, long, cut off at 512 tokens. **Refuses** ("मैं इसमें सहायता नहीं कर सकता") **most often at low** | hard refusals 14 / 3 / 6 | Yes |

Quick readability signal (median words per sentence, low → high): HiMed 13 → 19 → 19 and Apollo 19 → 23 → 29.
Apollo's text is meaningless, so **readability metrics can "improve" on garbage**. Degenerate outputs have to be removed
before readability is scored.

**Accuracy against the answer key is close to zero for every model.** The correct option text appears in 0–2 of 60 responses
per condition. On the items read by hand, no model gave the reference answer (for example, HiMed recommends "कड़े परिश्रम, ठंडे पानी स्नान" at low and
"take opium with rose water, then reduce the dose" at high for the opium-withdrawal item). Two things cause this. First, HiMed-Trad asks about specialised traditional-medicine
knowledge. Second, some reference answers are themselves noisy machine translations: "सदु कोष" is probably a garbled "मृदु कोष्ठ", and "शाकमिया" is not
a recognisable term.

### Consequences for the analysis design

1. **Floor effect.** When accuracy is ~0% at every level, "does accuracy hold from high to low?" can't be tested,
   because 0 vs 0 carries no information. Gold accuracy stays in the report as a *knowledge floor*, but it can't be the main accuracy measure.
2. **Add cross-level consistency** as the main *accuracy-preservation* metric. It asks: does the model's low-level explanation make the
   same core claim as its own high-level explanation? That is what "preserving accuracy while simplifying" means, and
   it doesn't depend on the model knowing Ayurveda. The opium example above is a clear failure: the claims contradict each other across levels.
3. **Usability is a headline result**, not preprocessing. For 3 of 5 models most outputs aren't usable. Report the rate of usable responses
   per model × level before anything else.
4. **Refusal is its own category.** Nanda refuses more often when asked to explain simply. That's an access/fairness finding: the
   low-literacy user gets less help.
5. **Separate generation artifacts from model behaviour.** HiMed high was cut off by `max_new_tokens`. Decoding also isn't the same
   across models (Airavata/Apollo greedy at 256 tokens; the others sample at 512). Fix what can be fixed cheaply (see Step 2b)
   and write the rest up as a limitation.

### Mapping the research question to metrics

RQ: *Can LLMs adapt the linguistic complexity of Hindi medical explanations to literacy levels **while preserving accuracy**?*

| RQ component | Primary metric | Secondary metrics | Method |
|---|---|---|---|
| Does the model produce a usable answer? | **Usable-response rate** | Refusal / echo / degenerate / truncated / non-Hindi rates | Rules (deterministic) |
| Does the language get simpler at low? (adaptation) | **Paired shift from high to low** in interpretable features, plus the **share of items ordered correctly** (low < mid < high) | RH1/RH2, length | Deterministic |
| Does the content survive simplification? (preservation) | **Cross-level consistency** (low vs high: SAME / PARTIAL / CONTRADICTS) | Key-fact coverage per level | LLM judge + human validation |
| Is the content correct? (knowledge floor) | **Blind option-mapping accuracy** (A–E / NONE) | Fact SUPPORTED / CONTRADICTED / ABSENT | LLM judge |
| Is it safe? | **Potentially-harmful-advice flag** | none | LLM judge flags, all confirmed by hand |

---

### Step 2b: Cheap reruns before analysis (cluster)
- **HiMed-8B high**: rerun with `max_new_tokens=1024`. 19/60 responses are cut off inside `## Thinking`, so the high condition
  currently measures the token budget rather than the model. Rerunning all 3 HiMed levels at 1024 keeps the conditions comparable.
- Optional: Nanda at 1024 as well (mid/high are often cut off, so length is capped).
- Keep Airavata/Apollo as they are and write up their failures. Their degeneration also happened on the curated and West tracks, so it's
  a stable finding, not a fluke.

### Step 3: Clean outputs and gate usability (`literacy_eval/analysis/clean.py`)
- Extraction: for HiMed, score the text after `## Response` (what the user sees). Keep the Thinking text in a separate column.
  Handle the `## Thinking1` variant.
- Flags (each a boolean column, checked in this order; the first match becomes `failure_type`):
  - `refusal`: regex on "सहायता नहीं कर सकता / मदद नहीं कर सकता / असमर्थ हूं"; check the matches by hand once
  - `echo`: the start of the response is ≥0.7 similar to the question stem (`difflib`), with nothing substantial after it
  - `degenerate`: repeated-4-gram share > 0.3, **or** a run of digits/commas (MMed "1,2,3,…")
  - `non_hindi`: Devanagari share of letters < 0.5
  - `too_short`: < 15 Hindi words after cleaning
  - `truncated`: no sentence-final punctuation at the end (informational only, doesn't make the response unusable)
- `usable = not any(refusal, echo, degenerate, non_hindi, too_short)`.
- Calibrate the thresholds on ~30 hand-labelled responses before freezing them, and report how many rows each flag catches.
- Readability and vocabulary are computed **on usable rows only**. For accuracy, unusable rows count as "no correct answer".

### Step 4: Adaptation metrics, deterministic (`readability.py`, `vocab.py`)
Primary features, all **normalised per word or per sentence** so that length doesn't drive them:
- `ASL`: words per sentence (split on `। ? ! .` and newlines)
- `AWL`: visual units (aksharas) per word. This is the strongest single feature in Sinha et al. (2012)
- `PSW_rate`: share of words with ≥3 syllables
- `JUK_rate`: conjuncts (consonant + `्` + consonant) per word
- `rare_rate`: share of Hindi tokens with `wordfreq.zipf_frequency(tok, "hi") < 3` (plain-language vocabulary)
- `latin_rate`: share of Latin-script tokens (English terms; the low prompt forbids them)
- `copy_rate`: share of content tokens copied from `topic_hi`. The technical wording of the question can carry into "low" answers
- `n_words`: reported, but **not** treated as adaptation evidence. It's capped by `max_new_tokens`, and HiMed low is short because
  it gives less information, not because the language is simpler

Secondary: **RH1 = −2.34 + 2.14·AWL + 0.01·PSW** and **RH2 = 0.211 + 1.37·AWL + 0.005·JUK** (Sinha et al. 2012,
checked against the paper, Table 5; lower is easier). Caveats to state: PSW and JUK are **raw counts** in the formula, so longer texts
score as harder. Compute them per 100 words and say you did so. The formulas were fitted on 24 adult readers and 16 texts
(R² 0.44 / 0.36), so treat them as rough indices, not grade levels.

**Anchoring (no grade claims).** Run the same features on a small **reference corpus of Hindi texts with known levels**, e.g.
health-related passages from NCERT Class 5 EVS (आस-पास) and Class 10 Science (विज्ञान) Hindi-medium textbooks, ~10 passages each.
Then you can write "HiMed-8B's low outputs have AWL/rare-word rates between the Class-5 and Class-10 NCERT passages", which is
honest, easy to follow, and needs no validated grade formula.

### Step 5: Accuracy and consistency via an LLM judge (`judge.py`)
Three judge tasks. Each is a separate call returning JSON, at temperature 0, with **instructions in English and content in Hindi**,
and every verdict must **quote the span from the response** that supports it (this makes checking by hand quick).

1. **Option mapping** (blind, as before): response + options A–E → A–E / NONE. Metric: accuracy against the key.
   *Report the NONE rate separately from the wrong-answer rate.* "Committed to a wrong answer" and "gave no answer" are different.
   To check position bias, run the judge again with the options shuffled on a 20% subsample.
2. **Key-fact check** (as before): SUPPORTED / CONTRADICTED / ABSENT per fact. Note: 27/125 key facts were written by the reviewer
   rather than taken from the CoT (e.g. "मृदु कोष्ठ में पित्त दोष की प्रधानता"). Add a `fact_source ∈ {cot, reviewer}` field to the
   reference JSON and report results for both groups.
3. **Cross-level consistency (NEW, primary preservation metric)**: given the same model's **high** and **low** responses to an item
   (order randomised; the judge isn't told which is which), extract the core answer from each, then label
   `SAME` / `PARTIAL` (low leaves out claims but contradicts nothing) / `CONTRADICTS`. Run it on pairs where both responses are usable.
   Also run it on mid–high pairs.
4. **Harm flag (NEW)**: `none / possibly_harmful` + reason (e.g. dosing instructions for opium, advising against seeing a doctor,
   dangerous self-treatment). This is rare, so **check every flagged row by hand** and report the counts.

Judge model: use the strongest judge you can afford. Studies of multilingual LLM judges find weak agreement across languages (average
Fleiss' κ ≈ 0.3) and poorer reliability in lower-resource languages (Fu & Liu, 2025). That's why Step 6 is required, not optional.
Budget: ~720 option/fact calls + ~240 consistency calls + 900 harm calls (the harm check can be merged into call 2 to save quota).

### Step 6: Validate the judge (`validation.py`)
- A stratified sample of 40 responses plus 30 consistency pairs, labelled blind by the user (and ideally a second Hindi speaker on a subset).
- Report percent agreement **and** Cohen's κ per task. Report κ between the two humans on the overlap as the ceiling.
- If κ < 0.6 for a task, revise the prompt and relabel **new** items. Don't tune the prompt against the same validation set.

### Step 7: Statistics and reporting (`aggregate.py`)
Items are **paired** (the same 60 items at every level), so use paired tests:
- Adaptation, per model and feature: the hypothesis is an **ordered** trend (low < mid < high), so use **Page's L trend test**
  (`scipy.stats.page_trend_test`), which is more powerful than Friedman for an ordered hypothesis. Report **Wilcoxon signed-rank low vs high**
  with an effect size (matched-pairs rank-biserial r). Also report the share of items where low < high, which is easy to interpret.
- Consistency, per model: % SAME / PARTIAL / CONTRADICTS with bootstrap 95% CIs (resampling items).
- Usability/refusal across levels: Cochran's Q (paired binary, 3 levels), then McNemar for low vs high.
- Gold accuracy: report it with CIs. With accuracy near 0%, **don't run McNemar**; say that it sits at the floor.
- Multiple comparisons: Holm correction within each family (features × models).
- Caveat: models with few usable rows (e.g. Airavata) get "n too small" instead of a p-value. Make the usable n visible in every table.

**Main results table:** one row per model × level, with usable %, refusal %, ASL, AWL, rare %, latin %, RH1, consistency-with-high
% (low/mid rows only), option accuracy, and harm count.
**Main figure:** for each model, the feature shift from high to low (x) against the consistency rate (y). Good adaptation
with preserved content lands top-right. That picture answers the RQ directly.

### Optional (strengthens the paper)
- **Neutral control condition**: the same 60 items with *no literacy instruction* (300 more generations). Without it you can't say
  whether "low" made the text simpler than the model's default, or only that low differs from high.
- **Small human comprehensibility check**: 2–3 Hindi speakers rank low/mid/high for 15 items (a pairwise "which is easier?"
  question is far more reliable than absolute grade ratings). This validates that the automatic features follow perceived difficulty.
- Run the same deterministic pipeline on the curated-30 and West outputs, which checks that the adaptation findings hold beyond one dataset.

## Critical files
- New: `literacy_eval/analysis/{clean,readability,vocab,judge,validation,aggregate}.py`, `literacy_eval/reference/readability_anchor/`
  (NCERT passages), `tests/test_literacy_analysis.py`
- Modified: `models/himed.py` (`max_new_tokens`), `himed_trad_reference.json` (`fact_source`), `requirements.txt` (`wordfreq`, `scipy>=1.11`),
  `context.md`
- Reused: `models/gemini.py` (client and rate limiting), the resume pattern from `run_eval.py`

## Risks and notes
- **Domain**: Trad items test specialised traditional-medicine knowledge. Gold accuracy mostly shows that knowledge gap, which is why
  consistency is the main preservation metric.
- **Reference noise**: some Trad options and CoTs are noisy machine translations. Note which items are affected during judge validation, and report
  accuracy with those items excluded as a robustness check.
- **Decoding confound**: greedy/256 (Airavata, Apollo) vs sampling/512 (others). Report it. Don't read the differences between models as
  purely a matter of capability.
- **Truncation**: Nanda is often cut off. Per-word features are unaffected. Length and fact coverage are affected, so note it.
- Five options gives a 20% random baseline for mapped accuracy, and NONE counts as wrong.

## Verification
- `pytest tests/`: flag rules on hand-made examples (echo, loop, digit run, refusal), akshara/conjunct counting on known words,
  RH1/RH2 on a hand-computed example, Thinking/Response extraction including `## Thinking1` and missing `## Response`, and judge JSON parsing with a mocked client.
- Flag audit: print 5 random rows per flag and read them.
- Sanity checks: HiMed-8B shows ASL/AWL rising from low to high (seen in the quick look). Apollo's "trend" should disappear once degenerate rows
  are removed. NCERT Class 5 passages should score easier than Class 10 on AWL and rare_rate. If they don't, the features are broken.
- Judge dry run on 5 items per task, checked by hand, then Step 6 before the full run.

## References
- Sinha, Sharma, Dasgupta & Basu (2012). *New Readability Measures for Bangla and Hindi Texts.* COLING Posters. https://aclanthology.org/C12-2111/
- Sinha & Basu (2014). *Text Readability in Hindi: A Comparative Study of Feature Performances Using Support Vectors.* ICON. https://aclanthology.org/W14-5134/
- Devaraj et al. (2022). *Evaluating Factuality in Text Simplification.* ACL. https://aclanthology.org/2022.acl-long.506/ (error taxonomy: insertions, omissions, substitutions)
- Fu & Liu (2025). *How Reliable is Multilingual LLM-as-a-Judge?* Findings of EMNLP. https://arxiv.org/abs/2505.12201

---

## Revision 3 (2026-10-01): implementation plan written

The HiMed-8B high level was rerun at `max_new_tokens=1024`, and all 60 answers now reach `## Response`. Steps 3–7 above are turned into tested,
task-by-task code in **`literacy_eval/docs/2026-10-01-analysis-implementation-plan.md`**. Corrections made there while prototyping
against the real outputs:
- `too_short` is < 3 Hindi words, not 15: HiMed's short low-level answers are real answers.
- Echo needs the answer to still be a question (or add < 3 new words). There is also an instruction-echo rule, and refusal requires < 100 words.
- The judge uses **two separate calls**: blind option mapping, then a fact check. Showing the key facts in the same call would reveal the answer.
- AWL uses Sinha et al.'s visual units. Aksharas are used only for the polysyllable count.
- Validation needs a Hindi-reading labeller. The judge also returns English summaries so the results can be checked without reading Hindi.
- Preliminary (rule-gated) results: HiMed shortens sentences at low (ASL 13 → 24, r = 0.95) but barely simplifies vocabulary.
  Nanda keeps its sentence length but shifts vocabulary slightly, and refuses most at low (13/60; Cochran's Q p = 0.019).

---

## Deterministic baseline results (2026-10-06)

Constraints: no Hindi-reading labeller and no paid model, so the LLM judge (Steps 5–6) is deferred. Everything below is
computed from the text alone by `literacy_eval/analysis/{clean,features,stats,aggregate}.py` (62 tests). Full tables are in
`literacy_eval/analysis_outputs/tables/`, and `results.md` has all of them in one file. "Usable" is the rule-based gate, which is an **upper bound**:
the rules can't catch fluent nonsense.

**1. Usable output (out of 60, low / mid / high):** HiMed-8B 58 / 58 / 60, Nanda 47 / 57 / 54, Apollo 32 / 20 / 14,
MMed-Llama 23 / 21 / 29, Airavata 7 / 10 / 19. Apollo becomes *less* usable as the level rises, because its repetition loops increase.
Nanda refuses 13 times at low against 3 at mid and 6 at high (Cochran's Q p = 0.019).

**2. Adaptation (paired, items usable at all 3 levels, Holm-corrected Page's L):**
- HiMed-8B (n = 56) changes **sentence length**: 13 → 20 → 24 words. r = 0.95, and 89% of items have shorter sentences at low.
  But only 50% of items are monotonic, because mid ≈ high. Vocabulary shifts are small: conjuncts r = 0.41 (p = 0.015), rare words r = 0.32 (p = 0.043).
  Word length and RH1/RH2 show no significant change.
- Nanda (n = 40) changes **vocabulary, not sentence length**: rare words r = 0.63 (p = 0.002), word length r = 0.63 (p = 0.007),
  conjuncts r = 0.50 (p = 0.004), RH2 r = 0.62 (p = 0.007). Sentence length shows no significant change (20 → 21 → 21).
- The other three models have too few usable items for the trend to be interpreted.

**3. Following the level instructions:**
- Low says "5–10 word sentences". **No model does this.** At low, the share of answers where most sentences are 5–10 words is
  HiMed 26%, MMed 22%, Airavata 14%, Apollo 3%, Nanda 2%.
- Low says "no English words". HiMed complies in 100% of answers, Nanda in 87%.
- Low says "no medical terms". HiMed uses a reference-item term in 19% of low answers, compared with 50–53% at mid/high. This is the
  direction the instruction asks for. Nanda's term use is 32% → 37% → 48%.
- Mid says "explain medical terms". Parenthetical glosses: HiMed 0% → 12% → 30% of answers, so they peak at **high**, not mid.
  Nanda's are 49% → 83% → 82%.

**Caveats:** the rule-based usability gate is an upper bound. The term lexicon (288 words) is built from machine-translated reference text, so it
includes noisy terms. Truncation (Nanda 14–17 at mid/high, and Apollo/MMed/Airavata often) reflects token limits and is reported
separately. Decoding settings differ across models. Accuracy and content preservation are **not yet measured**.
