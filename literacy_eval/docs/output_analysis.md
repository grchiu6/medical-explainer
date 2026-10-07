# Baseline Output Analysis & Next Steps

## Data Quality Summary

All 6 files are complete — **30 rows × 6 files = 180 responses**. Schema is correct. Here's what the data actually looks like:

| File | Rows | Median length | #Too short (<100 chars) | #Repetition loops |
|------|------|--------------|------------------------|-------------------|
| airavata/low | 30 | 523 chars | **12** | 5 |
| airavata/mid | 30 | 631 chars | **9** | 7 |
| airavata/high | 30 | 653 chars | 3 | 9 |
| apollo/low | 30 | 651 chars | 0 | **18** |
| apollo/mid | 30 | 663 chars | 0 | **16** |
| apollo/high | 30 | 636 chars | 0 | **22** |

---

## What's Working

**Airavata — high level** works best. When the topic is clear, it produces reasonable, coherent medical text in Hindi at professional register. Example (`depression/high`):
> *"अवसाद एक मानसिक स्थिति है जिसमें व्यक्ति उदासी, निराशा और निराशा की भावना महसूस करता है..."*

**Apollo — mid level** is the cleanest Apollo output. It consistently produces structured explanations with parenthetical term definitions, e.g.:
> *"मधुमेह एक ऐसी बीमारी है जो शरीर में रक्त शर्करा (sugar) के स्तर को नियंत्रित नहीं करने के कारण होती है..."*

---

## Failure Modes Found

### 1. Airavata: Prompt Echo (12–9 responses across low/mid levels)
Airavata frequently just **repeats the question or part of the prompt** instead of answering. Example (`tuberculosis/low`):
> *"तपेदिक (टी. बी.) क्या है, यह कैसे फैलता है और इसका इलाज कैसे किया जाता है?"* — that's just the question back.

**Root cause:** Airavata is a relatively weak instruction follower for novel tasks. When it doesn't know what to do with a complex Hindi prompt, it defaults to echoing. This is most common at `low` level (where the prompt has the most unusual linguistic constraints).

### 2. Apollo: Repetition Loops (16–22 responses per level)
Apollo produces decent opening sentences but then **degenerates into looping repetition**. Example (`diabetes/low` tail):
> *"...रोज़मर्रा के लिए एक व्यक्ति के लिए रोज़मर्रा के लिए एक व्यक्ति के लिए रोज़मर्रा के लिए..."* (repeated 14+ times)

**Root cause:** Apollo's generation was set to `max_new_tokens=256` with `do_sample=False` (greedy decoding). Greedy decoding on a model not well-suited to the task gets stuck in a probability loop. This is a well-known failure mode of greedy decoding.

### 3. Airavata: Prompt Echo of Full System Prompt (some low/mid responses)
For `depression/low`, Airavata literally output the system prompt back as its "answer" — it reproduced the instructions verbatim then tried to re-read them. **The model didn't understand the task boundary.**

### 4. Apollo: Medical Factual Errors
Apollo/mid for tuberculosis says it spreads *"by mosquito bite"* — that's malaria, not TB. This isn't a failure mode for the baseline collection (you're not evaluating accuracy yet), but it's an important note for the accuracy evaluation step later.

---

## What This Tells You About the Research Question

Even with these failure modes, the data is **research-useful**. The failures themselves are a finding:

- **Airavata struggles more with low-literacy register** (12 near-failures at low vs. 3 at high). This suggests it handles technical medical language better than simplified language — which is the *opposite* of what you'd want for the research goal.
- **Apollo degenerates equally across all 3 levels** (18-22 loops per level). This suggests it isn't adapting register at all — it's doing the same thing regardless of instruction.
- **Neither model shows a clear pattern of actually adapting linguistic complexity** across the 3 levels in the way the research question envisions.

This is exactly what a baseline should show: **current off-the-shelf models are not good at this task**, which motivates your research.

---

## Recommended Next Steps (Prioritized)

> [!IMPORTANT]
> **My strong recommendation: proceed in this order — fix the generation issues → add more models → then do analysis. Don't add more topics or more levels yet.**

### Priority 1: Fix the Generation Issues (before doing anything else)

Two quick fixes that will dramatically improve output quality:

**Fix A — Apollo repetition: add a repetition penalty**

In `models/apollo.py`, change the `.generate()` call:
```python
outputs = _model.generate(
    **inputs,
    max_new_tokens=512,          # increase — some topics need more tokens
    do_sample=True,              # enable sampling to break loops
    temperature=0.3,             # low temperature = mostly deterministic but no loops
    repetition_penalty=1.3,      # penalize repeating the same n-grams
)
```

**Fix B — Airavata prompt echo: strengthen the prompt format**

Airavata's chat format expects `<|user|>` tags. The current code in `airavata.py` wraps the prompt in those tags, but Airavata may not be distinguishing the meta-instructions from the content well. A stronger instruction boundary helps:
```
<|user|>
नीचे दिए गए निर्देशों का पालन करें और सीधे व्याख्या लिखें। प्रश्न को दोहराएं नहीं।

[your instruction here]
<|assistant|>
व्याख्या:
```

### Priority 2: Expand Models (high value, relatively easy)

You currently have 2 models. For a research paper, 4-6 is the standard. Here are the most suitable candidates given your research question:

| Model | Why it's relevant | Where to get it | GPU needed |
|-------|-------------------|-----------------|------------|
| **Gemma-2-9B-IT** (Google) | Strong Hindi multilingual, good instruction following, likely to handle register well | HuggingFace | 24GB |
| **Llama-3.1-8B-Instruct** (Meta) | State-of-the-art open-source general model, best instruction follower in this size class | HuggingFace | 16GB |
| **MMed-Llama-3-8B** (FreedomIntelligence) | Medical-domain fine-tune of Llama 3, supports Hindi — directly comparable to Apollo | HuggingFace | 16GB |
| **IndicBART / IndicGENERATION** (AI4Bharat) | India-specific multilingual model, trained heavily on Hindi | HuggingFace | 8GB |

> [!NOTE]
> I'd recommend starting with **Llama-3.1-8B-Instruct** and **Gemma-2-9B-IT** as the next 2 models. Both are strong instruction followers (which is the core capability being tested), and they contrast nicely with Airavata (Hindi-specialist) and Apollo (medical-specialist). Together that gives you a 4-model baseline covering: general-multilingual, Hindi-specialist, medical-specialist (6-language), and medical-specialist (Llama-based).

### Priority 3: Analysis (do this after Priority 1 & 2)

Once you have cleaner outputs from 4+ models, the analysis phase should:

1. **Automated readability scoring:** Hindi-adapted readability metrics. Options:
   - Syllable-count based Flesch-Kincaid equivalent adapted for Hindi
   - Average word length / sentence length as proxy metrics
   - Vocabulary richness (type-token ratio across levels)

2. **Vocabulary-level analysis:** Build or find a Hindi word frequency list graded by complexity. Count how many "hard" vs "easy" words appear in each level's output.

3. **Cross-level consistency:** For the same topic, does the model produce noticeably different text across levels? You can measure this with embedding cosine similarity — outputs at low/high should be far apart if the model is truly adapting.

4. **Failure rate by topic category:** Does the model fail more on certain medical domains (e.g., psychiatric topics vs. infectious disease)? The data you have already supports this analysis.

### What NOT to Do Right Now

- **Don't add more topics yet.** 30 is enough to find patterns. More topics add noise before you understand the signal.
- **Don't add more literacy levels yet.** 3 is the right number for an initial study. If analysis shows the gap between low and mid is negligible, you can split one later.
- **Don't do human evaluation yet.** It's expensive and you need cleaner outputs first.

---

## Summary Table

| Action | Effort | Value | Do Now? |
|--------|--------|-------|---------|
| Fix Apollo repetition penalty | 5 min | High | ✅ Yes |
| Fix Airavata prompt echo | 15 min | High | ✅ Yes |
| Add Llama-3.1-8B-Instruct | 1-2 hrs | High | ✅ Yes |
| Add Gemma-2-9B-IT | 1-2 hrs | High | ✅ Yes |
| Add MMed-Llama-3-8B | 1-2 hrs | Medium | Later |
| Readability/vocab analysis | 1-2 days | High | After fixes |
| Add more topics (30→50) | Low | Low | Not yet |
| Add more levels (3→5) | Medium | Low | Not yet |
| Human evaluation | High effort | High | Much later |
