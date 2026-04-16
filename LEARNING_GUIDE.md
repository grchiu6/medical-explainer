# Baseline Evaluation Pipeline — Learning Guide

Everything we built, why we built it, and the concepts behind it. Use this as a reference when working on future tasks or when your PhD student asks "why did you do it this way?"

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Task 1: Project Scaffolding](#task-1-project-scaffolding)
3. [Task 2: Dataset Download](#task-2-dataset-download)
4. [Task 3: Data Loader](#task-3-data-loader)
5. [Task 4: Shared Prompt Template](#task-4-shared-prompt-template)
6. [Task 5: Gemini Model Integration](#task-5-gemini-model-integration)
7. [Task 6: Airavata Model Integration](#task-6-airavata-model-integration)
8. [Task 7: Apollo Model Integration](#task-7-apollo-model-integration)
9. [Concepts Reference](#concepts-reference)

---

## Project Overview

### What this project does
Evaluates 3 existing LLMs on Hindi QA to establish "before" numbers (baselines). Later, we'll fine-tune a model on Hindi medical data and compare its scores to these baselines. The improvement is the research contribution.

### The 3 models we're comparing

| Model | Type | Where it runs | What it's good at |
|---|---|---|---|
| Gemini 2.5 Flash | Commercial API | Google's servers (internet) | Strong general-purpose, multilingual |
| Airavata | Open-source, local | HPC cluster GPU | Hindi language specifically |
| Apollo-7B | Open-source, local | HPC cluster GPU | Medical QA across 6 languages including Hindi |

### The dataset: IndicQA Hindi
- 1,547 total QA pairs from AI4Bharat (same team as Airavata)
- 1,052 answerable questions (we filter out the rest)
- SQuAD format: each question has a context passage and a reference answer
- General Hindi QA (not medical-specific, but has clean reference answers for automatic scoring)

### Project structure
```
Baseline Evaluation/
├── .env                    # Your API keys (secret, gitignored)
├── .env.example            # Template showing what keys are needed
├── .gitignore              # Tells git which files to ignore
├── requirements.txt        # All Python dependencies
├── data/
│   ├── __init__.py         # Makes data/ a Python package
│   ├── download.py         # Downloads IndicQA from HuggingFace
│   ├── loader.py           # Reads JSON, returns flat list of QA dicts
│   └── indicqa_hi.json     # The actual dataset (downloaded)
├── models/
│   ├── __init__.py         # Makes models/ a Python package
│   ├── base.py             # Shared prompt template (all models use this)
│   ├── gemini.py           # Gemini API integration
│   ├── airavata.py         # Airavata local GPU model
│   └── apollo.py           # Apollo local GPU model
├── tests/
│   ├── __init__.py         # Makes tests/ a Python package
│   ├── test_loader.py      # Tests for the data loader
│   ├── test_base_prompt.py # Tests for the prompt template
│   ├── test_gemini.py      # Tests for Gemini (needs API key)
│   ├── test_airavata.py    # Tests for Airavata (needs GPU)
│   └── test_apollo.py      # Tests for Apollo (needs GPU)
├── outputs/                # Where model answers and scores go
├── scripts/
│   └── run_local_model.sh  # SLURM script for HPC cluster (TODO)
├── evaluate.py             # Main runner script (TODO)
├── score.py                # BLEU/ROUGE scoring (TODO)
└── results_table.py        # Print comparison table (TODO)
```

---

## Task 1: Project Scaffolding

### What we did
Created the directory structure, requirements.txt, .env.example, .gitignore, set up a virtual environment, and initialized a git repo.

### Key files explained

**`requirements.txt`** — Lists every Python library the project needs. Anyone who clones your project can run `pip install -r requirements.txt` to get everything set up.

```
google-genai>=1.0.0         # Talk to Gemini API
transformers>=4.40.0        # Load HuggingFace models (Airavata, Apollo)
torch>=2.0.0                # PyTorch — the deep learning framework
accelerate>=0.30.0          # Speeds up model loading
pandas>=2.0.0               # Data manipulation (reading CSVs, etc.)
python-dotenv>=1.0.0        # Read .env files for secrets
rouge-score>=0.1.2          # ROUGE metric for scoring
nltk>=3.8.0                 # BLEU metric for scoring
kagglehub>=0.3.0            # Download datasets from Kaggle
datasets>=2.18.0            # HuggingFace datasets library
pytest>=8.0.0               # Testing framework
```

**`.env.example` vs `.env`** — `.env` holds your real secrets (API keys). `.env.example` is a template with placeholder values that you commit to git. This way, other people know what keys they need without seeing YOUR keys.

**`.gitignore`** — Tells git "don't track these files." We ignore:
- `.env` (secrets — NEVER commit real API keys)
- `data/himed/` (large data files — re-downloadable)
- `outputs/` (regeneratable results)
- `venv/` (virtual environment — recreatable from requirements.txt)
- `__pycache__/` (Python's compiled cache — auto-generated)

### Concepts learned

**Virtual environment (`venv`):** An isolated Python installation for your project. Without it, installing a library for one project could break another project. Think of it like a sandbox — each project gets its own sandbox with its own toys.
```bash
python3 -m venv venv          # Create the virtual environment
source venv/bin/activate       # Activate it (now 'python' and 'pip' use this venv)
pip install -r requirements.txt  # Install dependencies INTO this venv
```
**When to use:** ALWAYS. Every Python project should have its own venv. It's the first thing you set up.

**Git init and first commit:**
```bash
git init                       # Create a new git repository
git add file1 file2            # Stage files for commit
git commit -m "message"        # Save a snapshot with a description
```
**When to commit:** After every completed task. Each commit is a "save point" you can go back to.

---

## Task 2: Dataset Download

### What we did
1. Originally planned to use HiMed (403K Hindi medical entries from Kaggle)
2. Downloaded it and discovered it was NOT in QA format — it was a medical encyclopedia (disease descriptions, symptoms, treatments)
3. Switched to IndicQA Hindi from HuggingFace (1,547 QA pairs in SQuAD format)

### The download script (`data/download.py`)

```python
import urllib.request
from pathlib import Path

url = "https://huggingface.co/.../indicqa.hi.json"
urllib.request.urlretrieve(url, output_path)
```

**`urllib.request.urlretrieve(url, path)`** — Python's built-in way to download a file from the internet. No extra libraries needed. First argument is the URL, second is where to save it on your computer.

**`Path` (from `pathlib`)** — A modern, clean way to work with file paths. Instead of string concatenation like `"data" + "/" + "file.json"`, you write `Path("data") / "file.json"`. It handles OS differences (Windows uses `\`, Mac/Linux use `/`) automatically.

### Concepts learned

**Data validation:** Always check your dataset's actual format before building your pipeline. We assumed HiMed was QA pairs (it wasn't). This saved us from writing a bunch of code that wouldn't work.

**When to do this:** FIRST step in any data project. Download → open → look at it → verify it matches what you expected.

**Idempotency:** The download script checks `if output_path.exists()` before downloading. Running the script twice doesn't re-download — it skips. This saves time and bandwidth.

**When to use:** Any script that downloads, generates, or processes files. Always check if the output already exists before doing expensive work.

**SQuAD format:** The standard JSON structure for QA datasets in NLP:
```json
{
  "data": [
    {
      "paragraphs": [
        {
          "context": "passage text...",
          "qas": [
            {
              "question": "question?",
              "answers": [{"text": "answer text", "answer_start": 123}]
            }
          ]
        }
      ]
    }
  ]
}
```
Many QA datasets use this format (SQuAD, IndicQA, TyDi QA, etc.), so knowing how to parse it is useful.

---

## Task 3: Data Loader

### What we did
Created `data/loader.py` — a function that reads the nested SQuAD JSON and returns a flat list of dicts that the rest of the pipeline can use.

### The loader function

```python
def load_indicqa(sample_size=None) -> list[dict]:
    # 1. Read the JSON file
    # 2. Flatten nested SQuAD structure into simple list
    # 3. Filter out unanswerable questions (empty answers)
    # 4. Optionally sample a subset
    # Returns: [{"question": "...", "context": "...", "reference_answer": "..."}, ...]
```

**Key design decision:** The function returns a standardized format regardless of which dataset we loaded. If we add IndicQA or another dataset later, we just write a new loader that returns the same format. The rest of the pipeline doesn't change.

### The tests

**Test 1: `test_load_indicqa_returns_list_of_dicts`**
- Loads 3 items, checks they're dicts with the right keys
- **Why:** Catches structural bugs (misspelled keys, wrong types)
- **When to write this kind of test:** When your function returns structured data — always verify the shape

**Test 2: `test_load_indicqa_sample_size`**
- Checks that `sample_size=5` returns exactly 5 items
- **Why:** Catches bugs in sampling logic (e.g., sampling before filtering)
- **When:** When your function has parameters that affect output size

**Test 3: `test_load_indicqa_filters_unanswerable`**
- Checks that NO item has an empty `reference_answer`
- **Why:** 495 questions have empty answers. If filtering breaks, scoring becomes meaningless
- **When:** When your function filters data — verify the filter actually works

**Test 4: `test_load_indicqa_full_load`**
- Checks total count = 1,052
- **Why:** Sanity check. If this number changes, something is wrong
- **When:** When you know an expected value from manual exploration, hardcode it as a test

### Concepts learned

**Flattening nested data:** The SQuAD JSON has 3 levels of nesting (articles → paragraphs → QAs). Our pipeline needs a flat list. The loader handles this conversion so no other file needs to know about SQuAD format.

**When to flatten:** When your data comes in a complex structure but your processing logic just needs a simple list. The flattening should happen once, early in the pipeline.

**`random.seed(42)`:** Makes random sampling reproducible. Without it, `random.sample()` gives different results each run, making debugging and reproducibility impossible.

**Why 42?** It's an arbitrary number — any number works. 42 is a tradition in CS (from Hitchhiker's Guide to the Galaxy). The important thing is picking one number and sticking with it.

**When to use:** ALWAYS in research code when using randomness. Reproducibility is non-negotiable.

**`__init__.py` files:** Empty files that tell Python "this folder is a package — you can import from it." Without `data/__init__.py`, `from data.loader import load_indicqa` would fail.

**When to create:** Every time you make a new folder that will contain Python files you want to import.

---

## Task 4: Shared Prompt Template

### What we did
Created `models/base.py` with a `build_prompt()` function that all 3 models use.

### The prompt

```
You are a question-answering assistant. Answer the following question
in Hindi based on the provided context. Give a concise, accurate answer.

Context: {context}

Question: {question}

Answer:
```

### Why the instructions are in English, not Hindi
- Gemini and Apollo were primarily instruction-tuned in English
- Airavata handles English instructions fine (its base model was trained on English too)
- Research standard: multilingual evaluations typically use English instructions
- The question and context stay in Hindi because that's the data we're testing

### Concepts learned

**Controlling variables:** When comparing multiple models, keep everything the same except the one thing you're testing (the model itself). If each model got a different prompt, you couldn't tell if score differences came from the model or the prompt.

**When to apply:** Any comparative experiment. Change ONE variable at a time.

**`.format()` (string formatting):**
```python
template = "Hello {name}, you are {age}"
result = template.format(name="Grace", age=22)
# Result: "Hello Grace, you are 22"
```
The `{name}` placeholders get replaced with actual values. We use this to insert the question and context into our prompt template.

**Private functions (underscore prefix):**
```python
_PROMPT_WITH_CONTEXT = "..."  # Private — only used inside this file
def build_prompt():           # Public — other files call this
```
Starting a name with `_` is a Python convention meaning "internal use only." Other files should only call `build_prompt()`, not access the raw template strings directly.

**When to use `_` prefix:** For helper functions and internal variables that outside code shouldn't depend on. It signals "I might change this implementation detail without warning."

---

## Task 5: Gemini Model Integration

### What we did
Created `models/gemini.py` that sends questions to Google's Gemini API and returns answers.

### How API calls work

```
Your code → Internet → Google's servers → Gemini model → answer → Internet → your code
```

You never see or run the model. You just send text and get text back, like texting someone.

### Key code patterns

**Loading .env secrets:**
```python
from dotenv import load_dotenv
load_dotenv()                              # Read .env file
api_key = os.getenv("GEMINI_API_KEY")      # Get the value
```
This keeps secrets out of your code. If you hardcoded `api_key = "abc123"`, anyone reading your code (or your git history) could steal your key.

**When to use:** ALWAYS for API keys, passwords, database URLs, or anything sensitive.

**Lazy initialization (singleton pattern):**
```python
_client = None

def _get_client():
    global _client
    if _client is None:           # First call: create it
        _client = genai.Client(...)
    return _client                # Later calls: return cached one
```
The client is only created when first needed. This avoids wasting time on setup if `get_answer()` is never called.

**When to use:** When initialization is expensive (network connections, model loading, database connections). Don't do expensive work until you actually need the result.

**Exponential backoff (retry logic):**
```python
for attempt in range(3):
    try:
        response = call_api()    # Try the API call
        return response
    except Exception:
        wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
        time.sleep(wait)           # Wait longer each retry
```
If the API fails (network glitch, server busy), wait and try again. Each retry waits longer to give the server breathing room.

**When to use:** Any code that calls an external service (APIs, databases, web requests). Things fail temporarily — retrying with backoff handles that gracefully.

**Rate limiting:**
```python
time.sleep(4.0)  # Wait 4 seconds between API calls
```
Google's free tier allows 15 requests per minute. Without this delay, you'd hit the limit and get blocked. We wait 4 seconds between calls (60/15 = 4).

**When to use:** Any time you're calling a rate-limited API in a loop.

### Debugging journey: What went wrong

1. **Model name `gemini-1.5-pro` didn't exist** — Google renamed their models. Fix: listed available models and picked `gemini-2.5-flash`
2. **Old library `google-generativeai` was deprecated** — Google released a new library `google-genai`. Fix: switched to new library
3. **Rate limit exhausted on `gemini-2.0-flash`** — Each model has separate quotas. Fix: switched to `gemini-2.5-flash` which had available quota

**Lesson:** APIs and libraries change constantly. Read error messages carefully — they usually tell you exactly what to fix. This kind of debugging is 50% of real-world programming.

### Test concepts

**`@pytest.mark.skipif`** — A decorator that conditionally skips a test:
```python
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),    # Condition to skip
    reason="GEMINI_API_KEY not set"      # Message shown when skipped
)
def test_something():
    ...
```
The test only runs if the condition is False. Without the API key, the test is skipped (not failed).

**When to use:** Tests that depend on external resources (API keys, GPUs, network access, specific OS). You don't want tests to "fail" just because of a missing resource.

**Smoke test:** A test that checks "does this work at all?" — not "does this give the right answer?" We verify the API returns *something*, not that the answer is *correct*. Correctness is measured later with BLEU/ROUGE scoring across the full dataset.

---

## Task 6: Airavata Model Integration

### What we did
Created `models/airavata.py` that loads the Airavata model onto a GPU and runs it locally.

### API call vs Local model — the difference

| | API (Gemini) | Local (Airavata/Apollo) |
|---|---|---|
| Where model runs | Google's servers | Your GPU |
| Internet needed? | Yes | No (after first download) |
| Cost | Free tier / paid | Free |
| Speed | Depends on network | Depends on GPU |
| Code complexity | Simple (send text, get text) | Complex (load model, tokenize, generate, decode) |
| Reproducibility | Google may update model | Exact same model every time |

### How local model inference works (the 5-step pipeline)

```
1. Load model onto GPU (once, slow)
2. Tokenize: "मधुमेह क्या है?" → [1234, 5678, 9012]
3. Generate: model processes numbers → outputs new numbers [3456, 7890]
4. Decode: [3456, 7890] → "यह एक रोग है"
5. Return the text answer
```

### Key code explained

**Loading from HuggingFace:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ai4bharat/Airavata")
model = AutoModelForCausalLM.from_pretrained("ai4bharat/Airavata", torch_dtype=torch.bfloat16)
```
- `AutoTokenizer` — automatically picks the right tokenizer for the model
- `AutoModelForCausalLM` — automatically picks the right model architecture
- `from_pretrained("name")` — downloads from HuggingFace (cached after first download)
- `torch_dtype=torch.bfloat16` — use half-precision to save GPU memory (14GB vs 28GB)

**When to use `bfloat16`:** For inference (generating answers). It halves memory usage with minimal quality loss. DON'T use it for training (you might in fine-tuning, but with more care).

**Moving model to GPU:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```
`.to(device)` moves all the model's parameters to the specified device. "cuda" = NVIDIA GPU, "cpu" = regular processor. GPU is 10-100x faster for ML work.

**Tokenization (text → numbers):**
```python
inputs = tokenizer(text, return_tensors="pt")
# inputs["input_ids"] = tensor([[1234, 5678, 9012]])
# inputs["attention_mask"] = tensor([[1, 1, 1]])
```
- `return_tensors="pt"` — return PyTorch tensors (the number format PyTorch uses)
- `input_ids` — the actual token numbers
- `attention_mask` — which tokens are real (1) vs padding (0)

**Generation:**
```python
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
```
- `torch.no_grad()` — tells PyTorch "we're not training, skip gradient tracking" (saves memory)
- `max_new_tokens=256` — generate at most 256 tokens (prevents infinite output)
- `do_sample=False` — deterministic: always pick the most likely next word (no randomness)
- `**inputs` — Python unpacking: passes each key-value pair as a separate argument

**When to use `do_sample=False`:** In research when you need reproducible results. Same input → same output every time.

**Decoding (numbers → text):**
```python
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
```
- The model's output includes BOTH the input prompt AND the new answer
- `outputs[0][input_length:]` — slice off the prompt, keep only new tokens
- `skip_special_tokens=True` — remove `</s>`, `<pad>`, etc. from output

**Airavata's prompt format:**
```
<|user|>
{your prompt here}
<|assistant|>
```
Every model has its own format from instruction tuning. Getting this wrong = bad output. Always check the model card on HuggingFace.

---

## Task 7: Apollo Model Integration

### What we did
Created `models/apollo.py` — structurally identical to Airavata, but with Apollo's model name and prompt format.

### How Apollo differs from Airavata

| | Airavata | Apollo |
|---|---|---|
| Made by | AI4Bharat (IIT Madras) | FreedomIntelligence (CUHK) |
| Focus | Hindi language | Medical QA, 6 languages |
| Base model | OpenHathi / Llama 2 | Gemma |
| Size | ~7B parameters | ~9B parameters |
| Prompt format | `<\|user\|>\n...\n<\|assistant\|>\n` | `User:...\nAssistant:` |
| Best at | Hindi text in general | Medical questions across languages |

**Apollo's prompt format:**
```
User:{your prompt here}
Assistant:
```
Simpler than Airavata — just text prefixes, no special tokens.

### Why we kept separate files instead of one generic file
Each model has its own:
- Model name (for HuggingFace download)
- Prompt format (from its instruction tuning)
- Potentially different tokenizer settings

Keeping them separate makes each file self-contained and easy to understand. You can read `airavata.py` and understand everything about Airavata without looking at any other file. **Clarity beats DRY** when the "repeated" code has meaningful differences.

---

## Concepts Reference

### Programming Patterns

| Pattern | What it is | When to use |
|---|---|---|
| **TDD (Test-Driven Development)** | Write tests first, then code | Always in research code — ensures correctness |
| **Lazy initialization** | Create expensive objects only when first needed | Database connections, model loading, API clients |
| **Exponential backoff** | Retry with increasing wait times | Any external service call (APIs, databases) |
| **Idempotency** | Running something twice gives the same result | Download scripts, data processing pipelines |
| **Singleton** | Only one instance of something exists | Shared resources (model objects, DB connections) |

### Python Concepts

| Concept | Example | What it does |
|---|---|---|
| **`import`** | `from models.base import build_prompt` | Brings code from another file into this one |
| **`__init__.py`** | Empty file in a folder | Makes the folder a Python package (importable) |
| **`_underscore` prefix** | `_PROMPT = "..."` | Convention: "this is private, don't use from outside" |
| **`global`** | `global _model` | Lets a function modify a module-level variable |
| **`assert`** | `assert len(items) == 5` | "This must be true, or crash" (used in tests) |
| **`@decorator`** | `@pytest.mark.skipif(...)` | Modifies the function below it |
| **f-strings** | `f"Hello {name}"` | String with variables inserted using `{}` |
| **`.format()`** | `"Hello {name}".format(name="X")` | Another way to insert variables into strings |
| **List slicing** | `items[5:]` | Everything from index 5 onward |
| **`**kwargs` unpacking** | `func(**{"a": 1, "b": 2})` | Same as `func(a=1, b=2)` |
| **`with` statement** | `with open(f) as file:` | Ensures file is closed when done (resource management) |

### ML/NLP Concepts

| Concept | What it means |
|---|---|
| **Tokenization** | Converting text to numbers so a model can process it |
| **Inference** | Using a trained model to generate predictions (not training) |
| **bfloat16** | Half-precision numbers — halves GPU memory with minimal quality loss |
| **CUDA** | NVIDIA's GPU computing platform — all ML GPUs use it |
| **torch.no_grad()** | Tells PyTorch to skip gradient tracking (saves memory during inference) |
| **do_sample=False** | Deterministic generation — always picks the most likely word |
| **Instruction tuning** | Extra training that teaches a model to follow instructions |
| **Prompt format** | The specific text structure a model expects (varies per model) |
| **SQuAD format** | Standard JSON structure for QA datasets |
| **BLEU** | Metric measuring n-gram overlap between answer and reference |
| **ROUGE-L** | Metric measuring longest common subsequence between answer and reference |
| **Baseline** | The "before" number — current performance before your improvement |
| **Extractive QA** | The answer is a substring directly copied from the context |

### Best Practices Checklist

- [ ] Every project gets its own virtual environment (`venv`)
- [ ] API keys go in `.env`, never in code
- [ ] `.env` is gitignored, `.env.example` is committed
- [ ] Every dependency is listed in `requirements.txt`
- [ ] Add new dependencies to `requirements.txt` immediately when you install them
- [ ] Commit after every completed task (frequent save points)
- [ ] Write tests before code (TDD)
- [ ] Set `random.seed()` for reproducibility in research
- [ ] Use `do_sample=False` for deterministic model output
- [ ] Check dataset format before building your pipeline
- [ ] Use `_underscore` prefix for private functions
- [ ] Add `__init__.py` to every folder you want to import from

### Common Errors You'll See

| Error | What it means | How to fix |
|---|---|---|
| `ModuleNotFoundError` | Python can't find the file you're importing | Check the file exists, check `__init__.py` exists |
| `FileNotFoundError` | The file path doesn't exist | Check spelling, check current directory |
| `429 RESOURCE_EXHAUSTED` | API rate limit hit | Wait and retry, or use a different model/quota |
| `404 Not Found` | URL or model name is wrong | Check the correct name (APIs rename things often) |
| `CUDA out of memory` | Model is too large for your GPU | Use `bfloat16`, use a smaller model, or request a bigger GPU |
| `RuntimeError: No GPU` | Code needs a GPU but none available | Run on HPC cluster, not laptop |

---

*This guide will be updated as we complete remaining tasks (evaluate.py, score.py, results_table.py, SLURM script).*
