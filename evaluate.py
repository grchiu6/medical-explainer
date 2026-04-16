"""Main evaluation runner — ties the whole pipeline together.

THIS IS THE SCRIPT YOU RUN. It connects the data loader, model files,
and output saving into one workflow.

Usage:
    python evaluate.py --model gemini --sample 5   # test with 5 questions
    python evaluate.py --model gemini              # full run (all questions)
    python evaluate.py --model airavata            # run on HPC cluster
    python evaluate.py --model apollo              # run on HPC cluster

What it does:
    1. Loads the IndicQA Hindi dataset (via data/loader.py)
    2. Loads the specified model's get_answer() function
    3. Loops through each question, calls get_answer(), saves the result
    4. Saves everything to outputs/{model}_raw.csv

KEY FEATURES:
- Incremental saving: Each answer is written to disk immediately.
  If the script crashes at question 500, questions 1-499 are safe.
- Resume support: If you re-run the script, it skips questions that
  already have answers in the CSV. No wasted API calls or GPU time.
"""

import argparse
import csv
import importlib
from pathlib import Path
from typing import Callable

from data.loader import load_indicqa

# Where output CSVs go. Created automatically if it doesn't exist.
OUTPUTS_DIR = Path(__file__).parent / "outputs"

# The column names for our output CSV
FIELDNAMES = ["question", "context", "reference_answer", "model_answer"]


def _load_existing_questions(output_path: Path) -> set[str]:
    """Load questions that were already answered in a previous run.

    Reads the CSV file and returns a set of question strings that
    already have answers. We use a set (not a list) because checking
    "is this question in the set?" is instant with a set, but slow
    with a list. For 1,052 questions this doesn't matter much, but
    it's a good habit.

    When you'd use a set vs list:
    - Set: when you need fast lookups ("is X in here?") and don't
      care about order
    - List: when you need ordered items or duplicates

    Args:
        output_path: Path to an existing CSV file from a previous run.

    Returns:
        Set of question strings that are already answered.
    """
    if not output_path.exists():
        return set()

    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["question"] for row in reader}
    # The {... for row in reader} syntax is a "set comprehension" —
    # it creates a set by looping through the CSV rows and extracting
    # the "question" column from each one.


def run_evaluation(
    model_name: str,
    get_answer_fn: Callable[[str, str], str],
    sample_size: int | None,
    output_path: Path | None = None,
) -> None:
    """Run a model on the IndicQA dataset and save results to CSV.

    This is the core function. It:
    1. Loads data
    2. Checks what's already been answered (resume support)
    3. Loops through remaining questions
    4. Calls the model for each question
    5. Saves each answer to the CSV immediately (incremental saving)

    Args:
        model_name: Name of the model (used for the output filename).
        get_answer_fn: A function that takes (question, context) and
                       returns an answer string. This is the model's
                       get_answer() function.
        sample_size: Number of questions to evaluate, or None for all.
        output_path: Where to save results. If None, uses
                     outputs/{model_name}_raw.csv.

    The Callable[[str, str], str] type hint means:
    "A function that takes two strings and returns a string."
    This is exactly the signature of get_answer() in all our model files.
    """
    if output_path is None:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        output_path = OUTPUTS_DIR / f"{model_name}_raw.csv"

    # Step 1: Load dataset
    data = load_indicqa(sample_size=sample_size)

    # Step 2: Check what's already been answered (resume support)
    already_done = _load_existing_questions(output_path)
    remaining = [item for item in data if item["question"] not in already_done]

    if not remaining:
        print(f"All {len(data)} questions already answered. Nothing to do.")
        return

    print(
        f"Total questions: {len(data)}, "
        f"already done: {len(already_done)}, "
        f"remaining: {len(remaining)}"
    )

    # Step 3: Open the CSV for appending (not overwriting!)
    # "a" mode = append (add to end of file)
    # If the file doesn't exist, "a" mode creates it.
    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        # Write the header row only if this is a new file
        if not file_exists:
            writer.writeheader()

        # Step 4: Loop through each question
        for i, item in enumerate(remaining, 1):
            print(f"[{i}/{len(remaining)}] Asking {model_name}...")

            # Call the model — this is where the magic happens!
            # For Gemini: sends an API request over the internet
            # For Airavata/Apollo: runs inference on the GPU
            answer = get_answer_fn(item["question"], item["context"])

            # Step 5: Save this answer to the CSV immediately
            writer.writerow({
                "question": item["question"],
                "context": item["context"],
                "reference_answer": item["reference_answer"],
                "model_answer": answer,
            })

            # f.flush() forces Python to write the data to disk RIGHT NOW
            # instead of buffering it in memory. Without this, if the
            # script crashes, the last few answers might be lost because
            # they were still in the buffer.
            f.flush()

    print(f"Done. Results saved to {output_path}")


def _get_model_function(model_name: str) -> Callable[[str, str], str]:
    """Dynamically import the get_answer function for a given model.

    Instead of writing:
        if model_name == "gemini": from models.gemini import get_answer
        elif model_name == "airavata": from models.airavata import get_answer
        ...

    We use importlib to import the module by name at runtime.
    This means adding a new model DOESN'T require changing evaluate.py —
    just create a new file in models/ and add its name to the choices.

    This pattern is called "dynamic dispatch" — the code decides which
    module to use based on a string at runtime, not hardcoded at write time.

    When you'd use this: When you have multiple interchangeable modules
    that all have the same interface (like our model files all having
    get_answer(question, context) -> str).

    Args:
        model_name: The model name (matches the filename in models/).

    Returns:
        The get_answer function from that model's module.
    """
    module = importlib.import_module(f"models.{model_name}")
    return module.get_answer


def main():
    """Parse command-line arguments and run the evaluation.

    argparse is Python's built-in library for command-line interfaces.
    It turns:
        python evaluate.py --model gemini --sample 50
    Into:
        args.model = "gemini"
        args.sample = 50

    It also auto-generates help text:
        python evaluate.py --help

    When you'd use argparse: Any script that needs to accept options
    from the command line. It's the standard way to build CLI tools
    in Python.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the IndicQA Hindi dataset"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemini", "airavata", "apollo"],
        help="Which model to evaluate",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of questions to sample (default: all)",
    )
    args = parser.parse_args()

    get_answer_fn = _get_model_function(args.model)
    run_evaluation(args.model, get_answer_fn, args.sample)


if __name__ == "__main__":
    main()
