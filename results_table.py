"""Aggregate per-model scored CSVs into a single results table.

Reads every '*_scored.csv' file in outputs/, computes the right metric
for each (BLEU + ROUGE-L for IndicQA, accuracy for HiMed MCQ datasets),
and produces a comparison table for paper / analysis.

Usage:
    python results_table.py
"""

import argparse
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).parent / "outputs"

# Per-dataset metric specs. Each entry says: "for this dataset, the
# scored CSV has these column(s), and we want them displayed under
# these names in the final table."
#
# Format: dataset_name -> [(display_name, csv_column_name), ...]
#
# This is a dispatch table (same pattern as DATASETS in evaluate.py).
# Adding a new dataset = one line here. No if/elif anywhere.
_METRIC_SPECS = {
    "indicqa":    [("BLEU", "bleu"), ("ROUGE-L", "rouge_l")],
    "himed_trad": [("Accuracy", "is_correct")],
    "himed_west": [("Accuracy", "is_correct")],
}

# Combos we EXPECT to eventually have. Used to detect missing files and
# warn (rather than silently producing a partial table).
_EXPECTED_MODELS = ["gemini", "airavata", "apollo"]
_EXPECTED_DATASETS = list(_METRIC_SPECS.keys())


def parse_scored_filename(filename: str) -> tuple[str, str]:
    """Extract (model, dataset) from a scored CSV filename.

    Filenames follow the pattern '{model}_{dataset}_scored.csv'.
    The dataset name itself may contain underscores (e.g. 'himed_trad'),
    so we strip the suffix first and split on the FIRST underscore only.

    Args:
        filename: Just the filename, like 'gemini_himed_trad_scored.csv'.
                  Path components, if any, are ignored — pass the basename.

    Returns:
        (model_name, dataset_name) tuple.

    Examples:
        >>> parse_scored_filename("gemini_indicqa_scored.csv")
        ('gemini', 'indicqa')
        >>> parse_scored_filename("airavata_himed_trad_scored.csv")
        ('airavata', 'himed_trad')
    """
    # Step 1: drop the '_scored.csv' suffix → 'gemini_himed_trad'
    stem = filename.removesuffix("_scored.csv")

    # Step 2: split on the FIRST underscore only. maxsplit=1 means
    # "make at most 1 split" → produces a 2-element list regardless of
    # how many underscores remain in the dataset name.
    model, dataset = stem.split("_", maxsplit=1)
    return model, dataset


def compute_metrics_for_file(file_path: Path, dataset: str) -> dict[str, float]:
    """Read one scored CSV and compute its aggregate metric(s).

    Picks the right metric(s) based on the dataset:
        - indicqa → BLEU + ROUGE-L (means of those columns)
        - himed_* → Accuracy (mean of is_correct column)

    Args:
        file_path: Path to a scored CSV (output of score.py or score_mcq.py).
        dataset: Dataset name (used to look up which columns to aggregate).

    Returns:
        Dict mapping metric display name to mean value.
        E.g. {"BLEU": 0.42, "ROUGE-L": 0.51}.

    Raises:
        ValueError: If 'dataset' is not in _METRIC_SPECS.
    """
    if dataset not in _METRIC_SPECS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Valid: {sorted(_METRIC_SPECS.keys())}"
        )

    # pd.read_csv() reads the CSV file into a DataFrame.
    # pandas auto-infers column types — strings stay strings, numbers
    # become numbers (so '0.42' becomes a float, '1' becomes an int).
    df = pd.read_csv(file_path)

    # Build the result dict by walking the spec for this dataset.
    # df[col] gets that one column as a Series.
    # .mean() computes the average of all values in the Series.
    # float(...) converts numpy's float64 (pandas's internal type) to a
    # plain Python float — cleaner for downstream code and easier to
    # compare in tests.
    return {
        display_name: float(df[col].mean())
        for display_name, col in _METRIC_SPECS[dataset]
    }


def build_results_table(outputs_dir: Path) -> pd.DataFrame:
    """Aggregate every '*_scored.csv' in outputs_dir into one results DataFrame.

    Walks the expected (model, dataset) combos:
    - If the file exists → compute its metrics and add rows to the table.
    - If the file is missing → print a warning and continue.

    Args:
        outputs_dir: Directory containing the scored CSVs.

    Returns:
        Long-format DataFrame with columns: model, dataset, metric, value.
    """
    outputs_dir = Path(outputs_dir)
    rows = []
    missing = []

    # Walk every expected combination of model and dataset.
    for model in _EXPECTED_MODELS:
        for dataset in _EXPECTED_DATASETS:
            path = outputs_dir / f"{model}_{dataset}_scored.csv"

            if not path.exists():
                missing.append(f"{model}_{dataset}")
                continue

            metrics = compute_metrics_for_file(path, dataset=dataset)
            for metric_name, value in metrics.items():
                rows.append({
                    "model":   model,
                    "dataset": dataset,
                    "metric":  metric_name,
                    "value":   value,
                })

    # Print warnings for any expected files that were missing.
    if missing:
        print(f"WARNING: {len(missing)} expected scored file(s) missing:")
        for combo in missing:
            print(f"  - {combo}_scored.csv")

    return pd.DataFrame(rows)


def main():
    """Build the results table from outputs/ and print + save it."""
    parser = argparse.ArgumentParser(
        description="Aggregate scored CSVs into a results comparison table."
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=OUTPUTS_DIR,
        help="Directory containing scored CSVs (default: ./outputs/)",
    )
    parser.add_argument(
        "--save-as",
        type=Path,
        default=None,
        help="Where to save the table (default: <outputs-dir>/results_summary.csv)",
    )
    args = parser.parse_args()

    df = build_results_table(args.outputs_dir)

    if df.empty:
        print("No scored CSVs found. Run evaluate.py + score.py first.")
        return

    # Print: long format (one row per metric — easy to scan)
    print("\nResults table (long format):")
    print(df.to_string(index=False))

    # Print: wide format (pivoted — better for paper)
    wide = df.pivot_table(
        index=["model", "dataset"], columns="metric", values="value"
    )
    print("\nResults table (wide format):")
    print(wide.to_string())

    # Save the long-format table to CSV
    save_path = args.save_as or (args.outputs_dir / "results_summary.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved long-format table to: {save_path}")


if __name__ == "__main__":
    main()
