"""
Aggregate param_level_stats from all model score JSON files into a single CSV.
Each row = one model × one test category.
"""

import csv
import json
from pathlib import Path

SCORE_DIR = Path(__file__).parent / "score"
OUTPUT_CSV = SCORE_DIR / "param_scores_aggregated.csv"

PARAM_METRICS = [
    "missing_information",
    "hallucinated_params",
    "specification_mismatch",
    "task_deviation",
    "redundant_information",
]

STAT_FIELDS = ["mean", "median", "std", "n"]

COLUMNS = (
    ["model", "split", "test_category", "accuracy", "correct_count", "total_count"]
    # Group by stat so all means are adjacent, then all medians, etc.
    # Makes it easy to compare tendency across error types at a glance.
    + [f"{m}_{s}" for s in STAT_FIELDS for m in PARAM_METRICS]
    + ["redundant_information_total"]
)


def extract_param_stats(param_level_stats: dict) -> dict:
    row = {}
    for metric in PARAM_METRICS:
        stats = param_level_stats.get(metric, {})
        for s in STAT_FIELDS:
            row[f"{metric}_{s}"] = stats.get(s)
        if metric == "redundant_information":
            row["redundant_information_total"] = stats.get("total_redundant_params")
    return row


def parse_score_file(path: Path) -> dict | None:
    """Read the first line (aggregate row) of a score file."""
    with open(path) as f:
        first_line = f.readline().strip()
    if not first_line:
        return None
    data = json.loads(first_line)
    # Only aggregate rows have accuracy at the top level
    if "accuracy" not in data:
        return None
    return data


def main():
    rows = []

    for score_file in sorted(SCORE_DIR.rglob("*_score.json")):
        # skip the top-level CSV-adjacent files (data_*.csv etc.)
        parts = score_file.relative_to(SCORE_DIR).parts
        if len(parts) < 3:
            continue  # not inside a model/split/ subdirectory

        model = parts[0]
        split = parts[1]  # "live" or "non_live"
        # derive category from filename: BFCL_v4_<category>_score.json
        stem = score_file.stem  # e.g. BFCL_v4_simple_python_score
        test_category = stem.removeprefix("BFCL_v4_").removesuffix("_score")

        data = parse_score_file(score_file)
        if data is None:
            continue

        row = {
            "model": model,
            "split": split,
            "test_category": test_category,
            "accuracy": data.get("accuracy"),
            "correct_count": data.get("correct_count"),
            "total_count": data.get("total_count"),
        }

        param_stats = data.get("param_level_stats") or {}
        row.update(extract_param_stats(param_stats))
        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
