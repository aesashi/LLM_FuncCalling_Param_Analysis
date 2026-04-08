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
    + ["redundant_information_total", "redundant_information_any_redundant_rate"]
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
    """Read a score file: parse the aggregate header and compute any_redundant_rate from per-sample lines."""
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return None
    data = json.loads(lines[0].strip())
    # Only aggregate rows have accuracy at the top level
    if "accuracy" not in data:
        return None

    # Compute binary rate from per-sample lines (lines[1:])
    redundant_counts = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        pls = sample.get("param_level_stats")
        if pls is None:
            continue
        ri = pls.get("redundant_information")
        if ri is not None:
            redundant_counts.append(ri.get("count", 0))

    if redundant_counts:
        data["any_redundant_rate"] = round(
            sum(1 for c in redundant_counts if c > 0) / len(redundant_counts), 4
        )
    else:
        data["any_redundant_rate"] = None

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
        row["redundant_information_any_redundant_rate"] = data.get("any_redundant_rate")
        rows.append(row)

    rows.sort(key=lambda r: r["model"])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
