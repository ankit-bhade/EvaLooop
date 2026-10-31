"""Write summary.csv (one row per run) from EvaLoop results files.

Usage:
    python scripts/summarize_runs.py results/*_results.json --output summary.csv

Columns: model, configuration, n_tasks, success_rate, asl_score, where
success_rate is pass@1 / T and asl_score is ASL_pow = sum_i n_i * i^2 / (T * M),
the judge-free ASL (the leaderboard's `aslScore`).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from evaloop.analysis.metrics import asl_pow, pass_at_k

FIELDS = ["model", "configuration", "n_tasks", "success_rate", "asl_score"]


def configuration_name(result: Dict[str, Any]) -> str:
    """Names the loop variant, e.g. "standard-mbpp-plus" or "spanish-summary-mbpp-plus"."""
    language = result.get("summary_language") or "english"
    name = "standard-mbpp-plus" if language == "english" else f"{language}-summary-mbpp-plus"
    if result.get("max_tasks"):
        name += f"-subset{result['max_tasks']}-seed{result.get('subset_seed', 0)}"
    return name


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    """Computes one summary.csv row for a single model result."""
    cycles = [r["successful_cycles"] for r in result["prompt_results"]]
    max_cycles = result.get("max_cycles", 10)
    model = result.get("model_id") or result["model"]
    return {
        "model": model.split("/")[-1],
        "configuration": configuration_name(result),
        "n_tasks": len(cycles),
        "success_rate": round(pass_at_k(cycles, 1) / len(cycles), 3),
        "asl_score": round(asl_pow(cycles, max_cycles), 3),
    }


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="EvaLoop *_results.json files")
    parser.add_argument("--output", default="summary.csv")
    args = parser.parse_args(argv)

    rows = []
    for path in args.results:
        for result in json.loads(Path(path).read_text()):
            rows.append(summarize(result))
    rows.sort(key=lambda row: (row["model"], row["configuration"] != "standard-mbpp-plus", row["configuration"]))

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(",".join(str(row[k]) for k in FIELDS))


if __name__ == "__main__":
    main()
