"""Compare standard (English-summary) loops with translated-summary loops.

Usage:
    python scripts/analyze_multilingual.py results/*_results.json \
        --data_package ../assignment4-data/loops --output_dir analysis

Writes to output_dir:
    analysis.md         every number used in the report (per run, paired, language checks)
    failed_loops.md     side-by-side view of each loop that failed after cycle 1, for manual inspection
    failed_loops.csv    one row per failed loop, with empty columns for manual labels

A "loop failure" is a failure at cycle >= 2: the task's code passed once, and the loop broke
when the model re-implemented its own summary. Cycle-1 failures use the original MBPP+ prompt,
which is identical in both configurations, so they say nothing about the summary language.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

SPANISH_WORDS = {"que", "de", "la", "el", "los", "las", "una", "un", "y", "en", "para", "con", "por",
                 "función", "devuelve", "lista", "si", "del", "se", "cada", "número", "cadena", "utilizando"}
ENGLISH_WORDS = {"the", "and", "of", "that", "a", "to", "function", "returns", "which", "with", "for",
                 "list", "if", "each", "number", "string", "given", "in", "is"}


def load_runs(paths: List[str]) -> Dict[tuple, Dict[str, Any]]:
    runs = {}
    for path in paths:
        for result in json.loads(Path(path).read_text()):
            model = (result.get("model_id") or result["model"]).split("/")[-1]
            runs[(model, result.get("summary_language") or "english")] = result
    return runs


def survived(result: Dict[str, Any]) -> Dict[str, int]:
    return {p["task_id"]: p["successful_cycles"] for p in result["prompt_results"]}


def asl_pow(cycles: List[int], max_cycles: int = 10) -> float:
    return sum(c * c for c in cycles) / (len(cycles) * max_cycles) if cycles else 0.0


# The opening phrase is dictated by the prompt, so it is removed before guessing the language
STARTER = re.compile(r"^\s*(escribe una funci[oó]n de python que|write a python function( that| to)?)\s*", re.I)


def summary_language(text: str) -> str:
    """Majority vote over common function words; good enough to flag non-compliant summaries."""
    words = re.findall(r"[a-záéíóúñü]+", STARTER.sub("", text).lower())
    spanish = sum(w in SPANISH_WORDS for w in words)
    english = sum(w in ENGLISH_WORDS for w in words)
    if spanish == english:
        return "unclear"
    return "spanish" if spanish > english else "english"


def sentence_count(text: str) -> int:
    return max(1, len(re.findall(r"[.!?](?:\s+|$)", text.strip())))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pass_fraction(test_results: Dict[str, Any]) -> Optional[float]:
    """Share of base + plus test inputs the failing code still passed."""
    marks = list(test_results.get("base_details") or []) + [int(c) for c in str(test_results.get("plus_details") or "")]
    return round(sum(marks) / len(marks), 2) if marks else None


def run_section(name: str, result: Dict[str, Any]) -> List[str]:
    cycles = list(survived(result).values())
    n = len(cycles)
    passed_first = [c for c in cycles if c >= 1]
    dist = Counter(cycles)
    loop_failures = [c for c in cycles if 1 <= c < 10]
    summaries = [c.get("task_2_raw_output", "") for p in result["prompt_results"] for c in p["cycles"]]
    langs = Counter(summary_language(s) for s in summaries)
    multi = sum(sentence_count(s) > 1 for s in summaries)
    words = [len(s.split()) for s in summaries]
    return [
        f"### {name}",
        f"- tasks {n}; pass@1 {len(passed_first) / n:.3f}; ASL_pow {asl_pow(cycles):.3f}; "
        f"mean cycles survived {sum(cycles) / n:.2f}",
        f"- cycles survived (count of tasks): {dict(sorted(dist.items()))}",
        f"- of {len(passed_first)} tasks passing cycle 1: {dist[10]} reached cycle 10 "
        f"({dist[10] / len(passed_first):.1%}); {len(loop_failures)} failed later (loop failures)",
        f"- ASL_pow over tasks passing cycle 1 only: {asl_pow(passed_first):.3f}",
        f"- summaries: {len(summaries)}; detected language {dict(langs)}; "
        f"multi-sentence {multi} ({multi / len(summaries):.1%}); mean words {sum(words) / len(words):.1f}",
        "",
    ]


def paired_section(model: str, english: Dict[str, Any], other: Dict[str, Any], language: str) -> List[str]:
    en, ot = survived(english), survived(other)
    tasks = sorted(set(en) & set(ot))
    both = [t for t in tasks if en[t] >= 1 and ot[t] >= 1]
    wins = sum(en[t] > ot[t] for t in both)
    losses = sum(en[t] < ot[t] for t in both)

    first = {t: (english_cycle(english, t), english_cycle(other, t)) for t in tasks}
    same_code = sum(a is not None and b is not None and a["task_1_output"] == b["task_1_output"]
                    for a, b in first.values())
    same_pass = sum(a is not None and b is not None and a["success"] == b["success"] for a, b in first.values())
    return [
        f"### {model}: english vs {language}",
        f"- cycle 1 (identical prompt in both runs): same code {same_code}/{len(tasks)}, "
        f"same pass/fail {same_pass}/{len(tasks)}  -> run-to-run noise at temperature 0",
        f"- tasks passing cycle 1 in both runs: {len(both)}; "
        f"ASL_pow english {asl_pow([en[t] for t in both]):.3f} vs {language} {asl_pow([ot[t] for t in both]):.3f}",
        f"- english survived longer {wins}, {language} survived longer {losses}, tied {len(both) - wins - losses}; "
        f"sign test p = {sign_test(wins, losses):.3f}",
        "",
    ]


def english_cycle(result: Dict[str, Any], task_id: str, cycle: int = 1) -> Optional[Dict[str, Any]]:
    for p in result["prompt_results"]:
        if p["task_id"] == task_id:
            return next((c for c in p["cycles"] if c["cycle"] == cycle), None)
    return None


def data_package_section(model: str, english: Dict[str, Any], loops_dir: Path) -> List[str]:
    path = loops_dir / f"{model}.json"
    if not path.exists():
        return []
    package = {t["task_id"]: t["successful_cycles"] for t in json.loads(path.read_text())["tasks"]}
    ours = survived(english)
    tasks = sorted(set(package) & set(ours))
    agree_first = sum((package[t] >= 1) == (ours[t] >= 1) for t in tasks)
    agree_exact = sum(package[t] == ours[t] for t in tasks)
    return [
        f"### {model}: our standard run vs data package",
        f"- data package: pass@1 {sum(package[t] >= 1 for t in tasks) / len(tasks):.3f}, "
        f"ASL_pow {asl_pow([package[t] for t in tasks]):.3f}; "
        f"ours: pass@1 {sum(ours[t] >= 1 for t in tasks) / len(tasks):.3f}, ASL_pow {asl_pow([ours[t] for t in tasks]):.3f}",
        f"- same cycle-1 pass/fail on {agree_first}/{len(tasks)} tasks; same cycles survived on {agree_exact}/{len(tasks)}",
        "",
    ]


def failed_loops(model: str, language: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for p in result["prompt_results"]:
        failed_at = p["successful_cycles"] + 1
        if not 2 <= failed_at <= result.get("max_cycles", 10):
            continue
        cycles = {c["cycle"]: c for c in p["cycles"]}
        before, failing = cycles[failed_at - 1], cycles[failed_at]
        tests = failing["test_results"]
        renamed = renamed_functions(p["initial_prompt"], before["task_1_output"], failing["task_1_output"])
        rows.append({
            "model": model, "language": language, "task_id": p["task_id"], "failed_cycle": failed_at,
            "base_passed": tests.get("base_success"), "plus_passed": tests.get("plus_success"),
            "test_pass_fraction": pass_fraction(tests),
            "tests_ran": bool(tests.get("base_details") or tests.get("plus_details")),
            "renamed_function": ", ".join(renamed),
            "original_prompt": p["initial_prompt"].strip(),
            "last_passing_code": before["task_1_output"],
            "summary_given": before["task_2_output"].strip(),
            "failing_code": failing["task_1_output"],
            "failing_raw_output": failing.get("task_1_raw_output", ""),
        })
    return rows


def renamed_functions(prompt: str, passing_code: str, failing_code: str) -> List[str]:
    """Functions the assert line calls that the passing code defined but the failing code no longer does."""
    assert_line = next((line for line in prompt.splitlines() if line.strip().startswith("assert")), "")
    called = set(re.findall(r"(\w+)\(", assert_line))
    return sorted(name for name in called
                  if re.search(rf"def {name}\(", passing_code) and not re.search(rf"def {name}\(", failing_code))


def describe_failure(row: Dict[str, Any]) -> str:
    if row["renamed_function"]:
        return f"required function {row['renamed_function']} no longer defined, so no test ran"
    if not row["tests_ran"]:
        return "no test result (crash or timeout)"
    tests = "base and plus" if not row["base_passed"] else "plus only"
    return f"{tests} tests failed; {row['test_pass_fraction']} of test inputs still passed"


def failure_breakdown(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["## Loop failures (cycle >= 2) by mechanical type", "",
             "| model | language | loop failures | renamed required function | no test result | "
             "base tests failed | plus tests only | failed at cycle 2 |",
             "|---|---|---|---|---|---|---|---|"]
    for key in sorted({(r["model"], r["language"]) for r in rows}):
        group = [r for r in rows if (r["model"], r["language"]) == key]
        renamed = sum(bool(r["renamed_function"]) for r in group)
        no_result = sum(not r["tests_ran"] and not r["renamed_function"] for r in group)
        ran = [r for r in group if r["tests_ran"]]
        base = sum(not r["base_passed"] for r in ran)
        lines.append(f"| {key[0]} | {key[1]} | {len(group)} | {renamed} | {no_result} | {base} | "
                     f"{len(ran) - base} | {sum(r['failed_cycle'] == 2 for r in group)} |")
    return lines + [""]


def write_failed_loops(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    lines = ["# Loops that failed after cycle 1", "",
             "Each entry: the summary the model wrote of its last passing code, then the code it wrote "
             "from that summary, which failed the tests.", ""]
    for i, r in enumerate(rows, 1):
        lines += [
            f"## {i}. {r['model']} / {r['language']} / {r['task_id']}: failed at cycle {r['failed_cycle']} "
            f"({describe_failure(r)})",
            "", "**Original task**", "```", r["original_prompt"], "```",
            "**Last passing code**", "```python", r["last_passing_code"], "```",
            "**Summary given as the next task**", "```", r["summary_given"], "```",
            "**Failing code**", "```python", r["failing_code"], "```", "",
        ]
    (output_dir / "failed_loops.md").write_text("\n".join(lines))

    fields = ["id", "model", "language", "task_id", "failed_cycle", "base_passed", "plus_passed",
              "test_pass_fraction", "tests_ran", "renamed_function", "failure_mode", "notes"]
    with open(output_dir / "failed_loops.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, r in enumerate(rows, 1):
            writer.writerow({"id": i, **r, "failure_mode": "", "notes": ""})


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="EvaLoop *_results.json files")
    parser.add_argument("--data_package", help="assignment4-data/loops directory, for the sanity check")
    parser.add_argument("--output_dir", default="analysis")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results)
    models = sorted({model for model, _ in runs})

    lines = ["# Multilingual summary loop: analysis", "", "## Per run", ""]
    for (model, language), result in sorted(runs.items()):
        lines += run_section(f"{model} / {language}", result)

    lines += ["## Paired comparison (same model, same tasks)", ""]
    rows = []
    for model in models:
        english = runs.get((model, "english"))
        for (m, language), result in sorted(runs.items()):
            if m == model and language != "english" and english:
                lines += paired_section(model, english, result, language)
        for (m, language), result in sorted(runs.items()):
            if m == model:
                rows += failed_loops(model, language, result)

    lines += failure_breakdown(rows)

    if args.data_package:
        lines += ["## Sanity check against the data package", ""]
        for model in models:
            if (model, "english") in runs:
                lines += data_package_section(model, runs[(model, "english")], Path(args.data_package))

    (output_dir / "analysis.md").write_text("\n".join(lines))
    write_failed_loops(rows, output_dir)
    print("\n".join(lines))
    print(f"{len(rows)} failed loops written to {output_dir / 'failed_loops.md'}")


if __name__ == "__main__":
    main()
