#!/usr/bin/env python3
"""Ablation table from saved multi-sample runs. Makes no model calls.

    python3 src/ablate.py outputs/local_3x.json > docs/ablations.md

1x rows are the mean over each saved sample scored as its own 1x run (the spread is
shown); 3x rows are the per-line majority vote over the three samples.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_runs import run_score  # noqa: E402
from extract import extract_item  # noqa: E402
from score import score  # noqa: E402
from solve import solve_all  # noqa: E402

TYPES = ["block", "not_block", "station", "not_station", "before",
         "immediately_before", "between", "station_before"]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    items = json.loads((root / "data/items.json").read_text())
    key = json.loads((root / "data/visible_key.json").read_text())
    raws = json.loads(Path(sys.argv[1] + ".raw.json").read_text())
    n = min(len(v) for v in raws.values())

    def row(label: str, drop: tuple[str, ...] = (), minimal: bool = True) -> str:
        singles = [run_score(items, key, raws, [j], drop, minimal)["macro_exact_match"]
                   for j in range(n)]
        vote = run_score(items, key, raws, list(range(n)), drop, minimal)
        return (f"| {label} | {st.mean(singles):.1%} ({min(singles):.1%}-{max(singles):.1%}) "
                f"| {vote['macro_exact_match']:.1%} |")

    entries = [extract_item(it) for it in items]
    rules, _ = solve_all(entries)
    rules_score = score(key, rules, items)["macro_exact_match"]

    print("| Configuration | 1x: mean (range of 3 runs) | 3x: majority vote |")
    print("|---|---|---|")
    print(row("Full system (model labels + solver + minimal-conflict citations)"))
    print(row("- cite every constraint instead of a minimal conflict", minimal=False))
    for t in TYPES:
        print(row(f"- ignore `{t}` labels", drop=(t,)))
    print(f"| Rules-only extractor + solver (no model; not a valid submission) "
          f"| {rules_score:.1%} | {rules_score:.1%} |")
    print("| One-shot Granite, no harness (from the brief) | 0.0% | n/a |")


if __name__ == "__main__":
    main()
