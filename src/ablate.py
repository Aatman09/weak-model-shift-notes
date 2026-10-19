#!/usr/bin/env python3
"""Ablation table from saved runs of the full system. Makes no model calls.

    python3 src/ablate.py --set test --runs t_f_r1,t_f_r2,t_f_r3,final10_test

Each run is outputs/<name>.json.raw.json from `./run` with focused re-asking. A run's first
call is exactly a 1x run and its first three calls exactly a 3x run, so every run gives a
1x and a 3x score; runs with ten calls also give a 10x score. Cells are mean (min-max) over
runs. Each arm re-scores the same saved replies with one component removed or changed.
Arms that need different model calls (plain voting, other prompts) are separate runs.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_runs import load_set, run_score, set_arg  # noqa: E402
from extract import extract_item  # noqa: E402
from score import score  # noqa: E402
from solve import solve_all  # noqa: E402

TYPES = ["block", "not_block", "station", "not_station", "before",
         "immediately_before", "between", "station_before"]


def cell(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.1%}"
    return f"{st.mean(vals):.1%} ({min(vals):.1%}-{max(vals):.1%})"


def main() -> None:
    items, key = load_set(set_arg())
    names = sys.argv[sys.argv.index("--runs") + 1].split(",")
    root = Path(__file__).resolve().parent.parent
    runs = [json.loads((root / "outputs" / f"{n}.json.raw.json").read_text()) for n in names]

    def row(label: str, **kw) -> str:
        def sc(raws: dict, k: int) -> float:
            return run_score(items, key, raws, list(range(k)), **kw)["macro_exact_match"]
        one = [sc(r, 1) for r in runs]
        three = [sc(r, 3) for r in runs]
        ten = [sc(r, 10) for r in runs if max(len(v) for v in r.values()) > 3]
        return f"| {label} | {cell(one)} | {cell(three)} | {cell(ten)} |"

    print(f"Set: {set_arg()}; runs: {', '.join(names)}\n")
    print("| Configuration | 1x | 3x | 10x |")
    print("|---|---|---|---|")
    print(row("Full system"))
    print(row("- parser fallback for free-form replies", salvage=False))
    print(row("- low vote threshold (strict majority instead)", threshold=0.5))
    print(row("- minimal-conflict citations (cite every constraint)", minimal=False))
    for t in TYPES:
        print(row(f"- ignore `{t}` labels", drop=(t,)))
    if set_arg() == "visible":
        rules, _ = solve_all([extract_item(it) for it in items])
        r = score(key, rules, items)["macro_exact_match"]
        print(f"| Rules-only extractor + solver (no model; not a valid submission) "
              f"| {r:.1%} | {r:.1%} | {r:.1%} |")


if __name__ == "__main__":
    main()
