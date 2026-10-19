#!/usr/bin/env python3
"""Offline scoring of a multi-sample run from its saved raw replies (no model calls).

    python3 src/eval_runs.py outputs/final10_test.json [--set visible|dev|test]

Scores every single sample as a 1x run, and the per-line majority vote over the first
k samples (k = 2..n) as a kx run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import parse_header          # noqa: E402
from model_extract import parse_sample, split_item, vote  # noqa: E402
from score import score                   # noqa: E402
from solve import solve_all               # noqa: E402


def run_score(items: list[dict], key: dict, raws: dict[str, list[str]],
              pick: list[int], drop: tuple[str, ...] = (),
              minimal: bool = True, threshold: float = 0.2,
              salvage: bool = True) -> dict:
    entries = []
    for it in items:
        entry = {"id": it["id"], **parse_header(it["text"])}
        _, body = split_item(it["text"])
        samples = [parse_sample(raws[it["id"]][j], entry, body, salvage) for j in pick
                   if j < len(raws[it["id"]])]       # focused runs may stop early
        chosen = vote(samples, body, threshold)
        facts = [{"line": ln, **f, "text": t} for ln, t in body
                 if (f := chosen.get(ln)) is not None]
        entries.append({**entry, "facts": facts})
    answers, _ = solve_all(entries, drop, minimal)
    return score(key, answers, items)


def load_set(name: str) -> tuple[list[dict], dict]:
    """visible (the given items), dev or test (re-worded proxies from paraphrase.py)."""
    root = Path(__file__).resolve().parent.parent
    if name == "visible":
        return (json.loads((root / "data/items.json").read_text()),
                json.loads((root / "data/visible_key.json").read_text()))
    d = root / "data" / f"proxy_{name}"
    return json.loads((d / "items.json").read_text()), json.loads((d / "key.json").read_text())


def set_arg(default: str = "visible") -> str:
    return sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else default


def main() -> None:
    base = Path(sys.argv[1])
    items, key = load_set(set_arg())
    raws = json.loads(Path(str(base) + ".raw.json").read_text())
    n = max(len(v) for v in raws.values())

    def show(label: str, res: dict) -> None:
        r = res["per_case_rate"]
        print(f"{label:28s} macro={res['macro_exact_match']:.3f}  "
              f"unique={r['unique']:.2f} ambiguous={r['ambiguous']:.2f} "
              f"inconsistent={r['inconsistent']:.2f}")

    for j in range(n):
        show(f"1x (sample {j + 1})", run_score(items, key, raws, [j]))
    for k in range(2, n + 1):
        show(f"{k}x vote (samples 1..{k})", run_score(items, key, raws, list(range(k))))


if __name__ == "__main__":
    main()
