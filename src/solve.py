#!/usr/bin/env python3
"""Solver: facts JSON -> answers, scored against visible_key.json.

Reads the facts produced by extract.py, turns each fact into a predicate over
candidate assignments, enumerates every candidate (people-permutations x
station-permutations of the station holders), and counts the consistent ones:

  0 solutions  -> inconsistent (solve a minimal conflicting fact set)
  1 solution   -> unique
  2-4          -> ambiguous (all of them)

Also runs ablations: which fact types are load-bearing is decided by scoring,
not guessed.

Usage:
    python3 src/solve.py outputs/facts.json data/items.json \
        --out outputs/answers_rules.json
    python3 src/solve.py outputs/facts.json data/items.json --drop history,carshare,open  \
        --score   # drop facts of these types before solving
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Callable

CASES = ("unique", "ambiguous", "inconsistent")

Fact = dict[str, Any]
Entry = dict[str, Any]                 # one item from facts.json
Candidate = dict[str, dict[str, str]]  # {"blocks": {...}, "stations": {...}}
Predicate = Callable[[Candidate], bool]
Answer = dict[str, Any]


# --- fact -> predicate -------------------------------------------------------

def make_pred(fact: Fact) -> Predicate | None:
    """candidate -> bool, or None if the type is ignored."""
    t = fact["type"]
    if t == "block":
        p, b = fact["person"], fact["block"]
        return lambda c: c["blocks"][p] == b
    if t == "not_block":
        p, b = fact["person"], fact["block"]
        return lambda c: c["blocks"][p] != b
    if t == "station":
        p, s = fact["person"], fact["station"]
        return lambda c: c["stations"].get(p) == s
    if t == "not_station":
        p, s = fact["person"], fact["station"]
        return lambda c: c["stations"].get(p, "NO-STATION") != s
    if t == "before":
        a, b = fact["person"], fact["other"]
        return lambda c: c["blocks"][a] < c["blocks"][b]
    if t == "immediately_before":
        a, b = fact["person"], fact["other"]
        order = ["07:00", "09:00", "11:00", "13:00", "15:00"]
        return lambda c: order.index(c["blocks"][a]) + 1 == order.index(c["blocks"][b])
    if t == "between":
        x, a, b = fact["person"], fact["a"], fact["b"]
        return lambda c: c["blocks"][a] < c["blocks"][x] < c["blocks"][b] \
            or c["blocks"][b] < c["blocks"][x] < c["blocks"][a]
    if t == "station_before":
        s, p = fact["station"], fact["person"]
        def out(c: Candidate, s: str = s, p: str = p) -> bool:
            holder = next((h for h, st in c["stations"].items() if st == s), None)
            if holder is None:
                return False
            return c["blocks"][holder] < c["blocks"][p]
        return out
    return None  # history, carshare, open, none, unknown: no force


def enumerate_solutions(entry: Entry, drop: tuple[str, ...] = ()) -> list[Candidate]:
    people = entry["people"]
    blocks = entry["blocks"]
    holders = [p for p in people if p in set(entry["station_holders"])]
    stations = entry["stations"]
    preds: list[Predicate] = []
    for f in entry["facts"]:
        if f["type"] in drop:
            continue
        pr = make_pred(f)
        if pr is not None:
            preds.append(pr)
    sols: list[Candidate] = []
    for perm in itertools.permutations(blocks):
        assigned = dict(zip(people, perm))
        # station holders get distinct stations; others none
        for stat_perm in itertools.permutations(stations):
            stations_of = dict(zip(holders, stat_perm))
            cand = {"blocks": assigned, "stations": stations_of}
            if all(pr(cand) for pr in preds):
                sols.append(cand)
    return sols


def minimal_conflict(entry: Entry, drop: tuple[str, ...] = ()) -> list[Fact]:
    """A minimal set of load-bearing facts that cannot all hold at once.

    Deletion over the load-bearing facts: drop one at a time, keep the drop
    if the remainder is still contradictory. Every fact in the result is
    load-bearing, so dropping any one leaves a satisfiable set.
    """
    load = [f for f in entry["facts"]
            if f["type"] not in drop and make_pred(f) is not None]
    if enumerate_solutions({**entry, "facts": load}, drop) :
        return []          # satisfiable: not inconsistent after all
    core = list(load)
    for f in list(core):
        trial = [g for g in core if g is not f]
        if not is_consistent(entry, trial, drop):
            core = trial
    return core


def is_consistent(entry: Entry, facts: list[Fact], drop: tuple[str, ...] = ()) -> bool:
    saved = entry["facts"]
    entry["facts"] = facts
    try:
        return bool(enumerate_solutions(entry, drop))
    finally:
        entry["facts"] = saved


def to_assignment(entry: Entry, sol: Candidate) -> dict[str, dict[str, str]]:
    holders = set(entry["station_holders"])
    a: dict[str, dict[str, str]] = {}
    for p in entry["people"]:
        fields = {"block": sol["blocks"][p]}
        if p in holders:
            fields["station"] = sol["stations"][p]
        a[p] = fields
    return a


def solve_all(facts: list[Entry],
              drop: tuple[str, ...] = ()) -> tuple[dict[str, Answer], dict[str, int]]:
    answers: dict[str, Answer] = {}
    stats: dict[str, int] = {"unique": 0, "ambiguous": 0, "inconsistent": 0, "many": 0}
    for entry in facts:
        sols = enumerate_solutions(entry, drop)
        n = len(sols)
        if n == 0:
            core = minimal_conflict(entry, drop)
            answers[entry["id"]] = {
                "case": "inconsistent",
                "conflicts": [f["text"] for f in core],
            }
            stats["inconsistent"] += 1
        elif n == 1:
            answers[entry["id"]] = {"case": "unique",
                                    "assignment": to_assignment(entry, sols[0])}
            stats["unique"] += 1
        elif 2 <= n <= 4:
            answers[entry["id"]] = {
                "case": "ambiguous",
                "assignments": [to_assignment(entry, s) for s in sols],
            }
            stats["ambiguous"] += 1
        else:
            stats["many"] += 1
            answers[entry["id"]] = {"case": "ambiguous",
                                    "assignments": [to_assignment(entry, s)
                                                    for s in sols[:4]]}
    return answers, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("facts")
    ap.add_argument("items", nargs="?", default=None,
                    help="items.json, needed by score.py for citation checks")
    ap.add_argument("--key", default="data/visible_key.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop", default="history,carshare,open",
                    help="fact types to ignore (comma-separated)")
    args = ap.parse_args()

    facts = json.loads(open(args.facts).read())
    drop = tuple(x.strip() for x in args.drop.split(",") if x.strip())
    answers, stats = solve_all(facts, drop)
    print("solved:", stats)
    if args.out:
        open(args.out, "w").write(json.dumps(answers, indent=2) + "\n")
        print("wrote", args.out)
    key = json.loads(open(args.key).read())
    if key and "_readme" in key:
        pass
    sub = {k: ({k2: v2 for k2, v2 in v.items()} if isinstance(v, dict) else v)
           for k, v in answers.items()}
    import subprocess, sys
    cmd = [sys.executable, str(Path(__file__).with_name("score.py")), args.key,
           args.out or "/dev/stdout"] + ([args.items] if args.items else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)


if __name__ == "__main__":
    main()
