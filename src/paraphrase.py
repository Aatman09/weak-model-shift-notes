#!/usr/bin/env python3
"""Held-out proxy: re-render each item's constraint lines in a new phrasing pool.

    python3 src/paraphrase.py            # writes data/heldout_proxy/{items,key}.json

The truth of every item is unchanged: the constraint behind each line (taken from the
rules extractor) is re-worded with templates and hedges that appear neither in the
visible notes nor in the model prompt, and the key's conflict statements are mapped to
the new wording. Filler, wish and history lines are left as they were, so this tests
constraint phrasing only. It is a proxy for the private held-out set, not a copy of it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from extract import extract_item

HEDGES = ["", "", "Apparently, ", "If memory serves, ", "To the best of my knowledge, ",
          "I believe ", "Word is that ", "Per the whiteboard, "]

POOL = {
    "block": ["{person} has the {block} slot.", "{block} is when {person} works.",
              "{person} will be covering {block}.", "The {block} slot belongs to {person}."],
    "not_block": ["{person} can't do {block}.", "{person} is off at {block}.",
                  "The {block} slot isn't {person}'s.", "{person} won't be in for {block}."],
    "station": ["{person} is the one running {station}.", "{station} duty goes to {person}.",
                "{person} handles {station}.", "{person} has been given {station}."],
    "not_station": ["{person} is kept off {station}.", "{station} is not {person}'s job.",
                    "{person} isn't covering {station}.", "{person} has no part in {station}."],
    "before": ["{person} finishes up ahead of {other}.", "{person} goes on before {other} does.",
               "{other} follows {person} at some point.", "{person} starts earlier in the day than {other}."],
    "immediately_before": ["{other} takes the slot straight after {person}.",
                           "{person} is followed immediately by {other}.",
                           "{person} and {other} are consecutive, {person} first.",
                           "{other} steps in the moment {person} finishes."],
    "between": ["{person} is sandwiched between {a} and {b} in the schedule.",
                "{a} and {b} bracket {person}, in some order.",
                "In time order {person} lands somewhere in the middle of {a} and {b}.",
                "{person} is neither before both nor after both of {a} and {b}."],
    "station_before": ["The {station} holder starts ahead of {person}.",
                       "{person} comes on after the {station} person has already started.",
                       "Whoever runs {station} is on before {person}.",
                       "{person} is behind the {station} person in the order."],
}


def render(fact: dict, rng: random.Random) -> str:
    text = rng.choice(POOL[fact["type"]]).format(**fact)
    hedge = rng.choice(HEDGES)
    if hedge and text.split()[0] in ("The", "In", "Whoever"):   # never lowercase a name
        text = text[0].lower() + text[1:]
    if not hedge:
        text = text[0].upper() + text[1:]
    return hedge + text


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    items = json.loads((root / "data/items.json").read_text())
    key = json.loads((root / "data/visible_key.json").read_text())
    rng = random.Random(7)
    out_items, out_key = [], {}
    for it in items:
        facts = {f["line"]: f for f in extract_item(it)["facts"]}
        lines, mapping = it["text"].split("\n"), {}
        for i, raw in enumerate(lines):
            f = facts.get(i)
            if f and f["type"] in POOL:
                lines[i] = render(f, rng)
                mapping[raw.strip()] = lines[i]
        out_items.append({**it, "text": "\n".join(lines)})
        k = json.loads(json.dumps(key[it["id"]]))
        if "conflict_line_sets" in k:
            k["conflict_line_sets"] = [[mapping.get(s, s) for s in cs]
                                       for cs in k["conflict_line_sets"]]
        for field in ("injected", "injected_statement"):
            if field in k and k[field] in mapping:
                k[field] = mapping[k[field]]
        out_key[it["id"]] = k
    dest = root / "data/heldout_proxy"
    dest.mkdir(exist_ok=True)
    (dest / "items.json").write_text(json.dumps(out_items, indent=2) + "\n")
    (dest / "key.json").write_text(json.dumps(out_key, indent=2) + "\n")
    print(f"wrote {len(out_items)} items to {dest}")


if __name__ == "__main__":
    main()
