#!/usr/bin/env python3
"""Held-out proxies: re-word every line of every item, keeping each item's truth.

    python3 src/paraphrase.py      # writes data/proxy_dev/ and data/proxy_test/

Each note line's meaning comes from the rules extractor (constraint, failed request,
hypothetical, past arrangement, open question, car-share or filler) and is re-rendered
from a phrasing pool that appears neither in the visible notes nor in the model prompt.
The key's conflict statements are mapped to the new wording.

Two disjoint pools:
  dev  -- used while developing the prompt.
  test -- written before any tuning and only scored at the end. Do not tune on it.
The header sentence is left unchanged in both.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from extract import extract_item, parse_header

POOLS = {
    "dev": {
        "hedge": ["", "", "", "Apparently, ", "To the best of my knowledge, ", "Word is that ",
                  "Per the whiteboard, "],
        "repeat": ["Again, for the record: ", "Same point as before: "],
        "block": ["{person} has the {block} slot.", "{block} is when {person} works.",
                  "{person} will be covering {block}.", "The {block} slot belongs to {person}."],
        "not_block": ["{person} can't do {block}.", "{person} is off at {block}.",
                      "The {block} slot isn't {person}'s.", "{person} won't be in for {block}."],
        "station": ["{person} is the one running {station}.", "{station} duty goes to {person}.",
                    "{person} handles {station}.", "{person} has been given {station}."],
        "not_station": ["{person} is kept off {station}.", "{station} is not {person}'s job.",
                        "{person} isn't covering {station}.", "{person} has no part in {station}."],
        "before": ["{person} finishes up ahead of {other}.", "{person} goes on before {other} does.",
                   "{other} follows {person} at some point.",
                   "{person} starts earlier in the day than {other}."],
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
        "req_not_block": ["{person} asked for {block} but it was refused.",
                          "{person} hoped to get {block}; no luck."],
        "cf_not_block": ["{person} on {block} would have suited everyone better.",
                         "In hindsight {person} should have had {block}."],
        "history": ["{person} covered {station} last quarter.", "Years ago {person} ran {station}."],
        "open": ["Whether {person} has {block} is still undecided in the notes.",
                 "{person} and {block}: nobody wrote anything down."],
        "carshare": ["{p0} gives {p1} a lift when the timing works."],
        "none": ["{person} restocked the gloves.", "The kettle in the break room is broken again.",
                 "{person} reported a flickering light in the corridor.",
                 "A fire drill is planned for next week."],
    },
    "test": {
        "hedge": ["", "", "", "Honestly, ", "As I understand it, ", "From the sign-up sheet, ",
                  "Someone mentioned that ", "Pretty sure "],
        "repeat": ["Repeating for clarity: ", "To say it again, ", "Flagging again: "],
        "block": ["{person}'s shift is the {block} one.", "Put {person} down for {block}.",
                  "The {block} block is being worked by {person}.",
                  "{person} is working the {block} block."],
        "not_block": ["{person} is definitely not working {block}.", "Don't expect {person} at {block}.",
                      "The {block} block is out for {person}.",
                      "Whatever {person} does, it isn't the {block} block."],
        "station": ["{person} runs the {station} desk.", "It's {person} who looks after {station}.",
                    "{person} is in charge of {station} today.",
                    "The {station} position is filled by {person}."],
        "not_station": ["{person} won't be anywhere near {station}.",
                        "Someone other than {person} does {station}.",
                        "The {station} job isn't going to {person}.",
                        "{person} is not the {station} person."],
        "before": ["{other} doesn't start until after {person} has been on.",
                   "{person}'s slot comes earlier than {other}'s.",
                   "{other} is later on the board than {person}.",
                   "Earlier in the day you'll find {person}; {other} comes after."],
        "immediately_before": ["{other} picks up right where {person} leaves off.",
                               "The block right after {person}'s is {other}'s.",
                               "{person} is immediately followed on the board by {other}.",
                               "No gap: {person}, then straight away {other}."],
        "between": ["{person} is somewhere in between {a} and {b}, timewise.",
                    "One of {a} and {b} is before {person} and the other is after.",
                    "{a} and {b} are on either side of {person}, in some order.",
                    "Timewise, {person} falls inside the stretch between {a} and {b}."],
        "station_before": ["Whoever's doing {station} is on earlier than {person}.",
                           "The {station} shift starts before {person}'s.",
                           "{person} works after the person covering {station}.",
                           "The {station} desk is staffed at an earlier block than {person}'s."],
        "req_not_block": ["{person} applied to switch to {block} and the answer was no.",
                          "{person}'s request for {block} went nowhere."],
        "cf_not_block": ["It would have been neater with {person} at {block}.",
                         "If we were starting over, {person} might get {block}."],
        "history": ["{person} used to look after {station} before the reshuffle.",
                    "Before the current rota, {station} was {person}'s."],
        "open": ["Nobody confirmed if {person} is on {block}.",
                 "There's a question mark next to {person} for {block}."],
        "carshare": ["{p0} and {p1} sometimes travel in together."],
        "none": ["{person} is organising the leaving card.", "The vending machine is out of order.",
                 "{person} mentioned the car park barrier is stuck.",
                 "The inventory count happens on Friday."],
    },
}

REPEAT_MARK = re.compile(r"^(worth restating|this came up more than once|noted twice|it bears repeating)",
                         re.I)


def fields_for(fact: dict, raw: str, people: list[str], rng: random.Random) -> dict:
    """Template fields; open and filler lines carry no fields in the extractor."""
    f = {k: v for k, v in fact.items() if k not in ("line", "type", "text")}
    if fact["type"] == "carshare":
        f["p0"], f["p1"] = fact["people"][:2]
    if fact["type"] == "open":
        f["person"] = next((p for p in people if p in raw), rng.choice(people))
        f["block"] = (re.search(r"\d{2}:\d{2}", raw) or [rng.choice(["07:00", "11:00"])])[0]
    if fact["type"] in ("none", "unknown"):
        f["person"] = rng.choice(people)
    return f


def render(pool: dict, kind: str, f: dict, raw: str, rng: random.Random) -> str:
    text = rng.choice(pool[kind]).format(**f)
    hedge = rng.choice(pool["hedge"])
    if hedge and text.split()[0] in ("The", "In", "Whoever", "Whoever's", "Whether", "There's",
                                     "It", "If", "Earlier", "Timewise,", "One", "No", "Someone",
                                     "Before", "Years", "Don't", "Whatever", "Put", "A"):
        text = text[0].lower() + text[1:]
    out = hedge + text
    if REPEAT_MARK.match(raw):
        out = rng.choice(pool["repeat"]) + out
    return out[0].upper() + out[1:]


def build(name: str, items: list[dict], key: dict, seed: int) -> None:
    pool, rng = POOLS[name], random.Random(seed)
    out_items, out_key = [], {}
    for it in items:
        people = parse_header(it["text"])["people"]
        facts = {f["line"]: f for f in extract_item(it)["facts"]}
        lines, mapping = it["text"].split("\n"), {}
        for i, raw in enumerate(lines):
            fact = facts.get(i)
            if not fact:
                continue
            kind = "none" if fact["type"] == "unknown" else fact["type"]
            lines[i] = render(pool, kind, fields_for(fact, raw, people, rng), raw, rng)
            mapping[raw.strip()] = lines[i]
        out_items.append({**it, "text": "\n".join(lines)})
        k = json.loads(json.dumps(key[it["id"]]))
        if "conflict_line_sets" in k:
            k["conflict_line_sets"] = [[mapping.get(s, s) for s in cs]
                                       for cs in k["conflict_line_sets"]]
        out_key[it["id"]] = k
    dest = Path(__file__).resolve().parent.parent / "data" / f"proxy_{name}"
    dest.mkdir(exist_ok=True)
    (dest / "items.json").write_text(json.dumps(out_items, indent=2) + "\n")
    (dest / "key.json").write_text(json.dumps(out_key, indent=2) + "\n")
    print(f"wrote {len(out_items)} items to {dest}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    items = json.loads((root / "data/items.json").read_text())
    key = json.loads((root / "data/visible_key.json").read_text())
    build("dev", items, key, seed=7)
    build("test", items, key, seed=11)


if __name__ == "__main__":
    main()
