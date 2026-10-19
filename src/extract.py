#!/usr/bin/env python3
"""Rule-based extraction layer: shift notes -> structured facts JSON.

Reads items.json and writes a facts file: one entry per item with the rota
header (people, blocks, stations, station-holders) and a list of facts. Every
body line yields exactly one fact, tagged with its line number and the raw
line text, so a later step can quote the line verbatim.

Fact types
----------
  block             {person, block}
  not_block         {person, block}
  station           {person, station}
  not_station       {person, station}
  before            {person, other}        person is earlier than other
  immediately_before{person, other}        person is the block right before other
  between           {person, a, b}         person is strictly between a and b
  station_before    {station, person}      whoever holds station is before person
  history           {person, station}      past arrangement -- probably no force
  carshare          {people}               relationship fluff -- probably no force
  open              {}                     "left open" / "nothing minuted" -- no force
  none              {}                     filler line, constrains nothing
  unknown           {}                     not matched: needs a new rule

Usage:
    python3 src/extract.py data/items.json --out outputs/facts.json
    python3 src/extract.py data/items.json --sample B2-000     # print one item, no write
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Hedges and restatement openers. Stripped in a loop: some lines stack them
# ("Noted twice in the handover: As far as I know, ..."). A hedge carries full
# force, so stripping it changes nothing about the statement itself.
PREFIXES = [
    "worth restating, since it came up twice in handover:",
    "this came up more than once, so:",
    "noted twice in the handover:",
    "it bears repeating:",
    "as far as i know,",
    "going off the roster,",
    "speaking from memory,",
    "my recollection is that",
    "i'm fairly sure",
    "i am fairly sure",
    "i'm reasonably confident",
    "i believe",
    "if memory serves,",
    "from what i recall,",
    "unless i have this wrong,",
]

# Filler lines and distractor statements: they mention a person but say nothing
# about a block or a station. Matched by exact text or by ending.
EXACT_NONE = {
    "the visitor log was moved to the front desk in march.",
    "nobody has claimed the jacket left in the break room.",
    "the recycling collection has moved to wednesdays.",
    "catering is unchanged from last week.",
    "the site coordinator noted that the car park resurfacing runs through "
    "the end of the month.",
    "deliveries continue to arrive through the north gate.",
}
END_NONE = (
    "filled in the equipment survey.",
    "joined the site walkaround in march.",
    "completed the fire-safety refresher last quarter.",
    "attended the supplier training day in the spring.",
    "appeared on the old emergency contact sheet.",
    "sat on the interview panel in february.",
    "left a note about the recycling bins.",
    "has been chasing the maintenance ticket for the printer.",
    "keeps the sign-in sheet in the top drawer.",
    "is the point of contact for the fire drill.",
    "has the spare key to the supply cupboard.",
)


# --- builders: turn captured groups into a fact's fields ---------------------

def b_block(P, T, S):        return {"person": P[0], "block": T[0]}
def b_not_block(P, T, S):    return {"person": P[0], "block": T[0]}
def b_station(P, T, S):      return {"person": P[0], "station": S[0]}
def b_not_station(P, T, S):  return {"person": P[0], "station": S[0]}
def b_before(P, T, S):       return {"person": P[0], "other": P[1]}
def b_before_rev(P, T, S):   return {"person": P[1], "other": P[0]}
def b_imm(P, T, S):          return {"person": P[0], "other": P[1]}
def b_imm_rev(P, T, S):      return {"person": P[1], "other": P[0]}
def b_between(P, T, S):      return {"person": P[0], "a": P[1], "b": P[2]}
def b_between_third(P, T, S): return {"person": P[2], "a": P[0], "b": P[1]}
def b_station_before(P, T, S):
    # "N is on later than whoever has S": N is the person, S the station.
    return {"station": S[0], "person": P[0]}
def b_history(P, T, S):      return {"person": P[0], "station": S[0]}
def b_carshare(P, T, S):     return {"people": list(P)}
def b_empty(P, T, S):        return {}


# (kind, builder, template). {N}=name, {T}=time, {S}=station.
PATTERNS = [
    # block / not block
    ("block",             b_block,       "{N} is on the {T} block"),
    ("block",             b_block,       "{N} takes {T}, as things stand"),
    ("block",             b_block,       "The {T} block is {N}'s"),
    ("block",             b_block,       "{N} is the one who opens up at {T}"),
    ("block",             b_block,       "Whoever drew the {T} block, it was {N}"),
    ("block",             b_block,       "{T} is when {N} is scheduled"),
    ("not_block",         b_not_block,   "{N} is not on the {T} block"),
    ("not_block",         b_not_block,   "You will not find {N} on the {T} block"),
    ("not_block",         b_not_block,   "The {T} block is not {N}'s"),
    ("not_block",         b_not_block,   "{T} is the one block {N} is definitely not on"),
    ("not_block",         b_not_block,   "{N} is unavailable at {T}"),
    ("not_block",         b_not_block,   "{N} has a standing commitment that rules out the {T} block entirely"),
    ("req_not_block",     b_not_block,   "{N} wanted {T} and did not get it"),
    ("req_not_block",     b_not_block,   "{N} lobbied for {T} without success"),
    ("req_not_block",     b_not_block,   "{N} put in for the {T} block and was turned down"),
    ("req_not_block",     b_not_block,   "{N} asked to move to the {T} block; the request was declined"),
    ("cf_not_block",      b_not_block,   "{N} would have been a better fit for the {T} block"),
    ("cf_not_block",      b_not_block,   "If {N} had taken the {T} block the handover would have been smoother"),
    ("cf_not_block",      b_not_block,   "Putting {N} on the {T} block would have solved the parking problem"),
    ("cf_not_block",      b_not_block,   "Had the rota gone the other way, {N} would be on {T}"),
    # before
    ("before",            b_before,      "{N} works earlier in the day than {N}"),
    ("before",            b_before,      "{N} is done before {N} starts"),
    ("before",            b_before,      "{N}'s block falls somewhere earlier than {N}'s"),
    ("before",            b_before_rev,  "{N} comes later in the day than {N}"),
    ("before",            b_before_rev,  "By the time {N} starts, {N} has already been on"),
    ("before",            b_before_rev,  "{N} takes over from {N} later in the day"),
    # adjacency
    ("immediately_before", b_imm,        "{N} then {N}, back to back"),
    ("immediately_before", b_imm,        "{N} is on the block immediately before {N}"),
    ("immediately_before", b_imm,        "{N} hands straight over to {N}"),
    ("immediately_before", b_imm,        "There is no block between {N}'s and {N}'s, in that order"),
    ("immediately_before", b_imm_rev,    "{N} is on the block directly after {N}"),
    ("immediately_before", b_imm_rev,    "{N} relieves {N} directly, with no block in between"),
    # between
    ("between",           b_between,     "{N} is on after one of {N} and {N} and before the other"),
    ("between",           b_between,     "{N} sits between {N} and {N} on the rota"),
    ("between",           b_between,     "Put {N} between {N} and {N}, though not necessarily next to either"),
    ("between",           b_between,     "{N} works at some point between {N} and {N}, not necessarily adjacent to either"),
    ("between",           b_between,     "{N}'s block falls between {N}'s and {N}'s, in one order or the other"),
    ("between",           b_between_third, "Whichever way round {N} and {N} are, {N} is between them"),
    # station / not station
    ("station",           b_station,     "{N} is on {S}"),
    ("station",           b_station,     "{N} is assigned to {S}"),
    ("station",           b_station,     "{N} has {S} this week"),
    ("station",           b_station,     "{S} is {N}'s station"),
    ("station",           b_station,     "The {S} station is down to {N}"),
    ("station",           b_station,     "{S} is covered by {N}"),
    ("not_station",       b_not_station, "{N} is not on {S}"),
    ("not_station",       b_not_station, "{N} is not assigned to {S}"),
    ("not_station",       b_not_station, "{N} has not been put on {S}"),
    ("not_station",       b_not_station, "{S} is not {N}'s station"),
    ("not_station",       b_not_station, "You can rule {N} out for {S}"),
    ("not_station",       b_not_station, "{S} is covered by someone other than {N}"),
    # station ordering
    ("station_before",    b_station_before, "The {S} station is covered earlier in the day than {N}'s block"),
    ("station_before",    b_station_before, "Whoever is on {S} works earlier in the day than {N}"),
    ("station_before",    b_station_before, "The person on {S} precedes {N}"),
    ("station_before",    b_station_before, "{N} is on later than whoever has {S}"),
    ("station_before",    b_station_before, "Whoever has {S} is done before {N} starts"),
    ("station_before",    b_station_before, "{S} is covered before {N} comes on"),
    # flagged: probably no force, kept so we can test that claim
    ("history",           b_history,     "Back in the old arrangement, {N} took {S}"),
    ("history",           b_history,     "In the spring {N} held {S} for a while"),
    ("history",           b_history,     "Last month {N} was on {S}"),
    ("history",           b_history,     "On the previous cycle {S} was {N}'s"),
    ("carshare",          b_carshare,    "{N} and {N} car-share when their shifts allow"),
    ("open",              b_empty,       "Nobody could remember whether {N} was down for the {T} block"),
    ("open",              b_empty,       "The question of {N} on {T} was left open in the handover"),
    ("open",              b_empty,       "{N} and the {T} block came up, but nothing was minuted"),
    ("open",              b_empty,       "There was some disagreement about whether {N} had {T}"),
]


def strip_prefixes(line):
    s = line.strip()
    changed = True
    while changed:
        changed = False
        low = s.lower()
        for p in PREFIXES:
            if low.startswith(p):
                s = s[len(p):].strip()
                changed = True
                break
    return s


def parse_header(text):
    first = text.split("\n", 1)[0]
    grab = lambda pat: (re.search(pat, first) or [None, ""])[1]
    people = grab(r"on the rota:\s*(.*?)\.\s*Blocks")
    blocks = grab(r"Blocks run\s*(.*?),\s*one person per block")
    stations = grab(r"one person on each:\s*(.*?)\.\s*The people")
    holders = grab(r"The people on a station are\s*(.*?);")
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
    return {
        "people": split(people),
        "blocks": split(blocks),
        "stations": split(stations),
        "station_holders": split(holders),
    }


def template_regex(tmpl, name_alt, station_alt):
    parts = re.split(r"(\{N\}|\{T\}|\{S\})", tmpl)
    out, kinds = [], []
    for p in parts:
        if p == "{N}":
            out.append("(" + name_alt + ")")
            kinds.append("person")
        elif p == "{T}":
            out.append(r"(\d{2}:\d{2})")
            kinds.append("time")
        elif p == "{S}":
            out.append("(" + station_alt + ")")
            kinds.append("station")
        elif p:
            out.append(re.escape(p))
    return re.compile("^" + "".join(out) + "$", re.IGNORECASE), kinds


def canonical(value, options):
    low = value.lower()
    for o in options:
        if o.lower() == low:
            return o
    return value


def classify(core, names):
    low = core.lower().strip()
    if low in EXACT_NONE or any(low.endswith(e) for e in END_NONE):
        return "none", {}
    # patterns are written without the sentence-final period
    probe = core.strip()
    while probe.endswith("."):
        probe = probe[:-1].strip()
    core = probe
    name_alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    station_alt = "|".join(re.escape(s) for s in
                           sorted(("intake", "calibration", "packing"),
                                  key=len, reverse=True))
    for kind, builder, tmpl in PATTERNS:
        regex, kinds = template_regex(tmpl, name_alt, station_alt)
        m = regex.match(core)
        if not m:
            continue
        P, T, S = [], [], []
        for value, k in zip(m.groups(), kinds):
            if k == "person":
                P.append(canonical(value, names))
            elif k == "time":
                T.append(value)
            else:
                S.append(canonical(value, ("intake", "calibration", "packing")))
        return kind, builder(P, T, S)
    return "unknown", {}


def extract_item(item):
    text = item["text"]
    lines = text.split("\n")
    header = parse_header(text)
    facts = []
    for i, raw in enumerate(lines):
        if i < 2 or not raw.strip():   # line 0 header, line 1 blank
            continue
        core = strip_prefixes(raw)
        kind, fields = classify(core, header["people"])
        fact = {"line": i, "type": kind}
        fact.update(fields)
        fact["text"] = raw.strip()
        facts.append(fact)
    return {"id": item["id"], **header, "facts": facts}


def main():
    ap = argparse.ArgumentParser(description="Rules: shift notes -> facts JSON")
    ap.add_argument("items")
    ap.add_argument("--out", default="outputs/facts.json")
    ap.add_argument("--sample", default=None, help="print one item id and exit")
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text())
    if args.sample:
        for it in items:
            if it["id"] == args.sample:
                print(json.dumps(extract_item(it), indent=2))
                return
        sys.exit(f"no item {args.sample}")

    out = [extract_item(it) for it in items]

    counts = Counter()
    unknowns = defaultdict(list)
    for entry in out:
        for f in entry["facts"]:
            counts[f["type"]] += 1
            if f["type"] == "unknown":
                unknowns[entry["id"]].append(f["text"])

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    total = sum(counts.values())
    print(f"wrote {len(out)} items, {total} facts to {args.out}\n")
    print("facts by type:")
    for kind, n in counts.most_common():
        print(f"  {kind:20s} {n:4d}")
    n_unknown = counts.get("unknown", 0)
    print(f"\nunknown lines: {n_unknown}")
    seen = set()
    for iid, lines in unknowns.items():
        for ln in lines:
            if ln not in seen:
                seen.add(ln)
                print("  -", ln)
    if not n_unknown:
        print("  (none)")


if __name__ == "__main__":
    main()
