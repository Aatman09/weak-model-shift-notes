#!/usr/bin/env python3
"""Model layer: shift notes -> facts, extracted by the weak model.

The model reads the numbered note lines and labels each one with a constraint
from a small closed vocabulary (or "none"). It never sees the solving problem:
the solver in solve.py enumerates assignments from the labelled facts.

Why this split: the 8B model cannot reason its way to an assignment (0% one-shot)
but can classify one short sentence at a time. Citations are taken from the
notes by line number, so they are always verbatim.

Budget: `budget` calls per item. 1x is a single extraction; 3x / 10x draw that
many independent samples and take a per-line majority vote.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from typing import Any, Callable

Fact = dict[str, Any]

BLOCK_ORDER = ["07:00", "09:00", "11:00", "13:00", "15:00"]

SYSTEM = """\
You label lines of shift notes. Each note line is numbered L<n>. Output exactly one \
label line per note line, in this format:

L<n> | <type> | <arguments>

Types and arguments (names, times and stations exactly as in the header):
  block P T                        P works block T                 (T is a time like 11:00)
  not_block P T                    P does NOT work block T
  station P S                      P covers station S
  not_station P S                  P does NOT cover station S
  before A then B                  A's block is earlier than B's block; any gap
  immediately_before A then B      B's block comes directly after A's block, back to back
  between X between A and B        X's block lies between A's and B's, in either order
  station_before S then P          whoever covers station S works an earlier block than P
  none                             the line constrains nothing

How to decide:
- Every line is either a constraint about who works which block or station, or it is none. \
Decide quickly from the first match below and never explain or hesitate: no "Wait", no "?", no notes.
- A line that says where someone works (a block or a station), where they do NOT, or how two \
people's blocks are ordered IS a constraint. Label it. Most lines that name a person are.
- Hedges ("I'm fairly sure", "as far as I know", "from memory", "going off the roster") and \
restatement prefixes ("It bears repeating:", "Noted twice in the handover:", "This came up more \
than once, so:") do NOT weaken a line. Skip over them and label the statement itself.
- Ordering lines. Always write the EARLIER person first for before / immediately_before / \
station_before.
  "B comes later than A", "A is on earlier than B", "A's block falls earlier than B's", "A is done \
before B starts", "by the time B starts, A has already been on", "B takes over from A later in the \
day"  ->  before A then B.
  "B relieves A directly", "B is on the block directly after A", "A hands straight over to B", \
"A then B, back to back", "no block between A's and B's, in that order"  ->  \
immediately_before A then B.
  "P is on later than whoever has packing", "whoever has packing is on before P", "the person on \
packing precedes P", "packing is covered earlier in the day than P's block", "whoever is on \
intake works earlier in the day than P"  ->  station_before packing then P.  (A station name \
plus a comparison of time is ALWAYS station_before, never none.)
- Between lines. "X sits between A and B", "put X between A and B, not necessarily next to \
either", "whichever way round A and B are, X is between them", "X is on after one of A and B and \
before the other"  ->  between X between A and B.   (The middle person X is written FIRST.)
- Block lines. "P takes 09:00", "P is the one who opens up at 09:00", "whoever drew the 09:00 \
block, it was P"  ->  block P 09:00.   "You will not find P on the 09:00 block", "P is \
unavailable at 09:00", "P has a commitment that rules out 09:00", "09:00 is the one block P is \
not on"  ->  not_block P 09:00.
- Station lines. "P is on S", "S is down to P", "P is assigned to S", "S is P's station"  ->  \
station P S.   "S is covered by someone other than P", "S is not P's station", "P has not been \
put on S", "P is not on S", "P is not assigned to S"  ->  not_station P S.
- A request or wish that failed ("turned down", "did not get it") says nothing about the real \
block: label it none, not not_block.
- Use none ONLY for: filler (deliveries, keys, surveys, the jacket, car-sharing, walkarounds, \
notes about bins); requests that were refused ("lobbied for 11:00 without success", "put in for \
the 13:00 block and was turned down", "wanted 15:00 and did not get it"); regrets and \
hypotheticals ("would have been a better fit for", "If X had taken ..."); anything about the \
PAST ("last month", "in the spring", "back in the old arrangement", "used to", "took ... then") -- \
past stations and blocks do NOT bind the current rota; open questions ("nobody could remember \
whether", "came up, but nothing was minuted", "some disagreement about whether").
- When unsure between a constraint label and none, choose the constraint label: a wrong label is \
caught by a later check, but a missed constraint cannot be recovered. A line that pairs a person \
with a block, a station, or another person's position in the day is almost never none.
- Never invent names: arguments must be people, times or stations from the header.

Examples (the notes themselves are not repeated in your output):
  "Tomas is on intake."                                       ->  station | Tomas intake
  "Calibration is not Priya's station."                       ->  not_station | Priya calibration
  "Ingrid comes later in the day than Priya."                 ->  before | Priya then Ingrid
  "Ayesha takes over from Daniel later in the day."           ->  before | Daniel then Ayesha
  "Ingrid relieves Meera directly, with no block between."    ->  immediately_before | Meera then Ingrid
  "Tomas is on later than whoever has packing."               ->  station_before | packing then Tomas
  "Whoever is on intake works earlier than Priya."            ->  station_before | intake then Priya
  "Whoever drew the 13:00 block, it was Rohan."               ->  block | Rohan 13:00
  "You will not find Daniel on the 15:00 block."              ->  not_block | Daniel 15:00
  "Meera has not been put on intake."                         ->  not_station | Meera intake
  "Priya is unavailable at 11:00."                            ->  not_block | Priya 11:00
  "Rohan put in for the 09:00 block and was turned down."     ->  none
  "Tomas wanted 09:00 and did not get it."                    ->  none
  "Last month Samuel was on calibration."                     ->  none
  "Nadia sits between Priya and Tomas on the rota."           ->  between | Nadia between Priya and Tomas
  "Put Rohan between Samuel and Priya, not next to either."   ->  between | Rohan between Samuel and Priya
  "Meera has the spare key."                                  ->  none

Your output looks like this and contains nothing else:
L2 | none
L3 | station | Tomas intake
L4 | none
L5 | between | Nadia between Priya and Tomas
L6 | before | Priya then Ingrid

For lines that constrain nothing write just "L<n> | none". Output only the label lines: no \
explanations, no comments, no notes."""


def split_item(text: str) -> tuple[str, list[tuple[int, str]]]:
    """(header, [(line_no, text), ...]) with the same line numbering as extract.py."""
    lines = text.split("\n")
    body = [(i, raw.strip()) for i, raw in enumerate(lines)
            if i >= 2 and raw.strip()]
    return lines[0], body


def build_user(header: str, body: list[tuple[int, str]]) -> str:
    notes = "\n".join(f"L{i}: {t}" for i, t in body)
    return f"Header: {header}\n\nNotes:\n{notes}"


FIELDS: dict[str, tuple[str, ...]] = {
    "block": ("person", "block"),
    "not_block": ("person", "block"),
    "station": ("person", "station"),
    "not_station": ("person", "station"),
    "before": ("person", "other"),
    "immediately_before": ("person", "other"),
    "between": ("person", "a", "b"),
    "station_before": ("station", "person"),
}


# argument order in the line format
ARGS: dict[str, tuple[str, ...]] = {
    "block": ("person", "block"),
    "not_block": ("person", "block"),
    "station": ("person", "station"),
    "not_station": ("person", "station"),
    "before": ("person", "other"),
    "immediately_before": ("person", "other"),
    "between": ("person", "a", "b"),
    "station_before": ("station", "person"),
}


def _canon(value: Any, options: list[str]) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().rstrip(".")
    for o in options:
        if o.lower() == v.lower():
            return o
    return None


def clean_fact(obj: Any, entry: dict[str, Any]) -> Fact | None:
    """Validate one model-labelled fact against the header; None if unusable."""
    if not isinstance(obj, dict):
        return None
    kind = obj.get("type")
    if kind not in FIELDS:
        return None
    options = {"person": entry["people"], "other": entry["people"],
               "a": entry["people"], "b": entry["people"],
               "block": entry["blocks"], "station": entry["stations"]}
    out: Fact = {"type": kind}
    for f in FIELDS[kind]:
        val = _canon(obj.get(f), options[f])
        if val is None:
            return None
        out[f] = val
    # distinct people in the relational types
    people = [out[f] for f in ("person", "other", "a", "b") if f in out]
    if len(set(people)) != len(people):
        return None
    return out


def fact_key(fact: Fact | None) -> str:
    if fact is None:
        return "none"
    f = dict(fact)
    if f["type"] == "between":              # a,b are unordered
        f["a"], f["b"] = sorted((f["a"], f["b"]))
    return json.dumps(f, sort_keys=True)


# The model often keeps talking after a valid label on the same line; keep only the label.
CUT = r"<-|#|->|\u2192|\(|\?|;|\.\s|:\s"
QUOTED = re.compile(r"^\s*L?(\d+)\s*[|:]\s*[\"'].*?[\"']\s*(?:->|\u2192)\s*([a-z_]+(?:\s.*)?)$", re.I)
LABEL = re.compile(r"^\s*L?(\d+)\s*[|:]\s*([a-z_]+)\s*(?:[|:]\s*(.*))?$", re.I)


def parse_sample(raw: str, entry: dict[str, Any],
                 body: list[tuple[int, str]]) -> dict[int, Fact | None]:
    """line_no -> validated fact (None = labelled none / unusable)."""
    wanted = {i for i, _ in body}
    out: dict[int, Fact | None] = {}
    for row in raw.splitlines():
        quoted = QUOTED.match(row)             # L6: "the note" -> type args
        if quoted:
            kind_args = quoted.group(2).split(None, 1)
            row = f"L{quoted.group(1)} | {kind_args[0]} | " + (kind_args[1] if len(kind_args) > 1 else "")
        m = LABEL.match(re.split(CUT, row)[0])
        if not m:
            continue
        ln, kind, rest = int(m.group(1)), m.group(2).lower(), (m.group(3) or "")
        if ln not in wanted or ln in out:
            continue
        toks = [t for t in re.split(r"[\s,]+", rest) if t]
        if kind == "between" and "between" in [t.lower() for t in toks]:
            i = [t.lower() for t in toks].index("between")
            toks = toks[:i] + toks[i + 1:]      # "X between A and B" -> X A B
        args = [t for t in toks if t.lower() not in ("then", "and")]
        if kind not in FIELDS or len(args) != len(ARGS[kind]):
            out[ln] = None
            continue
        out[ln] = clean_fact({"type": kind, **dict(zip(ARGS[kind], args))}, entry)
    return out


def vote(samples: list[dict[int, Fact | None]], body: list[tuple[int, str]],
         threshold: float = 0.5, abstain: bool = True) -> dict[int, Fact | None]:
    """Per-line vote over samples.

    A sample that never labelled a line abstains (abstain=True) instead of voting
    "none". The most common non-none label wins if more than `threshold` of the
    voting samples chose it; lowering the threshold favours recall, which is the
    model's weak side (it misses constraints far more often than it invents them).
    """
    result: dict[int, Fact | None] = {}
    for ln, _ in body:
        voters = [s for s in samples if ln in s] if abstain else samples
        if not voters:
            result[ln] = None
            continue
        keys = [fact_key(s.get(ln)) for s in voters]
        counts = Counter(k for k in keys if k != "none")
        if not counts:
            result[ln] = None
            continue
        best, n = counts.most_common(1)[0]
        if n / len(voters) > threshold:
            result[ln] = next(s.get(ln) for s in voters if fact_key(s.get(ln)) == best)
        else:
            result[ln] = None
    return result


def extract_with_model(entry: dict[str, Any], item: dict[str, Any], calls: int,
                       complete: Callable[[str, str, str], str],
                       threshold: float = 0.2) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run `calls` model extractions for one item and vote.

    `complete(item_id, system, user)` makes exactly one chat call and returns the text.
    Returns (entry-with-facts for solve.py, debug info).
    """
    header, body = split_item(item["text"])
    user = build_user(header, body)
    samples, raws = [], []
    for _ in range(calls):
        try:
            raw = complete(item["id"], SYSTEM, user)
        except Exception as exc:  # a failed call still counts against the budget
            raw = ""
            print(f"  {item['id']}: call failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        raws.append(raw)
        samples.append(parse_sample(raw, entry, body))
    chosen = vote(samples, body, threshold)
    facts = []
    for ln, text in body:
        f = chosen.get(ln)
        if f is not None:
            facts.append({"line": ln, **f, "text": text})
    return {**entry, "facts": facts}, {"raw": raws}
