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
You turn shift notes into constraints. The header says who is on the rota, the time blocks \
(earliest first), the stations, and which people hold a station. Every person works exactly one \
block. Each note line is numbered L<n>.

For every note line, output exactly one label line:

L<n> | <gloss> | <type> | <arguments>

The gloss is what the line says about the current rota in at most eight plain words \
("Bea before Cal", "Dee middle of Alex and Bea", "past, not now", "a wish"), written before \
you choose the type.

Step 1. Does the line state how the CURRENT rota actually is? If not, the label is none. \
Lines that state nothing about the current rota include: requests, wishes or preferences, \
whether granted or refused; what would, could or should have happened; past arrangements \
(how things used to be or were before); things that are uncertain, undecided or unrecorded; \
and anything unrelated to blocks or stations. How confidently a line is worded does not matter: \
a line that is hedged ("I think", "apparently", "if I remember right") or repeated \
("to repeat", "again") counts exactly as much as a plain one.

Step 2. If it does, decide what it tells you and pick the type:
  block P T                       P's block is T.
  not_block P T                   P's block is not T.
  station P S                     P holds station S.
  not_station P S                 P does not hold station S.
  before A then B                 A works at an earlier block than B (any gap).
  immediately_before A then B     B's block is the very next block after A's (no block between them).
  between X between A and B       X's block is later than one of A and B and earlier than the \
other (either order).
  station_before S then P         whoever holds station S works at an earlier block than P.

How to get the arguments right:
- Names, times and station names must be copied exactly from the header.
- For before / immediately_before / station_before, ask "who works first?" and write that one \
first, whatever order the sentence mentions them in.
- For between, ask "who is in the middle?" and write that person first.
- A line that names a station but not who holds it, and compares it with a person's time, is \
station_before.
- A request or preference, even a refused one, says nothing about the actual block: none, never \
not_block. A past arrangement says nothing about the current station: none, never station.
- When unsure whether a line about the current rota is a constraint, label it: a missed \
constraint cannot be recovered later.

Examples. The names and the station "sorting" are made up; use the header's.
  Bea is booked for 11:00.                                       ->  block | Bea 11:00
  The 07:00 start is Cal's.                                      ->  block | Cal 07:00
  Alex works the 13:00 block, no change there.                   ->  block | Alex 13:00
  11:00 is not an option for Cal.                                ->  not_block | Cal 11:00
  Bea is not rostered at 09:00.                                  ->  not_block | Bea 09:00
  Keep Dee off the 15:00 block.                                  ->  not_block | Dee 15:00
  Dee covers sorting.                                            ->  station | Dee sorting
  Sorting has been assigned to Alex.                             ->  station | Alex sorting
  Cal is our sorting person this shift.                          ->  station | Cal sorting
  Sorting will be done by somebody besides Alex.                 ->  not_station | Alex sorting
  Bea does not do sorting.                                       ->  not_station | Bea sorting
  Cal has been left off sorting.                                 ->  not_station | Cal sorting
  Cal arrives once Bea has already done her block.               ->  before | Bea then Cal
  Alex works before Dee, though not necessarily right before.    ->  before | Alex then Dee
  Dee is on at a later time than Bea.                            ->  before | Bea then Dee
  Dee's block is the one just before Alex's.                     ->  immediately_before | Dee then Alex
  Bea comes in at the very next block after Cal.                 ->  immediately_before | Cal then Bea
  Alex and Dee work back-to-back blocks, Alex first.             ->  immediately_before | Alex then Dee
  Of Alex and Cal, one works before Bea and one after her.       ->  between | Bea between Alex and Cal
  Dee's slot is somewhere between Alex's and Bea's.              ->  between | Dee between Alex and Bea
  Cal is later than Bea but earlier than Dee, or the reverse.    ->  between | Cal between Bea and Dee
  Dee only starts after the sorting person.                      ->  station_before | sorting then Dee
  The person on sorting is on earlier than Cal.                  ->  station_before | sorting then Cal
  Sorting is staffed before Bea's block begins.                  ->  station_before | sorting then Bea
  Alex asked for 07:00, which was refused.                       ->  none
  Bea would rather have 13:00.                                   ->  none
  If Dee had 09:00 the day would run smoother.                   ->  none
  Bea did sorting last year.                                     ->  none
  Before the change, Alex was on sorting.                        ->  none
  It is unclear whether Dee is on 09:00.                         ->  none
  Whether Cal works 11:00 was never settled.                     ->  none
  Alex and Cal share a locker.                                   ->  none
  Bea is training a new starter.                                 ->  none

Output only the label lines, one per note line, and nothing else: no explanations, no \
comments, no question marks. Your output looks like this:
L2 | a wish | none
L3 | Bea holds sorting | station | Bea sorting
L4 | Bea earlier than Cal | before | Bea then Cal
L5 | Dee middle of Alex and Bea | between | Dee between Alex and Bea"""


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


def parse_sample(raw: str, entry: dict[str, Any], body: list[tuple[int, str]],
                 salvage: bool = True) -> dict[int, Fact | None]:
    """line_no -> validated fact (None = labelled none / unusable)."""
    wanted = {i for i, _ in body}
    out: dict[int, Fact | None] = {}
    for row in raw.splitlines():
        quoted = QUOTED.match(row)             # L6: "the note" -> type args
        if quoted:
            kind_args = quoted.group(2).split(None, 1)
            row = f"L{quoted.group(1)} | {kind_args[0]} | " + (kind_args[1] if len(kind_args) > 1 else "")
        parts = [p.strip() for p in row.split("|")]
        if len(parts) >= 3 and parts[1].lower() not in FIELDS and parts[1].lower() != "none" \
                and (parts[2].lower() in FIELDS or parts[2].lower().startswith("none")):
            row = " | ".join([parts[0]] + parts[2:])    # drop the gloss
        m = LABEL.match(re.split(CUT, row)[0])
        if not m:
            continue
        ln, kind, rest = int(m.group(1)), m.group(2).lower(), (m.group(3) or "")
        if ln in wanted and ln not in out:
            out[ln] = fact_from(kind, rest, entry)
    # Fallback for replies that drift into free-form reasoning: for a line the strict parse
    # never saw, use the last well-formed "<type> | <args>" in the text after its L<n> mention.
    marks = list(re.finditer(r"\bL(\d+)\b", raw)) if salvage else []
    for j, mk in enumerate(marks):
        ln = int(mk.group(1))
        if ln not in wanted or ln in out:
            continue
        seg = raw[mk.end(): marks[j + 1].start() if j + 1 < len(marks) else len(raw)]
        for hit in reversed(list(SALVAGE.finditer(seg))):
            fact = fact_from(hit.group(1).lower(), re.split(CUT, hit.group(2))[0], entry)
            if fact is not None:
                out[ln] = fact
                break
    return out


def fact_from(kind: str, rest: str, entry: dict[str, Any]) -> Fact | None:
    """One label's type and argument text -> validated fact, or None."""
    toks = [t for t in re.split(r"[\s,]+", rest) if t]
    if kind == "between" and "between" in [t.lower() for t in toks]:
        i = [t.lower() for t in toks].index("between")
        toks = toks[:i] + toks[i + 1:]      # "X between A and B" -> X A B
    args = [t for t in toks if t.lower() not in ("then", "and")]
    if kind not in FIELDS or len(args) != len(ARGS[kind]):
        return None
    return clean_fact({"type": kind, **dict(zip(ARGS[kind], args))}, entry)


SALVAGE = re.compile(r"\b(" + "|".join(sorted(FIELDS, key=len, reverse=True)) +
                     r")\s*\|\s*([^|\n]+)")


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


def uncertain_lines(samples: list[dict[int, Fact | None]], body: list[tuple[int, str]],
                    entry: dict[str, Any]) -> list[tuple[int, str]]:
    """Lines worth asking about again, judged from the samples so far.

    Uncertain = never labelled, labelled differently by different samples, or labelled
    none although it mentions a header time, a header station or two rota people. The cue
    is only which header strings a line contains, never how it is worded.
    """
    out = []
    for ln, text in body:
        labels = {fact_key(s[ln]) for s in samples if ln in s}
        if not labels or len(labels) > 1:
            out.append((ln, text))
            continue
        if labels == {"none"}:
            names = sum(p in text for p in entry["people"])
            cue = (any(b in text for b in entry["blocks"]) or names >= 2
                   or any(st.lower() in text.lower() for st in entry["stations"]))
            if cue:
                out.append((ln, text))
    return out


def extract_with_model(entry: dict[str, Any], item: dict[str, Any], calls: int,
                       complete: Callable[[str, str, str], str],
                       threshold: float = 0.2,
                       focus: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Up to `calls` model extractions for one item, combined by a per-line vote.

    Call 1 labels the whole page. With focus=True every later call re-asks only the
    uncertain lines (see uncertain_lines) and the loop stops early when none are left;
    with focus=False every call relabels the whole page (plain self-consistency).
    `complete(item_id, system, user)` makes exactly one chat call and returns the text.
    Returns (entry-with-facts for solve.py, debug info).
    """
    header, body = split_item(item["text"])
    samples, raws = [], []
    for c in range(calls):
        ask = body if (c == 0 or not focus) else uncertain_lines(samples, body, entry)
        if not ask:
            break
        try:
            raw = complete(item["id"], SYSTEM, build_user(header, ask))
        except Exception as exc:  # a failed call still counts against the budget
            raw = ""
            print(f"  {item['id']}: call failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
        raws.append(raw)
        samples.append(parse_sample(raw, entry, ask))
    chosen = vote(samples, body, threshold)
    facts = []
    for ln, text in body:
        f = chosen.get(ln)
        if f is not None:
            facts.append({"line": ln, **f, "text": text})
    return {**entry, "facts": facts}, {"raw": raws}
