#!/usr/bin/env python3
"""One-shot baseline for the shift-notes task: one model call per item.

The model (ibm-granite/granite-4.2-8b) is asked to solve each item directly in
the submission JSON shape, in a single call. This is the "characterise the
model" baseline -- it is expected to be poor; its value is the raw replies it
saves for inspection.

Usage:
    python3 src/oneshot.py data/items.json --budget 1x --out outputs/answers_oneshot.json
    python3 src/score.py data/visible_key.json outputs/answers_oneshot.json

Fixed by the brief, on every call: temperature=1.0, top_p=0.95, reasoning
disabled, and an X-Item-Id header. Exactly one model call happens per item at
every budget, which stays inside all of them (1x wants exactly one; 3x/10x want
at least one and at most three/ten).

Endpoint, model id and key are read from .env next to this file, or from
--base-url/--api-key/--model. The key is never printed or written anywhere.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # keep the module importable for offline parser tests
    OpenAI = None

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "ibm-granite/granite-4.2-8b"

SYSTEM = """\
You are reading a page of shift notes from a facility. The notes constrain \
which staff member works which time block and which station each \
station-holder covers. Work out the assignment, then answer in JSON only.

How to read the notes:
- Hedged lines ("As far as I know, ...", "Speaking from memory, ...", \
"I'm fairly sure ...") state their content exactly as firmly as unhedged \
lines. A hedge is not doubt.
- A restatement prefix ("It bears repeating:", "Noted twice in the handover:", \
"Worth restating, ...") does not weaken the statement after it.
- Lines about jackets, deliveries, surveys, interviews, keys, the visitor log \
and the like constrain nothing unless they say something about a block or a \
station. A line saying something "came up, but nothing was minuted" says \
nothing.
- There are five blocks, "07:00", "09:00", "11:00", "13:00", "15:00": one \
person per block, and each person works exactly one block. The header names \
the rota and the people on a station; there are three stations, "intake", \
"calibration" and "packing", one person on each.

Classify the item as exactly one of:
- "unique": exactly one assignment is consistent with the notes.
- "ambiguous": two to four assignments are consistent. List ALL of them.
- "inconsistent": no assignment is consistent. Cite statements that cannot \
all hold at once.

Answer with JSON only, in exactly one of these shapes:
{"case": "unique", "assignment": {"<name>": {"block": "<time>", "station": "<station>"}, "<name>": {"block": "<time>"}}}
{"case": "ambiguous", "assignments": [{"<name>": {"block": "<time>", "station": "<station>"}}, {"<name>": {"block": "<time>"}}]}
{"case": "inconsistent", "conflicts": ["<one full statement copied exactly from the notes>", "<another full statement>", "<a third full statement>"]}

Rules:
- Replace every <...> placeholder with the actual names, times and stations \
from the notes; never output the angle brackets.
- Every person on the rota must appear in every assignment you give.
- Give "station" only to people the notes put on a station.
- Copy names, block times and station names exactly as the notes write them.
- For "inconsistent": cite the smallest set of statements that cannot all \
hold at once and where dropping any one of them leaves a set that can. Cite \
each statement in full on its own, exactly as written, one entry per \
statement -- do not merge lines and do not paraphrase.
- Output the JSON object and nothing else.
"""


def load_env_file(path: Path) -> dict:
    """Minimal .env reader (KEY=VALUE lines). No dependency, no printing."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


def resolve_config(env: dict, args):
    """Find base URL / key / model: CLI > standard names > .env heuristics."""

    def pick(explicit, standard, needles, default=None):
        if explicit:
            return explicit
        for name in standard:
            value = os.environ.get(name) or env.get(name)
            if value:
                return value
        for source in (env, os.environ):
            for key, value in source.items():
                if value and any(n in key.upper() for n in needles):
                    return value
        return default

    base_url = pick(args.base_url,
                    ["OPENAI_BASE_URL", "OPENAI_API_BASE", "BASE_URL"],
                    ["URL", "BASE"])
    api_key = pick(args.api_key, ["OPENAI_API_KEY", "API_KEY"], ["KEY"])
    model = pick(args.model, ["MODEL_ID", "OPENAI_MODEL", "MODEL"],
                 ["MODEL"], DEFAULT_MODEL)
    if not base_url:
        sys.exit("No base URL found: pass --base-url or put one in .env")
    if not api_key:
        sys.exit("No API key found: pass --api-key or put one in .env")
    return base_url, api_key, model


def find_json_object(text: str):
    """First balanced {...} span in the text that parses as JSON, else None."""
    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            return None
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start += 1


def _clean_fields(fields):
    if not isinstance(fields, dict):
        return None
    out = {}
    for key in ("block", "station"):
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value.strip()
    return out or None


def _clean_assignment(assignment):
    if not isinstance(assignment, dict) or not assignment:
        return None
    out = {}
    for person, fields in assignment.items():
        if not isinstance(person, str):
            return None
        cleaned = _clean_fields(fields)
        if cleaned is None:
            return None
        out[person] = cleaned
    return out


def normalize_answer(obj):
    """Coerce a parsed model object into one of the three answer forms.

    Returns None if the shape is unusable. A bare assignment (no "case") is
    read as a unique claim, same as the scorer does.
    """
    if not isinstance(obj, dict):
        return None
    case = obj.get("case")
    if case is None:
        assignment = _clean_assignment(obj)
        return {"case": "unique", "assignment": assignment} if assignment else None
    if case == "unique":
        assignment = obj.get("assignment")
        if assignment is None:
            lst = obj.get("assignments")
            assignment = lst[0] if isinstance(lst, list) and len(lst) == 1 else None
        assignment = _clean_assignment(assignment)
        return {"case": "unique", "assignment": assignment} if assignment else None
    if case == "ambiguous":
        lst = obj.get("assignments")
        if lst is None and obj.get("assignment") is not None:
            lst = [obj["assignment"]]
        if not isinstance(lst, list) or not lst:
            return None
        cleaned = [_clean_assignment(x) for x in lst]
        if any(c is None for c in cleaned):
            return None
        if not (2 <= len(cleaned) <= 4):
            print(f"  note: ambiguous with {len(cleaned)} assignments "
                  f"(expected 2-4)", file=sys.stderr)
        return {"case": "ambiguous", "assignments": cleaned}
    if case == "inconsistent":
        conflicts = obj.get("conflicts")
        if conflicts is None:
            conflicts = obj.get("conflict_lines")
        if not isinstance(conflicts, list):
            return None
        conflicts = [c.strip() for c in conflicts
                     if isinstance(c, str) and c.strip()]
        return {"case": "inconsistent", "conflicts": conflicts}
    return None


def main():
    parser = argparse.ArgumentParser(
        description="One-shot baseline: one model call per item.")
    parser.add_argument("items", help="path to items.json")
    parser.add_argument("--budget", default="1x", choices=["1x", "3x", "10x"],
                        help="kept for CLI compatibility; this baseline always "
                             "uses exactly one call per item")
    parser.add_argument("--out", default="outputs/answers_oneshot.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="only run the first N items (for quick probes)")
    args = parser.parse_args()

    if OpenAI is None:
        sys.exit("The openai package is required: pip install openai")

    env = load_env_file(HERE.parent / ".env")
    base_url, api_key, model = resolve_config(env, args)
    print(f"model={model}  budget={args.budget} (one call per item)",
          file=sys.stderr)
    print(f"endpoint={base_url}", file=sys.stderr)

    items = json.loads(Path(args.items).read_text())
    if args.limit:
        items = items[: args.limit]

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
    answers, raws, errors = {}, {}, {}
    started = time.time()
    for n, item in enumerate(items, 1):
        item_id = item["id"]
        raw = ""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": f"Shift notes:\n\n{item['text']}"},
                ],
                temperature=1.0,
                top_p=0.95,
                max_tokens=1600,
                extra_body={"reasoning": {"enabled": False}},
                extra_headers={"X-Item-Id": item_id},
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:  # exactly one call per item: no retry
            errors[item_id] = f"{type(exc).__name__}: {exc}"
        raws[item_id] = raw
        answer = normalize_answer(find_json_object(raw))
        if answer is not None:
            answers[item_id] = answer
        case = answer["case"] if answer else "UNPARSED"
        print(f"[{n}/{len(items)}] {item_id}: {case}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(answers, indent=2) + "\n")
    raw_path = out_path.with_suffix(out_path.suffix + ".raw.json")
    raw_path.write_text(json.dumps(raws, indent=2) + "\n")
    if errors:
        err_path = out_path.with_suffix(out_path.suffix + ".errors.json")
        err_path.write_text(json.dumps(errors, indent=2) + "\n")
    print(f"wrote {len(answers)}/{len(items)} answers to {out_path} "
          f"in {time.time() - started:.0f}s; raw replies in {raw_path}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
