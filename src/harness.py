#!/usr/bin/env python3
"""Pipeline: items.json -> (weak model labels each note line) -> solver -> answers.json.

    ./run data/items.json --budget 3x --out outputs/answers.json

Model calls per item: exactly 1 at 1x, 3 at 3x, 10 at 10x (majority vote per line).
Every call carries the X-Item-Id header.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import parse_header          # noqa: E402  (header is structural, not free text)
from model_extract import extract_with_model  # noqa: E402
from solve import solve_all               # noqa: E402

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "ibm-granite/granite-4.2-8b"


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


CALLS = {"1x": 1, "3x": 3, "10x": 10}


def make_complete(args: argparse.Namespace):
    if OpenAI is None:
        sys.exit("The openai package is required: pip install -r requirements.txt")
    env = load_env_file(HERE.parent / ".env")
    base_url, api_key, model = resolve_config(env, args)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=240, max_retries=0)
    print(f"model={model}  endpoint={base_url}", file=sys.stderr)
    # The graded endpoint takes {"reasoning": {"enabled": False}}. Local Ollama
    # ignores that and needs reasoning_effort="none" to switch thinking off.
    extra_body: dict = {"reasoning": {"enabled": False}}
    if "localhost" in base_url or "127.0.0.1" in base_url:
        extra_body = {"reasoning_effort": "none"}  # the reasoning object re-enables thinking

    def complete(item_id: str, system: str, user: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=1.0,
            top_p=0.95,
            max_tokens=2000,
            extra_body=extra_body,
            extra_headers={"X-Item-Id": item_id},
        )
        return r.choices[0].message.content or ""

    return complete


def main() -> None:
    ap = argparse.ArgumentParser(description="Model extraction + symbolic solver")
    ap.add_argument("items")
    ap.add_argument("--budget", default="1x", choices=list(CALLS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-focus", action="store_true",
                    help="ablation: every call relabels the whole page (plain voting)")
    ap.add_argument("--drop", default="",
                    help="fact types to ignore (ablation), comma-separated")
    args = ap.parse_args()

    items = json.loads(Path(args.items).read_text())
    if args.limit:
        items = items[: args.limit]
    complete = make_complete(args)
    calls = CALLS[args.budget]
    drop = tuple(x.strip() for x in args.drop.split(",") if x.strip())

    def run_item(item: dict) -> tuple[dict, dict]:
        entry = {"id": item["id"], **parse_header(item["text"])}
        return extract_with_model(entry, item, calls, complete, focus=not args.no_focus)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_item, items))

    entries = [e for e, _ in results]
    answers, stats = solve_all(entries, drop)
    print("solved:", stats, file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(answers, indent=2) + "\n")
    out.with_suffix(out.suffix + ".facts.json").write_text(
        json.dumps(entries, indent=2) + "\n")
    out.with_suffix(out.suffix + ".raw.json").write_text(
        json.dumps({e["id"]: d["raw"] for e, d in results}, indent=2) + "\n")
    print(f"wrote {len(answers)} answers to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
