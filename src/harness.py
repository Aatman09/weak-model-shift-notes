#!/usr/bin/env python3
"""Pipeline: items.json -> (weak model labels each note line) -> solver -> answers.json.

    ./run data/items.json --budget 3x --out outputs/answers.json

Model calls per item: exactly 1 at 1x, 3 at 3x, 10 at 10x (majority vote per line).
Every call carries the X-Item-Id header.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import parse_header          # noqa: E402  (header is structural, not free text)
from model_extract import extract_with_model  # noqa: E402
from oneshot import HERE, load_env_file, resolve_config  # noqa: E402
from solve import solve_all               # noqa: E402

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

CALLS = {"1x": 1, "3x": 3, "10x": 10}


def make_complete(args: argparse.Namespace):
    if OpenAI is None:
        sys.exit("The openai package is required: pip install -r requirements.txt")
    env = load_env_file(HERE.parent / ".env")
    base_url, api_key, model = resolve_config(env, args)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=900, max_retries=0)
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
