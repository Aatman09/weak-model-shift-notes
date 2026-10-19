#!/usr/bin/env python3
"""Compare the model's labels in a saved run with the rules extractor (reference only).

    python3 src/diagnose.py outputs/final_10x.json [--show] [--set visible|dev|test]
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_runs import load_set, set_arg              # noqa: E402
from extract import extract_item, parse_header      # noqa: E402
from model_extract import fact_key, parse_sample, split_item  # noqa: E402

SOLV = {"block", "not_block", "station", "not_station", "before",
        "immediately_before", "between", "station_before"}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    # Reference labels come from the original notes: the proxies keep every line in place.
    originals = {it["id"]: it for it in json.loads((root / "data/items.json").read_text())}
    items, _ = load_set(set_arg())
    raws = json.loads(Path(sys.argv[1] + ".raw.json").read_text())
    show = "--show" in sys.argv
    tp, fn, wr, fp = (collections.Counter() for _ in range(4))
    cov, examples = [], collections.defaultdict(list)
    for it in items:
        ref = {f["line"]: f for f in extract_item(originals[it["id"]])["facts"]
               if f["type"] in SOLV}
        entry = {"id": it["id"], **parse_header(it["text"])}
        _, body = split_item(it["text"])
        text = dict(body)
        for raw in raws[it["id"]]:
            got = {int(m.group(1)) for m in re.finditer(r"^\s*L?(\d+)\s*[|:]", raw, re.M)}
            cov.append(len({l for l, _ in body} & got) / len(body))
            mod = {l: f for l, f in parse_sample(raw, entry, body).items() if f}
            k = lambda f: fact_key({a: b for a, b in f.items() if a not in ("line", "text")})
            for l in set(ref) | set(mod):
                a, b = ref.get(l), mod.get(l)
                if a and b and k(a) == k(b):
                    tp[a["type"]] += 1
                elif a and not b:
                    fn[a["type"]] += 1
                    examples[("miss", a["type"])].append(text[l])
                elif b and not a:
                    fp[b["type"]] += 1
                    examples[("spurious", b["type"])].append(text[l])
                else:
                    wr[a["type"]] += 1
                    examples[("wrong", a["type"])].append(f'{text[l]}  => {b["type"]}')
    print(f"mean line coverage {sum(cov)/len(cov):.3f}; calls under 90%: "
          f"{sum(c < .9 for c in cov)}/{len(cov)}")
    print(f"{'type':20s} correct missed wrong spurious")
    for t in sorted(SOLV):
        print(f"{t:20s} {tp[t]:7d} {fn[t]:6d} {wr[t]:5d} {fp[t]:8d}")
    if show:
        for key, v in sorted(examples.items()):
            print("##", key, len(v))
            for x in list(dict.fromkeys(v))[:6]:
                print("   ", x)


if __name__ == "__main__":
    main()
