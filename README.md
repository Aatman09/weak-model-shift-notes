# Architecture under a fixed, weak model

Brief: `data/CANDIDATE_BRIEF.md`.

```
run                 launcher: ./run <items.json> --budget <1x|3x|10x> --out <answers.json>  (placeholder: one-shot baseline)
src/extract.py      rule-based: shift notes -> facts JSON
src/solve.py        facts -> answers, then scores them
src/oneshot.py      one-call-per-item model baseline
src/score.py        scorer
data/               items.json, visible_key.json, example_answers.json, brief
outputs/            generated files (git-ignored)
docs/               ablation table and two-page writeup (todo)
```

Symbolic pipeline:
```
python3 src/extract.py data/items.json --out outputs/facts.json
python3 src/solve.py outputs/facts.json data/items.json --out outputs/answers_rules.json
```
Baseline: `./run data/items.json --budget 1x --out outputs/answers_oneshot.json` (needs `.env` at repo root).
