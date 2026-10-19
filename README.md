# Architecture under a fixed, weak model

Weak model: `ibm-granite/granite-4.2-8b`. Brief: `data/CANDIDATE_BRIEF.md`. Write-up: `docs/WRITEUP.md`. Ablations: `docs/ablations.md`.

**How it works.** The model labels each note line with one constraint from a closed list (or `none`); a symbolic solver enumerates every assignment that satisfies the labels and returns unique / ambiguous (all of them) / inconsistent (a minimal conflicting set, cited verbatim).

## Results (60 visible items, macro exact match)

Measured with a local Ollama copy of `granite4.2:8b` (Q4_K_M), not the provided endpoint.

| Setting | Result |
|---|---|
| One-shot Granite, no harness (brief) | 0.0% |
| This system, 1x | 34.4% (three runs: 33.3%, 35.0%, 35.0%) |
| This system, 3x (per-line majority vote) | 36.7% (one run) |
| This system, 10x | not run |

Per-component ablations are in `docs/ablations.md`.

## Run

Python 3.12.12.

```
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
./run data/items.json --budget <1x|3x|10x> --out outputs/answers.json
```

The endpoint, key and model id are read from `.env` (never committed). Budget = exactly 1, 3 or 10 model calls per item, each sent with an `X-Item-Id` header, `temperature=1.0`, `top_p=0.95`, reasoning disabled.

Local Ollama: start the server yourself, then add
`--base-url http://localhost:11434/v1 --api-key ollama --model granite4.2:8b`.

Score and reproduce the ablations (no model calls, from saved replies):
```
python3 src/score.py data/visible_key.json outputs/answers.json
python3 src/eval_runs.py outputs/local_3x.json
python3 src/ablate.py outputs/local_3x.json
```
The saved replies in `outputs/` are git-ignored; re-create them with a 3x run.

## Layout

```
run                 entrypoint
src/harness.py      pipeline: model labels -> solver -> answers.json
src/model_extract.py  prompt, label parser, validation, per-line vote
src/solve.py        symbolic solver
src/eval_runs.py    offline scoring of saved multi-sample runs
src/ablate.py       ablation table
src/extract.py      rules extractor (reference only, not in the pipeline)
src/oneshot.py      one-call baseline
src/score.py        scorer from the brief
data/  docs/  outputs/
```
