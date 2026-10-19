# Architecture under a fixed, weak model

Weak model: `ibm-granite/granite-4.2-8b`. Brief: `data/CANDIDATE_BRIEF.md`. Write-up: `docs/WRITEUP.md`. Ablations: `docs/ablations.md`.

**How it works.** The model labels each note line with one constraint from a closed list (or `none`); a symbolic solver enumerates every assignment that satisfies the labels and returns unique / ambiguous (all of them) / inconsistent (a minimal conflicting set, cited verbatim).

## Results (macro exact match, 60 items, provided endpoint)

| Budget | Visible wording | Re-worded constraint lines (proxy) |
|---|---|---|
| One-shot Granite, no harness (brief) | 0.0% | n/a |
| 1x | 65.0% (mean of 10 runs; 56.7-71.7%) | 45.3% |
| 3x (per-line vote) | 72.8% (mean of 3 triples; 70.0-76.7%) | 52.2% |
| 10x (per-line vote) | 73.3% (one run) | 55.0% |

I tuned the prompt on the visible items, so the proxy column is the fairer one. Ablations, prompt
history and the proxy method are in `docs/ablations.md`; the write-up is `docs/WRITEUP.md`.

## Run

Python 3.12.12.

```
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
./run data/items.json --budget <1x|3x|10x> --out outputs/answers.json
```

The endpoint, key and model id are read from `.env` (never committed). Budget = exactly 1, 3 or 10 model calls per item, each sent with an `X-Item-Id` header, `temperature=1.0`, `top_p=0.95`, reasoning disabled.

Local Ollama (development only): start the server yourself, then add
`--base-url http://localhost:11434/v1 --api-key ollama --model granite4.2:8b`.

Score and reproduce the ablations (no model calls, from saved replies):
```
python3 src/score.py data/visible_key.json outputs/answers.json
python3 src/eval_runs.py outputs/final_10x.json
python3 src/ablate.py outputs/final_10x.json
```
The saved replies in `outputs/` are git-ignored; re-create them with a 10x run (`./run data/items.json --budget 10x --out outputs/final_10x.json`).

## Layout

```
run                 entrypoint
src/harness.py      pipeline: model labels -> solver -> answers.json
src/model_extract.py  prompt, label parser, validation, per-line vote
src/solve.py        symbolic solver
src/eval_runs.py    offline scoring of saved multi-sample runs
src/ablate.py       ablation table
src/extract.py      rules extractor (reference only, not in the pipeline)
src/paraphrase.py   builds the re-worded held-out proxy in data/heldout_proxy/
src/diagnose.py     per-type label accuracy against the rules extractor
src/oneshot.py      one-call baseline
src/score.py        scorer from the brief
data/  docs/  outputs/
```
