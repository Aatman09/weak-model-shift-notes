# Architecture under a fixed, weak model

Weak model: `ibm-granite/granite-4.2-8b`. Brief: `data/CANDIDATE_BRIEF.md`. Write-up: `docs/WRITEUP.md`. Ablations: `docs/ablations.md`.

**How it works.** The model labels each note line with one constraint from a closed list (or `none`); a symbolic solver enumerates every assignment that satisfies the labels and returns unique / ambiguous (all of them) / inconsistent (a minimal conflicting set, cited verbatim).

## Results (macro exact match, provided endpoint)

| Notes | 1x | 3x | 10x |
|---|---|---|---|
| One-shot Granite, no harness (brief) | 0.0% | | |
| Visible, 60 items | 58.8% | 70.0% | 71.7% |
| **Test: every line re-worded, frozen, never tuned on** | **44.2%** | **57.9%** | **63.3%** |
| Dev: re-worded, used while tuning | 51.3% | 67.1% | 66.7% |

1x and 3x are means over independent runs (the test 1x range is 35.0-51.7%); 10x is one run. My first
prompt quoted visible phrasings and scored 65.0% / 72.8% on visible but only 35.0% / 38.3% on the test set,
so the current prompt describes constraints by meaning with invented examples. Details, ablations and the
prompt history are in `docs/ablations.md`; the write-up is `docs/WRITEUP.md`.

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
python3 src/paraphrase.py            # builds data/proxy_dev and data/proxy_test
python3 src/eval_runs.py outputs/final10_test.json --set test
python3 src/ablate.py --set test --runs t_f_r1,t_f_r2,t_f_r3,final10_test
```
The saved replies in `outputs/` are git-ignored; re-create them with a 10x run (for example `./run data/proxy_test/items.json --budget 10x --out outputs/final10_test.json`).

## Layout

```
run                 entrypoint
src/harness.py      pipeline: model labels -> solver -> answers.json
src/model_extract.py  prompt, label parser, validation, per-line vote
src/solve.py        symbolic solver
src/eval_runs.py    offline scoring of saved multi-sample runs
src/ablate.py       ablation table
src/extract.py      rules extractor (reference only, not in the pipeline)
src/paraphrase.py   builds the re-worded dev and test proxies in data/proxy_dev, data/proxy_test
src/diagnose.py     per-type label accuracy against the rules extractor
src/oneshot.py      one-call baseline
src/score.py        scorer from the brief
data/  docs/  outputs/
```
