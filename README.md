# Architecture under a fixed, weak model

`ibm-granite/granite-4.2-8b` on the provided endpoint (temperature 1.0, top_p 0.95, thinking off). Macro exact match; 1x and 3x are means over independent runs, 10x is one run.

| Notes | 1x | 3x | 10x |
|---|---|---|---|
| Visible (60 items) | 58.8% | 70.0% | 71.7% |
| **Test: every line re-worded, frozen, never tuned on** | **44.2%** (35-52) | **57.9%** | **63.3%** |
| Dev: re-worded, used while tuning | 51.3% | 67.1% | 66.7% |

One-shot Granite scores 0%. Trust the test row: it is my own re-wording, a proxy for the private held-out set. All ablations: [docs/ablations.md](docs/ablations.md).

## Approach

The model labels each note line with one of eight constraint types (or `none`); a symbolic solver enumerates every assignment, counts the survivors (0 inconsistent, 1 unique, 2-4 ambiguous) and, for inconsistent items, deletes constraints one at a time to find a minimal contradictory set, cited verbatim by line number. The model never reasons about combinations, which is what it cannot do.

**Characterising the model.** Whole-page answers fail; single-sentence classification works. Its errors are mostly misses and wrong arguments, rarely invented constraints (489 missed, 213 wrong, 22 invented, 4668 correct). The same sentence is labelled differently from sample to sample, it rambles until it runs out of tokens, and format matters more than wording (JSON broke on one stray quote; annotated examples were copied).

**Components, and what each is worth** (visible, 1x unless noted):
- *Minimal-conflict citations*: citing every constraint instead costs about 20 points.
- *Label types*: ignoring `between` or `station_before` costs about 33 points each; `block`, `station`, `immediately_before` about 20.
- *Header validation*: names, times and stations must exist on the rota.
- *Per-line vote at 3x/10x*: abstain on unlabelled lines, accept a label if more than 20% of voters chose it (misses dominate, so recall beats strict majority: dev 3x 67.1% vs 62.8%). Voting adds about 11 points from 1x to 3x; 3x to 10x adds little.
- *Focused re-asking* (later calls re-ask only uncertain lines): dev 3x 67.1% vs 59.4% for plain voting, but on one test run plain voting scored higher (68.3% vs 57.9%). I kept it because I chose it on dev before seeing test; the evidence is mixed.

**Built and removed.** A rules-based extractor (100% visible, 0% on re-worded notes; reference only); whole-array JSON output; annotated examples; token-type argument swapping (no effect); two shorter prompts (worse on dev).

**Overfitting.** My first prompt quoted visible phrasings and scored 65.0% / 72.8% (1x / 3x) on visible but 35.0% / 38.3% on the test set. I rewrote it to define constraints by meaning with invented examples. Visible 1x fell to 58.8%; test rose to 44.2% / 57.9% / 63.3%.

## Where it breaks

- Held-out wording: test 1x ranges 35-52%; misses and reversed direction concentrate in `between`, `station_before`, `immediately_before`.
- At 10x on test, wrong labels create false contradictions: 7 of 20 unique items are declared inconsistent.
- Calls that hang to the client timeout cannot be retried within a 1x budget; each test run lost five, which explains much of the 1x spread.
- My proxies share my idea of what a re-wording looks like. The graders' counting proxy was not tested; each item gets at most 1, 3 or 10 calls (focused mode can stop early) with `X-Item-Id` on every call.

## Extraction: model-based or symbolic?

Model-based. Two parts are symbolic: the header (rota, blocks, stations) is parsed with a regular expression, and the solver. With different authors the model sees wording outside my examples: test is about 15 points below visible at 1x and 8 at 10x, mostly in ordering constraints. The header regex assumes the generator's header sentence and would need a model-based replacement for free-form headers.

## With a month

Constrained decoding for the label lines (no rambling, fewer timeouts); a verifier call that checks labels against the sentence and the solver's contradictions; a larger, adversarial paraphrase set in several styles; retries within the budget at 3x/10x.

## Run

Python 3.12.12.

```
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
./run data/items.json --budget <1x|3x|10x> --out outputs/answers.json
```

The endpoint, key and model id are read from `.env` (never committed). Budget = exactly 1, 3 or 10 model calls per item, each sent with an `X-Item-Id` header, `temperature=1.0`, `top_p=0.95`, reasoning disabled.

Local Ollama (development only): start the server yourself, then add
`--base-url http://localhost:11434/v1 --api-key ollama --model granite4.2:8b`.

Reproduce the ablations. The ablation script re-scores saved replies without calling the model, so first
create the runs (about 600 calls each, a few cents on the provided endpoint):
```
python3 src/paraphrase.py            # rebuilds data/proxy_dev and data/proxy_test (deterministic)
for r in 1 2 3; do ./run data/proxy_test/items.json --budget 3x  --out outputs/t_f_r$r.json; done
./run data/proxy_test/items.json --budget 10x --out outputs/final10_test.json
python3 src/ablate.py --set test --runs t_f_r1,t_f_r2,t_f_r3,final10_test
```
Use `--set visible` with `data/items.json`, or `--set dev` with `data/proxy_dev/items.json`, for the other
tables. Score any run with `python3 src/eval_runs.py outputs/<run>.json --set <visible|dev|test>`, or one
answers file with `python3 src/score.py data/visible_key.json outputs/answers.json data/items.json`.

## Layout

```
run                    entrypoint: ./run <items.json> --budget <1x|3x|10x> --out <answers.json>
src/harness.py         pipeline: model labels each note line, solver answers
src/model_extract.py   prompt, label parser, header validation, per-line vote, focused re-asking
src/solve.py           symbolic solver (enumeration, minimal-conflict citations)
src/eval_runs.py       scores saved runs at 1x / 3x / 10x
src/ablate.py          ablation table
src/paraphrase.py      builds the re-worded dev and test sets
src/extract.py         header parser (used by the pipeline); rules extractor (reference + paraphrase builder only)
src/score.py           scorer from the brief
data/                  items, visible key, dev and test re-worded sets with keys
docs/ablations.md      all ablation tables
```
