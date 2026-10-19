| Configuration | 1x: mean (min-max of 10 runs) | 3x: vote, mean (min-max) of disjoint triples | 10x: vote |
|---|---|---|---|
| Full system (model labels + solver + minimal-conflict citations) | 65.0% (56.7%-71.7%) | 72.8% (70.0%-76.7%) | 73.3% |
| - majority vote (accept only labels most samples agree on) | 65.0% (56.7%-71.7%) | 72.2% (68.3%-76.7%) | 70.0% |
| - cite every constraint instead of a minimal conflict | 41.2% (33.3%-45.0%) | 46.7% (45.0%-48.3%) | 48.3% |
| - ignore `block` labels | 39.5% (33.3%-48.3%) | 45.6% (43.3%-48.3%) | 45.0% |
| - ignore `not_block` labels | 49.7% (45.0%-55.0%) | 54.4% (51.7%-58.3%) | 55.0% |
| - ignore `station` labels | 39.8% (35.0%-45.0%) | 44.4% (40.0%-48.3%) | 45.0% |
| - ignore `not_station` labels | 50.8% (45.0%-55.0%) | 56.7% (55.0%-60.0%) | 56.7% |
| - ignore `before` labels | 48.3% (43.3%-53.3%) | 53.9% (53.3%-55.0%) | 53.3% |
| - ignore `immediately_before` labels | 41.3% (35.0%-46.7%) | 47.8% (46.7%-50.0%) | 46.7% |
| - ignore `between` labels | 27.8% (21.7%-31.7%) | 29.4% (28.3%-31.7%) | 28.3% |
| - ignore `station_before` labels | 27.0% (21.7%-31.7%) | 31.7% (30.0%-33.3%) | 30.0% |
| Rules-only extractor + solver (no model; not a valid submission) | 100.0% | 100.0% | 100.0% |
| One-shot Granite, no harness (from the brief) | 0.0% | n/a | n/a |

## Re-worded notes (held-out proxy)

`src/paraphrase.py` re-renders every constraint line in a phrasing pool and hedge set that is
in neither the visible notes nor the model prompt; the truth and the key are unchanged
(conflict statements are mapped to the new wording). Filler, wish and history lines are
untouched. 10 samples per item.

| Notes | 1x: mean (min-max of 10 runs) | 3x: mean (min-max) of disjoint triples | 10x |
|---|---|---|---|
| Visible wording | 65.0% (56.7%-71.7%) | 72.8% (70.0%-76.7%) | 73.3% |
| Re-worded constraint lines | 45.3% (41.7%-50.0%) | 52.2% (50.0%-55.0%) | 55.0% |
| Rules-only extractor on re-worded notes | 0.0% | 0.0% | 0.0% |

Reproduce: `python3 src/paraphrase.py`, then run the system on `data/heldout_proxy/items.json`
and score with `python3 src/eval_runs.py <run> --proxy`.

## Prompt iterations (real endpoint, 3 samples each, mean 1x / 3x vote)

| Version | 1x | 3x |
|---|---|---|
| v1: first line format and rubric | 40.0% | 43.3% |
| v2: full rubric (direction, `between`, station lines), "never hesitate" | 55.0% | 60.0% |
| v3: + failed requests are `none`, token cap 600 to 1000, abstain + lower vote threshold | 54.4% | 65.0% |
| v4: + "when unsure, label it", extra `not_station` example | 63.3% | 70.0% |
| v4 on a fresh 10-sample run (final) | 65.0% | 72.8% |

Local Ollama copy (`granite4.2:8b`, Q4_K_M) with the v1 prompt: 34.4% at 1x, 36.7% at 3x.
