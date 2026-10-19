# Ablations and results

All numbers are macro exact match, scored with the provided scorer, on the provided endpoint (`ibm-granite/granite-4.2-8b`, temperature 1.0, top_p 0.95, thinking off). Cells are mean (min-max) over independent runs; a run's first call is a 1x run and its first three calls a 3x run. Nothing here is tuned on the **test** set.

## Visible notes (60 items)
Set: visible; runs: f8_vis, f8_vis_r2, f8_vis_r3, final10_visible

| Configuration | 1x | 3x | 10x |
|---|---|---|---|
| Full system | 58.8% (56.7%-60.0%) | 70.0% (68.3%-73.3%) | 71.7% |
| - parser fallback for free-form replies | 58.3% (56.7%-60.0%) | 68.3% (65.0%-70.0%) | 71.7% |
| - low vote threshold (strict majority instead) | 58.8% (56.7%-60.0%) | 69.6% (66.7%-73.3%) | 70.0% |
| - minimal-conflict citations (cite every constraint) | 37.5% (35.0%-40.0%) | 46.2% (41.7%-51.7%) | 48.3% |
| - ignore `block` labels | 37.5% (36.7%-38.3%) | 46.7% (40.0%-51.7%) | 48.3% |
| - ignore `not_block` labels | 43.8% (41.7%-48.3%) | 53.3% (50.0%-56.7%) | 51.7% |
| - ignore `station` labels | 38.3% (36.7%-40.0%) | 46.7% (45.0%-48.3%) | 48.3% |
| - ignore `not_station` labels | 45.8% (45.0%-46.7%) | 55.0% (51.7%-58.3%) | 53.3% |
| - ignore `before` labels | 48.8% (46.7%-50.0%) | 56.2% (55.0%-58.3%) | 58.3% |
| - ignore `immediately_before` labels | 38.8% (35.0%-40.0%) | 47.1% (45.0%-50.0%) | 46.7% |
| - ignore `between` labels | 25.8% (23.3%-28.3%) | 27.5% (26.7%-28.3%) | 28.3% |
| - ignore `station_before` labels | 24.6% (21.7%-28.3%) | 27.9% (26.7%-30.0%) | 26.7% |
| Rules-only extractor + solver (no model; not a valid submission) | 100.0% | 100.0% | 100.0% |

## Dev: every line re-worded (used for tuning)
Set: dev; runs: f8_dev, f8_dev_r2, f8_dev_r3, final10_dev

| Configuration | 1x | 3x | 10x |
|---|---|---|---|
| Full system | 51.3% (48.3%-53.3%) | 67.1% (63.3%-70.0%) | 66.7% |
| - parser fallback for free-form replies | 50.4% (46.7%-53.3%) | 65.8% (63.3%-70.0%) | 63.3% |
| - low vote threshold (strict majority instead) | 51.3% (48.3%-53.3%) | 63.8% (61.7%-66.7%) | 63.3% |
| - minimal-conflict citations (cite every constraint) | 30.8% (25.0%-35.0%) | 41.7% (38.3%-45.0%) | 43.3% |
| - ignore `block` labels | 29.6% (28.3%-30.0%) | 42.5% (38.3%-46.7%) | 41.7% |
| - ignore `not_block` labels | 41.7% (38.3%-45.0%) | 55.4% (53.3%-58.3%) | 51.7% |
| - ignore `station` labels | 30.4% (26.7%-33.3%) | 41.7% (40.0%-45.0%) | 36.7% |
| - ignore `not_station` labels | 38.3% (33.3%-43.3%) | 49.6% (46.7%-53.3%) | 50.0% |
| - ignore `before` labels | 40.4% (38.3%-41.7%) | 52.5% (50.0%-55.0%) | 51.7% |
| - ignore `immediately_before` labels | 31.2% (26.7%-38.3%) | 43.3% (36.7%-48.3%) | 43.3% |
| - ignore `between` labels | 28.3% (25.0%-31.7%) | 32.9% (31.7%-35.0%) | 31.7% |
| - ignore `station_before` labels | 21.3% (20.0%-25.0%) | 26.7% (25.0%-28.3%) | 26.7% |

## Test: every line re-worded, frozen (never tuned on)
Set: test; runs: t_f_r1, t_f_r2, t_f_r3, final10_test

| Configuration | 1x | 3x | 10x |
|---|---|---|---|
| Full system | 44.2% (35.0%-51.7%) | 57.9% (53.3%-65.0%) | 63.3% |
| - parser fallback for free-form replies | 42.9% (31.7%-51.7%) | 52.9% (48.3%-60.0%) | 60.0% |
| - low vote threshold (strict majority instead) | 44.2% (35.0%-51.7%) | 58.8% (55.0%-63.3%) | 61.7% |
| - minimal-conflict citations (cite every constraint) | 26.3% (21.7%-31.7%) | 36.7% (33.3%-40.0%) | 40.0% |
| - ignore `block` labels | 27.9% (21.7%-35.0%) | 37.9% (35.0%-41.7%) | 40.0% |
| - ignore `not_block` labels | 34.6% (28.3%-41.7%) | 46.7% (45.0%-51.7%) | 51.7% |
| - ignore `station` labels | 29.2% (23.3%-36.7%) | 38.3% (33.3%-43.3%) | 40.0% |
| - ignore `not_station` labels | 32.1% (21.7%-40.0%) | 42.5% (38.3%-50.0%) | 46.7% |
| - ignore `before` labels | 32.9% (26.7%-40.0%) | 44.2% (38.3%-48.3%) | 46.7% |
| - ignore `immediately_before` labels | 26.7% (21.7%-31.7%) | 39.2% (35.0%-41.7%) | 48.3% |
| - ignore `between` labels | 21.7% (16.7%-26.7%) | 26.7% (21.7%-30.0%) | 28.3% |
| - ignore `station_before` labels | 20.0% (16.7%-23.3%) | 25.4% (21.7%-28.3%) | 26.7% |

## Prompt and voting variants (separate runs)

| Variant | 1x | 3x |
|---|---|---|
| Old prompt (tuned on visible), test | 35.0% | 38.3% |
| New prompt, plain voting (--no-focus), test | 53.3% | 68.3% |
| New prompt, plain voting, dev (mean of 3) | 52.2% (50.0%-53.3%) | 59.4% (58.3%-61.7%) |
| New prompt, plain voting, visible (mean of 3) | 58.9% (58.3%-60.0%) | 67.8% (63.3%-73.3%) |
