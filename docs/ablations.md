| Configuration | 1x: mean (range of 3 runs) | 3x: majority vote |
|---|---|---|
| Full system (model labels + solver + minimal-conflict citations) | 34.4% (33.3%-35.0%) | 36.7% |
| - cite every constraint instead of a minimal conflict | 23.9% (23.3%-25.0%) | 25.0% |
| - ignore `block` labels | 20.6% (18.3%-21.7%) | 23.3% |
| - ignore `not_block` labels | 28.3% (28.3%-28.3%) | 31.7% |
| - ignore `station` labels | 23.9% (23.3%-25.0%) | 25.0% |
| - ignore `not_station` labels | 31.7% (30.0%-33.3%) | 35.0% |
| - ignore `before` labels | 23.3% (21.7%-25.0%) | 25.0% |
| - ignore `immediately_before` labels | 20.0% (18.3%-21.7%) | 23.3% |
| - ignore `between` labels | 17.8% (13.3%-21.7%) | 18.3% |
| - ignore `station_before` labels | 14.4% (13.3%-15.0%) | 15.0% |
| Rules-only extractor + solver (no model; not a valid submission) | 100.0% | 100.0% |
| One-shot Granite, no harness (from the brief) | 0.0% | n/a |
