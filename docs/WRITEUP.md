# Architecture under a fixed, weak model: write-up

**Headline.** Measured on the provided endpoint (`ibm-granite/granite-4.2-8b`, temperature 1.0, top_p 0.95, thinking off), on the 60 visible items, macro exact match:

| Budget | Visible wording | Re-worded constraint lines (held-out proxy) |
|---|---|---|
| 1x | 65.0% (mean of ten runs; range 56.7–71.7%) | 45.3% (41.7–50.0%) |
| 3x | 72.8% (mean of three disjoint triples; 70.0–76.7%) | 52.2% (50.0–55.0%) |
| 10x | 73.3% (one run) | 55.0% (one run) |

One-shot Granite scores 0.0%. The proxy drop of about 20 points is my best estimate of what unseen wording costs. Full tables, including every ablation, are in `docs/ablations.md`.

## How I characterised the model

One call per item with no scaffolding gives no usable schedule. So I asked a narrower question: can it classify a single sentence? I read many replies and found:

1. **Per sentence it is decent; per page it is not.** It usually picks the right constraint type for a sentence, and usually ignores filler, wishes and old arrangements. It cannot combine eight or so constraints into one schedule.
2. **Its errors are mostly misses.** On the final prompt, over ten samples per item, 489 constraints were missed, 213 were labelled with the wrong arguments and only 22 were invented (4668 correct). A missed constraint leaves too many schedules, so the item is declared ambiguous.
3. **It is inconsistent on the same sentence.** The same line is labelled correctly in one sample and `none` in the next. Voting helps with this kind of error.
4. **It rambles.** Often it writes a label and keeps reasoning, then runs out of tokens. The final run labelled 93% of note lines on average, and 98 of 600 calls labelled fewer than 90%.
5. **Format matters more than wording.** One stray quote in JSON invalidated a whole reply, and example annotations in the prompt were copied by the model. Flat `L5 | type | args` lines worked best.

## Components and why each exists

- **Model labels each note line** with one of eight constraint types or `none` (the only place the model is used). The prompt gives a rubric per type, with several paraphrases, plus rules: hedges and "noted twice" prefixes do not weaken a line, failed requests and the past are `none`, write the earlier person first, and when unsure, label.
- **Validation against the header:** names, times and stations must exist on the rota; otherwise the label is dropped.
- **Symbolic solver:** enumerates every assignment, keeps the ones satisfying all labels, and classifies by count (0 inconsistent, 1 unique, 2–4 ambiguous). The model never reasons about combinations.
- **Minimal-conflict citations:** deletes constraints one at a time and keeps a minimal contradictory set. Citations come from the note by line number, so they are verbatim. Citing every constraint instead drops 1x from 65.0% to 41.2%.
- **Per-line vote at 3x/10x:** a sample that never labelled a line abstains; the most common label wins if more than 20% of the voting samples chose it. Because misses dominate, favouring recall beats a strict majority (10x: 73.3% vs 70.0%).

Which labels carry the score (1x, ignoring a type): `station_before` −38 points, `between` −37, `block` −26, `station` −25, `immediately_before` −24, `before` −17, `not_block` −15, `not_station` −14.

## What I built and removed

- **Rules-based extractor** (`src/extract.py`): 100% on the visible items, 0.0% on the re-worded notes. It is built from the visible phrasings, so it is not in the submitted path; it is the reference I measure the model's labels against.
- **Whole-array JSON output:** removed after one malformed element lost whole items.
- **Annotated prompt examples:** removed because the model copied the annotations.
- **Swapping arguments by token type:** tried and removed; it changed nothing on saved replies.
- **A local run on a Q4 Ollama copy:** useful for development but not comparable (34% at 1x with the first prompt; more rambling), so it is not used for the headline.

Prompt versions on the endpoint went from 40% to 63% at 1x (table in `docs/ablations.md`).

## Where it breaks

- **I tuned on the visible items.** The rubric describes phrasing families I saw there, so the visible score flatters the system. The proxy (my own new phrasings, not the graders') is the fairer number, and even it keeps the original filler and wish lines unchanged.
- **Misses and rambling.** Missed constraints, mostly `between`, `not_station` and `station_before`, produce the main error: unique items declared ambiguous (4 of 20 at 10x), and ambiguous items with the wrong set of assignments. Voting cannot fix a phrasing the model misses in every sample.
- **Wrong direction or wrong middle person** on some `immediately_before` and `between` labels, which voting does not repair.
- **1x timeouts are not retried.** In one end-to-end 1x run, four calls timed out and those items scored zero; the budget forbids a second call, so the real 1x score has this extra variance (that run scored 60.0%).
- **Voting gives about +8 points at 3x and almost nothing from 3x to 10x.**
- **Not verified:** the counting proxy that enforces the budget; the code sends `X-Item-Id` on every call and makes exactly 1, 3 or 10 calls per item (checked from saved runs).
- **Case matrix at 10x (true to declared):** unique 14 / 4 ambiguous / 2 inconsistent; ambiguous 0 / 17 / 3; inconsistent 3 / 0 / 17. Over-declaring ambiguous is the main error direction, as the brief anticipates.

## Extraction: model-based or symbolic, and different authors

Constraint extraction is **model-based**. Two parts are symbolic: the **header** (rota, blocks, stations, who holds a station) is parsed with a regular expression, and the **solver**. The rules extractor is symbolic and not part of the submission.

With different authors the model will see wording outside my rubric. The proxy suggests a loss of roughly 20 points at every budget, concentrated in `before`, `between` and `station_before`, where the new phrasings are less like the examples. A closed list of eight constraint types limits the damage, but a real author's constraints may not fit the list at all, and the header regex assumes the generator's header sentence.

## With a month

1. **Constrained decoding** (a grammar for the label lines) to remove rambling and invalid output.
2. **Solver-guided repair** at 3x/10x: when the solver finds more than four candidates, re-ask only about lines labelled `none` that mention a person.
3. **A larger and more varied paraphrase set** (including filler and wishes), and tuning against it rather than the visible notes.
4. A model-based header parser, and an open-ended "other" constraint type.
5. Retry policy for timed-out calls within the budget at 3x/10x.
