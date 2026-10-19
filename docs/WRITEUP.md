# Architecture under a fixed, weak model: write-up

**Headline.** Granite 8B with this harness reaches **34% macro exact match at 1x** (mean of three runs, range 33–35%) and **37% at 3x** on the 60 visible items, against 0% one-shot. 10x was not run. These numbers come from a local Ollama copy of `granite4.2:8b` (Q4_K_M), not the provided endpoint, because the endpoint key hit its $50 limit during development (see "Where it breaks"). Each 1x figure is one saved run; the 3x figure is one run of three calls per item. Full table: `docs/ablations.md`.

## How I characterised the model

I started with a one-call-per-item run and read the replies. The model could not produce a schedule. Then I asked a narrower question: can it classify one note sentence at a time? That was much better. It picks the right constraint type for most sentences ("Meera sits between Tomas and Rohan", "Packing is covered by someone other than Ayesha") and it ignores filler, wishes and old arrangements most of the time. Three further findings shaped the design:

1. **Its failures are lopsided.** Per sentence, it misses a real constraint far more often than it invents one. On the 1x run with the second prompt: 214 missed, 37 wrong, 16 invented. A missed constraint leaves too many candidate schedules, so the item is declared ambiguous or gets the wrong ambiguous set.
2. **It rambles.** Often it labels a line and then starts reasoning in the output ("…Wait need correct unique labeling…"). Over 180 calls only 80% of note lines received a label, and 74 of 180 calls labelled under 90% of their lines. This is the largest loss I did not fix.
3. **Format matters more than wording.** In JSON, one stray quote (`"line":5"`) invalidated a whole reply. When I added example annotations to the prompt, it copied them and spent its tokens on them. A flat `L5 | type | args` line, with the middle person of `between` written first, worked best.

## What each component is for

- **Model labels each line (the only place the model is used).** It outputs one constraint per note line from a closed list of eight types or `none`. The rest of the pipeline depends only on these labels, so a different model output changes the answer.
- **Validation against the header.** Every name, time and station must be on the rota; anything else is dropped. This removes invented names.
- **Symbolic solver.** It enumerates every assignment, keeps the ones satisfying all labelled constraints, and classifies the item by the count (0 = inconsistent, 1 = unique, 2–4 = ambiguous). The model never has to reason about combinations, which is what it cannot do.
- **Minimal-conflict citations.** For inconsistent items the solver deletes constraints one at a time and keeps those whose removal makes the set satisfiable. Citations are taken from the note by line number, so they are always verbatim. Ablation: citing every constraint drops the 1x score from 34.4% to 23.9%.
- **Per-line majority vote at 3x / 10x.** Independent samples are combined line by line.

## What each label type is worth (ablation, 1x mean)

Ignoring a type's labels costs: `station_before` −20 points, `between` −17, `immediately_before` −14, `block` −14, `before` −11, `station` −10, `not_block` −6, `not_station` −3. `between` and `station_before` carry the score and are also where the model misses the most.

## What I built and removed

- **Whole-array JSON output.** Removed: one malformed element lost the item. Replaced by one line per note.
- **Rules-based extractor (`src/extract.py`).** It scores 100% on the visible items but is built from the visible phrasings, so it is exactly the thing the held-out set is designed to break. It is not in the submitted path. I kept it as a reference to measure the model's labels and as an ablation row (100%, no model, not a valid submission).
- **Explanatory annotations in the prompt examples.** Removed: the model copied them.
- **Sending `{"reasoning": {"enabled": false}}` to Ollama.** This does not disable thinking there and used up the token cap; locally the harness sends `reasoning_effort: "none"` instead. Calls to the provided endpoint are unchanged.
- **Cutting off trailing commentary on a label line.** Kept. It moved 1x results by about one item, so I count it as neutral.

## Where it breaks

- **Truncation and rambling** (above). A line the model never labels cannot be recovered by voting or by the solver.
- **Over-declaring ambiguous.** On the 3x vote, 12 of 20 unique items were declared ambiguous, because a missed constraint leaves several candidates. 25 of 60 items had more than four candidates, which the solver can only answer with a wrong ambiguous guess. The directions of error are therefore the "honest failure" kind the brief describes, but they are still wrong answers. Kind accuracy is much higher than exact match: ambiguous 17/20, inconsistent 12/20, unique 5/20 correct kind on the 3x vote.
- **Voting helps little.** 3x is about two points above the 1x mean, within the run-to-run spread. The errors are systematic (the same phrasing is missed in every sample), so independent samples agree on them.
- **Held-out wording is untested.** I have no measurement on unseen phrasings. The prompt's examples come from the visible notes' sentence shapes, so I expect some drop.
- **Not verified end to end on the provided endpoint.** The key reached its limit, so the budget proxy and `X-Item-Id` header were not exercised against the real endpoint. The code sends the header on every call and makes exactly 1, 3 or 10 calls per item.
- **10x not run.** At about one minute per call on my GPU it is a ten-hour run. I did not run it, and I do not report a number for it.
- **Sampling variance.** Three 1x runs on the same prompt scored 33%, 35% and 35%.

## Extraction: model-based or symbolic, and different authors

Constraint extraction is **model-based**. Two parts are symbolic: the **header** (rota, blocks, stations, who holds a station) is read with a regular expression, and the **solver**. The rules extractor in `src/extract.py` is symbolic but not part of the submission.

If the notes were written by different people, the model-based labelling should degrade more gracefully than templates, because it classifies meaning into a closed list rather than matching strings; the rules extractor would fail on every unseen phrasing. I can only argue this, not show it: the 8B model is weak, my examples reflect the visible wording, and the header regex assumes the generator's exact header sentence. A real author might also write constraints that do not fit the eight types.

## With a month

1. **Constrained decoding** (a grammar for the label lines), which removes rambling and invalid output and should push coverage to about 100%.
2. **Solver-guided repair** at 3x/10x: when the solver finds more than four candidates, ask the model only about lines it labelled `none` that mention a person, rather than resampling everything.
3. **A held-out paraphrase set**, built by rewriting the visible notes in new styles, to measure generalisation instead of arguing it.
4. A model-based header parser, and a ninth type for constraints outside the current list.
5. A faster serving setup, so that 10x and many repeated runs are practical.
