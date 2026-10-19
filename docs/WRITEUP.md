# Architecture under a fixed, weak model: write-up

**Headline.** Full system, provided endpoint (`ibm-granite/granite-4.2-8b`, temperature 1.0, top_p 0.95, thinking off), macro exact match. Cells are mean (min-max) over independent runs; 10x is one run.

| Notes | 1x | 3x | 10x |
|---|---|---|---|
| Visible (60 items) | 58.8% (56.7–60.0) | 70.0% (68.3–73.3) | 71.7% |
| **Test**: every line re-worded, frozen, never tuned on | **44.2%** (35.0–51.7) | **57.9%** (53.3–65.0) | **63.3%** |
| Dev: every line re-worded, used while tuning | 51.3% (48.3–53.3) | 67.1% (63.3–70.0) | 66.7% |

One-shot Granite scores 0.0%. The test row is the number to trust; it approximates the private held-out set but is my own re-wording, not theirs. Every ablation is in `docs/ablations.md`.

## The overfitting problem, and what I did about it

My first system scored 65.0% at 1x and 72.8% at 3x on the visible notes, with a prompt that quoted phrasings from them and was tuned on those same 60 items. When I re-worded every line (new vocabulary, hedges, filler, wishes), the same prompt fell to **35.0% at 1x and 38.3% at 3x** on a frozen test set. That gap is the honest measure of how much it had memorised.

So I rebuilt it: the prompt now defines each constraint by meaning and uses only invented examples (made-up names, a made-up station); two re-worded sets were generated, a dev set for tuning and a test set frozen before tuning. The price is visible accuracy (65.0% to 58.8% at 1x); the gain is that test is 44% to 63% instead of 35% to 38%.

## How I characterised the model

One call per item with no scaffolding gives no usable schedule. I then asked whether it can classify one sentence at a time. Findings from reading many replies:

1. **Per sentence it is decent; per page it is not.** It cannot combine eight constraints into a schedule.
2. **Errors are mostly misses and wrong arguments, rarely invented constraints.** Over ten samples per item on the first endpoint run: 489 missed, 213 wrong arguments, 22 invented (4668 correct).
3. **The same sentence is labelled differently from one sample to the next**, which voting can partly fix.
4. **It rambles.** Without constraints it explains itself in the output and runs out of tokens; coverage of note lines was 80% locally and about 93-95% on the endpoint.
5. **Format matters more than wording.** JSON broke on one stray quote; annotated examples got copied.

## Components and why each exists

- **Model labels each note line** (the only model use): a one-line restatement of what the line says about the *current* rota, then a type from a closed list of eight, or `none`. Writing the restatement first helped (+~15 points on dev).
- **Validation against the header:** names, times, stations must exist on the rota.
- **Symbolic solver:** enumerates every assignment, counts survivors (0 inconsistent, 1 unique, 2-4 ambiguous). The model never reasons about combinations.
- **Minimal-conflict citations:** deletion-based minimal contradictory set, cited verbatim by line number. Citing every constraint instead costs about 20 points at every budget.
- **Voting at 3x/10x:** a sample that never labelled a line abstains; the most common label wins if more than 20% of voting samples chose it. Because misses dominate, favouring recall beats strict majority (visible 10x 71.7% vs 70.0%; dev 3x 67.1% vs 62.8%).
- **Focused re-asking:** call 1 labels the page; later calls re-ask only uncertain lines (unlabelled, disagreeing, or `none` though the line contains a header time, a header station, or two rota people). Dev 3x: 67.1% vs 59.4% with plain voting. On the frozen test set one plain-voting run scored higher (68.3% at 3x vs 57.9% mean), so I cannot claim it helps in general; I kept it because I chose it on dev before looking at test, and I report both.

Which labels carry the score (visible, 1x, ignoring a type): `between` and `station_before` each cost about 33 points; `block`, `station`, `immediately_before` about 20.

## What I built and removed

- **Rules-based extractor (`src/extract.py`):** 100% on the visible items, 0% on re-worded notes. Not in the submitted path; kept as the reference the model's labels are checked against.
- **Prompt that quoted visible phrasings:** removed (the overfitting above).
- **Whole-array JSON output, annotated examples, argument swapping by token type:** removed (broke, copied, no effect).
- **Prompt versions v5/v6** (shorter rubric, more examples): scored below the baseline on dev, discarded.

## Where it breaks

- **Held-out wording.** Test 1x is 35-52% across runs. Misses and wrong direction concentrate in `between`, `station_before` and `immediately_before`.
- **Case matrix at 10x on test (true to declared):** unique 10 right, 3 ambiguous, 7 inconsistent; ambiguous 16 right, 4 inconsistent; inconsistent 18 right. Wrong labels create spurious contradictions, so unique items are often called inconsistent, the opposite of the over-ambiguous error seen earlier.
- **Timeouts.** In three test runs five calls each hung to the client timeout (15 minutes then; now 240 s). At 1x a lost call cannot be retried within the budget, which accounts for much of the 1x spread.
- **Voting is a small help.** About +11 points from 1x to 3x on visible, and 3x to 10x adds little.
- **My proxies are my wording.** They share my idea of what a re-wording looks like, so the real held-out set may be easier or harder.
- **Not verified:** the graders' counting proxy. Each item gets exactly 1, 3 or 10 calls at most (focused mode may stop earlier), with `X-Item-Id` sent on every call.

## Extraction: model-based or symbolic, and different authors

Constraint extraction is **model-based**. The **header** (rota, blocks, stations, station holders) is parsed with a regular expression and the **solver** is symbolic; the rules extractor is not part of the submission. With different authors the model sees wording outside my examples: the test set shows a drop of about 15 points at 1x and 10 at 10x relative to the visible notes, concentrated in ordering constraints. The header regex assumes the generator's header sentence and would need a model-based replacement for free-form headers.

## With a month

1. **Constrained decoding** (a grammar for label lines): removes rambling, invalid labels and most of the timeouts.
2. **A verifier call** that checks proposed labels against the sentence and the solver's contradictions, instead of resampling.
3. **A larger, adversarial paraphrase set** (several re-wording styles, not mine alone), including filler lines.
4. **A model-based header parser** and an "other" constraint type.
5. **Retry within the budget** at 3x/10x when a call times out.
