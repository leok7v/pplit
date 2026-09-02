---
type: Repo
title: pplit
description: Chat-framed perplexity and teacher-free top-k for any
  instruction-trained model, from one text corpus, over llama.cpp.
okf_version: "1.0"
timestamp: 2026-08-30T05:10:00Z
---

# pplit

README.md is the front door: what this is, how to build it, how to run it, and
how to read a result line. This file is the other half: why it exists, what the numbers mean, and what
a quoted number may and may not claim. It does not repeat the README.

This is the public edition. Approaches that were tried and refuted, and notes
on the source, are in the development repository.

Rationale, measurements and dead ends live here rather than in the source.


## 1. Why this exists

`llama-perplexity` cannot measure an RL-instruction-trained model. Its protocol
(`tools/perplexity/perplexity.cpp`, `perplexity()`) tokenizes the whole `-f`
file, splits it into `n_ctx` chunks and scores the second half of each
(`const int first = n_ctx/2`). Sound and standard, and useless here: gemma-4-it
reads raw prose as something to ANSWER, not continue.

Measured on `gemma-4-12B-it-qat`, ctx 512, 15 chunks, 3840 scored tokens each,
every corpus trimmed to 120 KB so the comparison is fair. These rows are ONE
model; section 1.1 re-measures the claim across nine and finds the effect is
far weaker outside the dense line:

```
┌─────────────────────────────────────┬────────┬───────────────────────────┐
│ corpus                              │  ppl   │          register         │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ wiki.test.raw (WikiText-2)          │ 308.79 │ encyclopedia prose        │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ IFBench prompts                     │ 173.16 │ user                      │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ no_robots, HUMAN-written assistant  │ 155.14 │ assistant                 │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ the model's OWN greedy replies, raw │ 77.42  │ assistant, self-generated │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ IFBench prompt + reply, raw Q/A     │ 47.78  │ Q/A                       │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ IFBench multi-turn, model-written   │ 45.08  │ assistant, LLM-style      │
├─────────────────────────────────────┼────────┼───────────────────────────┤
│ the same text, chat-framed          │ 7.17   │ assistant + framing       │
└─────────────────────────────────────┴────────┴───────────────────────────┘
```


### 1.1 The same claim, measured across thirteen models

Section 1 is one model on seven corpora. This is thirteen models on three
protocols, one host, one backend, so the rows are comparable to each other.
`raw replies` is this repo's corpus with the framing stripped, so register is
held right and only the frame is missing.

```
┌─────────────────┬───────────┬────────────┬─────────────┬─────────────┬────────┐
│ model           │   quant   │ WikiText-2 │ raw replies │ chat-framed │ ratio  │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 E2B     │ q4_0 QAT  │ 46.18      │ 9.91        │ 6.4478      │ 7.2x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 E4B     │ q4_0 QAT  │ 29.98      │ 7.71        │ 5.1457      │ 5.8x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 12B     │ q4_0 QAT  │ 245.73     │ 66.07       │ 7.5881      │ 32.4x  │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 26B-A4B │ q4_0 QAT  │ 740.16     │ 179.50      │ 7.0651      │ 104.8x │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 31B     │ q4_0 QAT  │ 1261.09    │ 64.70       │ 8.3109      │ 151.7x │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 E2B     │ Q4_K_M    │ 145.79     │ 28.65       │ 17.4717     │ 8.3x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 E4B     │ Q4_K_M    │ 56.68      │ 12.22       │ 9.7030      │ 5.8x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ gemma-4 12B     │ Q4_K_M    │ 521.65     │ 84.21       │ 8.4448      │ 61.8x  │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ Qwen3.5 0.8B    │ Q4_K_M    │ 18.17      │ 9.28        │ 6.8480      │ 2.7x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ Qwen3.5 2B      │ Q4_K_M    │ 12.13      │ 6.81        │ 5.4447      │ 2.2x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ Qwen3.5 4B      │ Q4_K_M    │ 9.33       │ 5.37        │ 4.4990      │ 2.1x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ Qwen3.5 9B      │ Q4_K_M    │ 7.79       │ 4.74        │ 3.9483      │ 2.0x   │
├─────────────────┼───────────┼────────────┼─────────────┼─────────────┼────────┤
│ Qwen3.8 27B     │ UD-Q4_K_M │ 6.39       │ 4.44        │ 5.8820      │ 1.1x   │
└─────────────────┴───────────┴────────────┴─────────────┴─────────────┴────────┘
```

Register and framing are separable. WikiText-2 against raw replies is register
alone and costs gemma E2B 4.7x. Raw replies against chat-framed is framing
alone on identical text.

The effect is a gemma property and it scales with size there: 5.8x and 7.2x at
E4B and E2B, then 32x, 105x and 152x at 12B, 26B and 31B. The 26B is a
mixture-of-experts, so it is not confined to dense models. Qwen sits flat near
2x at every size, and Qwen3.8-27B is the lowest at 1.13x, so this does not
grow with model size across vendors.

Section 1's own rows are 12B-specific and do not generalise to the family.
The E-series barely shows the effect.

## 3. Why ppl ~1.3 is not a perplexity

Scoring a model over ITS OWN greedy replies, chat-framed, reads ~1.29. Low for
a trivial reason: every scored token IS that model's argmax by construction, so
it gets near-1 probability. It is a drift signal, useful for comparing
quantisers, and never a capability number.

This tool therefore uses responses written by a DIFFERENT model, which
separates framing (keep) from self-generation (discard). 7.17 instead of 1.29.


## 4. Why no teacher

The predecessor harness measured KL against a cached bf16 teacher: generate
replies, freeze the ids, dump the teacher's top-64 per position, score every
student against that dump. It works, and KL is far more sensitive than any rank
statistic -- measured, two quantisations of one 12B separated 1470x on KL
against 26-30x on tau/top1/top5, because KL is probability-weighted where rank
statistics saturate.

But it needs a reference model and a ~29 MB dump per corpus, and llama.cpp's
equivalent (`--kl-divergence-base`) writes FULL logits: at gemma's 262144
vocabulary that is ~525 KB per token, three orders of magnitude above a top-64
dump.

A dataset's own responses supply the in-distribution text the teacher was only
ever a means of producing.

That reasoning still sets the DEFAULT, and the default is what the five
headline metrics use. `--kld` adds the teacher back as an OPT-IN mode for the
one question a corpus cannot answer: how far a quantisation moved from the
weights it came from. It keeps the property that made the dump affordable --
top-K plus a lumped tail rather than full logits, ~13 MB against the ~26 GB
that 262144 f32 logits per 24958 rows would cost. Section 9.9 has the
measurements.


## 5. The corpus

The file format is in README.md. `parse_corpus` is deliberately dumb about it
-- a line must EQUAL a sentinel -- so nothing inside prose can be mistaken for
markup.

The marks are not markdown-ish on purpose: across 5.87 MB of the source dataset
`<|@PROMPT@|>` and `<|@RESPONSE@|>` occur 0 times where `---` occurs 313.

Source: the `ifbench_constraints` split of `allenai/IFBench_multi-turn`. That
split plus `ifeval_constraints` hold 2428 unique assistant turns between them;
the 100 shipped items come from `ifbench_constraints` alone. `IFBench_test` is
prompts ONLY, so responses cannot come from it. Full credits and licences in
section 12.

How the 100 were selected, so the file can be regenerated: take the FIRST user
message and the FIRST assistant message of each row, drop rows where either is
empty or contains a sentinel, de-duplicate by prompt text, keep the first 100.
Responses are stored in FULL; the 320-token cap is applied at scoring time by
`--cap`, so changing the cap needs no new corpus.

Rejected: `HuggingFaceH4/no_robots`. Human-written, therefore harder (155 raw),
but CC-BY-NC-4.0.

Residual bias to state whenever a number is quoted: the responses are
model-written, so they carry the style of whichever model wrote them, and a
gemma resembling that model is flattered. Far milder than self-generation, but
not zero. See section 9.6.

Size the corpus to what will be scored. A 5 MB corpus costs parse time for
nothing, and `llama-perplexity` has the same flaw at a worse place --
`perplexity()` tokenizes the whole file before applying `n_chunks`. This tool
tokenizes per item and caps per item, so corpus size costs only parsing.


## 6. Why the template is read from the GGUF, per model, at run time

The E-series and the dense sizes ship DIFFERENT templates. Diffed from Google's
own GGUFs:

            {{- '<|turn>model\n' -}}
        -   {%- if not enable_thinking -%}
        -       {{- '<|channel>thought\n<channel|>' -}}
        -   {%- endif -%}

A 12B prompt ends `<|turn>model\n<|channel>thought\n<channel|>`; an E-series
prompt ends `<|turn>model\n`. A corpus of FROZEN TOKEN IDS can only ever be
correctly framed for one of them.

That trailing block is the NO-THINKING marker, not a dangling channel:
`chat_template.jinja` lines 384-386 emit `<|channel>thought\n<channel|>` when
`enable_thinking` is false, which is its default. An empty, immediately-closed
thought channel.

**TRAP.** `common_chat_templates_inputs::enable_thinking` defaults to TRUE in
llama.cpp while gemma's own jinja defaults it to FALSE. Left at the default the
template opens a thought channel and the framing silently stops matching how
the model is prompted for a direct answer. `render` sets it explicitly.

`--template=<file>` overrides the template, which is how framing is
held FIXED across families. `--chat-template-file` cannot be used: it is
registered for the COMPLETION / CLI / SERVER examples in `common/arg.cpp`, not
for `LLAMA_EXAMPLE_PERPLEXITY`, and is rejected with
`error: invalid argument`. `--load-mode` / `-lm` has no such gating.


## 7. Teacher-free top-k, by counting ranks

`score_row` computes

    rank = #{ i : logits[i] > logits[want] }

so top1 is `rank == 0`, top5 is `rank < 5`, top10 is `rank < 10`, and the whole
rank distribution comes out too -- one O(n_vocab) pass, fused into the scan
that has to find the max for the log-sum-exp anyway. No heap, no sort, no
second pass. Ties count as NOT beating `want`, so the reported rank is the
optimistic one.

This is agreement with the CORPUS ("is the dataset's next token the model's
argmax"), NOT llama.cpp's `Same top p`, which is agreement with a base run and
needs its multi-GB logits file.

It also avoids a dilution trap the teacher-based harness had. Scoring a model
over its OWN output, 58% of positions sat at p(top-1) > 0.9 where every student
necessarily agrees, so `top1 95%` looked near-perfect while the model was in
fact missing 1 in 9 of the genuinely uncertain picks -- measured: ALL top-1
misses, 3 of ours and 84 of google's, fell at p < 0.9. Against a foreign corpus
there is no such floor, so top-1 in the 50-70% range carries real information
instead of being pinned near 100%.

What llama.cpp offers for comparison. Plain `--ppl` prints ONE number,
`Final estimate: PPL = x +/- y`, and no top-1. Everything else lives in
`--kl-divergence` and needs a base run: `PPL(Q)` / `PPL(base)` with their
correlation and ratio, mean KLD plus a full percentile ladder, `Δp` with the
same ladder, `MSE Δp`, `RMS Δp`, and `Same top p`. No top-5, and no rank
distribution in any mode.


## 9. Results

The result tables are in README.md. Conditions for every run: each model's OWN
template, 100 items, cap 320, ctx 8192, one engine, one model resident at a
time. Sections 9.1 to 9.6 read the Google q4_0 QAT set over identical 24958
target tokens; 9.7 adds the post-training Q4_K_M counterparts of two of them
and 9.8 adds a Qwen3.5 ladder. What follows is how to read them and what they
do not say.

### 9.1 What these numbers actually mean

Take one line and read it in plain language:

    E4B   ppl 5.1417   top1 59.81%   top5 85.41%   rank p50 0   p90 8

**top1 59.81%** means: show E4B the prompt and the reply so far, ask "what is
the very next token", and almost exactly six times out of ten the token it
considers most likely is the one the corpus actually has. Four times out of ten
it would have written something else.

**rank p50 0** is the same statement from a different angle. Sort all 262144
tokens in the vocabulary by how likely the model thinks each one is. The median
position of the right answer in that sorted list is 0 -- first place. Half the
time the model's top pick IS the corpus token.

**rank p90 8** says the tail is well behaved too: nine times out of ten the
right answer is somewhere in the model's top nine guesses out of 262144. It is
almost never wildly lost.

**top5 85.41%** fills in between: the right answer is in the top five 85% of
the time.

**ppl 5.1417** is the odd one out, because it is not about ranking at all. It
asks how much PROBABILITY the model put on the right token, not merely whether
it ranked it first. Roughly: the model is about as uncertain as if it were
choosing uniformly among five equally-likely options at each step. Lower is
more confident-and-correct; a model that ranks the right token first but only
gives it 30% still scores a mediocre perplexity.

### 9.2 The 12B is right more often and confident less often

That distinction is not academic here, because the models disagree about who is
better depending on which number you ask.

```
┌─────┬────────┬────────┐
│     │  ppl   │  top1  │
├─────┼────────┼────────┤
│ E4B │ 5.1417 │ 59.81% │
├─────┼────────┼────────┤
│ 12B │ 7.6710 │ 60.23% │
└─────┴────────┴────────┘
```

The 12B picks the corpus token as its single best guess MORE often than E4B
does -- 60.23% against 59.81%. By the ranking measure it is the better model.
Yet its perplexity is nearly 50% worse.

Both can be true at once, and the reason is worth understanding because it is a
real fact about these two models rather than an artifact. The 12B spreads its
probability mass more thinly. It gets the ordering right slightly more often,
but it commits less: where E4B might put 60% on its top choice, the 12B puts
35% and scatters the rest over plausible alternatives. Ranking rewards it;
perplexity punishes it, because perplexity is the exponential of the mean
negative log-probability and therefore cares about how much mass landed on the
right answer, not where it placed.

Put non-numerically: **E4B is more decisive, the 12B is more hedging.** If you
are sampling one token, the 12B's ordering serves you marginally better. If you
care how sharply the model has narrowed things down -- which is what
perplexity measures -- E4B is clearly ahead.

E4B wins four of the five metrics; the 12B wins only top-1. So WHICH METRIC IS
QUOTED CHANGES THE ORDERING, which is exactly why the tool prints all of them
on one line instead of picking. A single scalar would have quietly chosen a
winner on the reader's behalf.

### 9.3 And bigger is not better here

    E4B 5.1417  <  E2B 6.4465  <  26B 7.0977  <  12B 7.6710  <  31B 8.3957

The 31B -- the largest model measured -- is the WORST on every single metric:
highest perplexity, lowest top-1 (below even the small E2B), worst rank p90 at
17. That is not a rounding effect; it is the clearest result in the table.

An earlier reading of the first four rows claimed top-1 was monotonic in
capability. The 31B refutes it, and the refutation is the useful part: see
section 9.6 for what this metric is really measuring, because "predicting this
particular corpus well" and "being a more capable model" are not the same
thing and this table is the proof.

### 9.4 How to read each metric

- `ppl` -- exp of the mean negative log-probability of the corpus token.
  Sensitive to CALIBRATION: how much mass, not just which peak. The only one
  of the five that notices the difference between a confident right answer and
  a barely-preferred one.
- `top1` -- how often the corpus token IS the argmax. Sensitive to RANKING
  only, and indifferent to whether the win was by 0.001 or by 0.9.
- `top5` / `top10` -- the same, relaxed. They track top1 closely and add
  little, the same redundancy the KL harness showed for tau/top1/top5.
- `rank p50` / `p90` -- where the corpus token sits in the model's ordering of
  262144 candidates, at the median and the 90th percentile. The most
  interpretable of the five, and the one `llama-perplexity` does not offer in
  any mode.

A model with a diffuse-but-well-ordered distribution and one with a
sharp-but-occasionally-wrong distribution can trade places depending on which
of these you quote. So a cross-model claim of "better" must name its metric.

### 9.5 The family split, and the 2x2 that says what it is NOT

Group by whether the template emits the empty thought-channel block:

    no  block:  E4B 5.1417   E2B 6.4465
    has block:  26B 7.0977   12B 7.6710   31B 8.3957

The groups do not overlap, which invited the obvious hypothesis: the block
costs perplexity. IT WAS TESTED AND IT IS FALSE.

```
┌─────────────────────┬─────────┬────────┬──────────┬───────┐
│ run                 │   ppl   │  top1  │ rank p90 │       │
├─────────────────────┼─────────┼────────┼──────────┼───────┤
│ 12B, own template   │ 7.6710  │ 60.23% │ 9        │       │
├─────────────────────┼─────────┼────────┼──────────┼───────┤
│ 12B, E4B's template │ 28.2981 │ 48.75% │ 219      │ +269% │
├─────────────────────┼─────────┼────────┼──────────┼───────┤
│ E4B, own template   │ 5.1417  │ 59.81% │ 8        │       │
├─────────────────────┼─────────┼────────┼──────────┼───────┤
│ E4B, 12B's template │ 5.2246  │ 59.86% │ 8        │ +1.6% │
└─────────────────────┴─────────┴────────┴──────────┴───────┘
```

The effect is ASYMMETRIC. Taking the block AWAY from the 12B is catastrophic --
3.69x the perplexity, top-1 down 11.5 points, median rank off 0 and p90 from 9
to 219. GIVING it to E4B costs 1.6%, i.e. nothing. The 12B REQUIRES that block;
its absence is out of distribution. E4B merely tolerates it -- those control
tokens are in its vocabulary and its template carries the `strip_thinking`
macro.

RETRACTED by this: the block does not explain the family split. Under the 12B's
EXACT framing E4B still reads 5.2246 against the 12B's 7.6710, so the E-series
advantage is real and intrinsic to the models, not an artifact of the prompt.
Also retracted: "top-1 is framing-robust". E4B's barely moved (59.81 ->
59.86), the 12B's collapsed (60.23 -> 48.75). Robust to a COMPATIBLE template,
not to any.

CONFIRMED, and the most load-bearing number here: RENDERING EACH MODEL'S OWN
TEMPLATE IS NOT A NICETY. A frozen-id corpus mis-framed for one family yields
ppl 28 where correct framing yields 7.67 -- which reads as a broken MODEL
rather than a broken PROMPT.

### 9.6 What this metric actually measures

Agreement with WHOEVER WROTE THE CORPUS, not capability. The IFBench multi-turn
responses were produced by some specific model. A larger, more capable model
with a style of its own has no reason to be better at predicting another
model's prose; ability to predict a particular author does not scale with
ability. The 31B row is that bias in action, not a claim that the 31B is a weak
model.

Where the instrument IS sound, and what it was built for: comparing QUANTISERS
of ONE model with framing held fixed. Two quantisations of the same 12B, same
template, separated cleanly and in the right direction -- 7.1723 for a
grid-recovering repack against 7.5921 for a re-quantisation, +5.85%.


### 9.7 Quantisation-aware training, measured

The instrument's intended use, from 9.6, is comparing quantisations of one
model with framing held fixed. Below is that comparison on two models, with
corpus, template and the 24958 target tokens all pinned.

**READ 9.9 BEFORE QUOTING THESE RATIOS.** An earlier version of this section
claimed the only variable was HOW the weights were quantised. That is FALSE
and section 12.2 already said so: the Google QAT releases derive from the
`-qat-q4_0-unquantized` checkpoint, while the unsloth Q4_K_M files quantise
the base `-it` release. The two files are different TRAINING RUNS as well as
different quantisers, so the ratios below bound the combined effect and cannot
be attributed to quantisation alone. Section 9.9 measures the two apart with
KL divergence.

```
┌─────────┬────────────┬─────────┬────────┬────────┬────────┬─────┐
│ gemma-4 │   quant    │   ppl   │  top1  │  top5  │ top10  │ p90 │
├─────────┼────────────┼─────────┼────────┼────────┼────────┼─────┤
│ E2B     │ QAT q4_0   │ 6.4465  │ 57.83% │ 83.26% │ 89.02% │ 11  │
├─────────┼────────────┼─────────┼────────┼────────┼────────┼─────┤
│ E2B     │ PTQ Q4_K_M │ 17.2175 │ 56.66% │ 81.30% │ 87.45% │ 14  │
├─────────┼────────────┼─────────┼────────┼────────┼────────┼─────┤
│ E4B     │ QAT q4_0   │ 5.1417  │ 59.81% │ 85.41% │ 91.03% │ 8   │
├─────────┼────────────┼─────────┼────────┼────────┼────────┼─────┤
│ E4B     │ PTQ Q4_K_M │ 9.7072  │ 56.70% │ 81.63% │ 88.05% │ 13  │
└─────────┴────────────┴─────────┴────────┴────────┴────────┴─────┘
```

The 12B was measured at Q4_K_M only -- 8.4623 / 58.37% / p90 14 -- so it is NOT comparable to the 7.6710 QAT row above it; no 12B QAT file was measured on this host.

The PTQ perplexity penalty is 2.67x on E2B and 1.89x on E4B. Two things are
worth extracting.

First, the damage is WORSE ON THE SMALLER MODEL. The E2B has less redundancy
to absorb naive 4-bit rounding, so the same quantiser costs it more.

Second, and more useful: the two KINDS of metric diverge. Top-1 barely moves
(E2B 57.83% -> 56.66%, 1.17 points) while perplexity nearly triples. PTQ
largely preserves WHICH token ranks first and wrecks HOW MUCH PROBABILITY it
gets. That is the calibration-versus-ranking split of section 9.4 appearing
as a property of a quantiser rather than of a model, and it is the single
best argument for printing all five numbers instead of one: a rank-only
metric would have reported this file as nearly unharmed.

The consequence for anyone quoting a number: a perplexity attributed to
"gemma-4 E2B" is meaningless without the quantisation, because on this corpus
that name spans 6.4465 to 17.2175.

What this does NOT say: it is not a claim that Q4_K_M is a bad quantiser in
general. It is one corpus, one metric family, two models, and QAT has the
structural advantage of having seen the quantisation during training.

### 9.8 A family that IS monotonic in size

Section 9.3 recorded that on the gemma QAT set bigger was not better, and the
31B was worst on every metric. That result stands, and section 9.6 explains it
as corpus-author bias rather than capability. But it invited an over-general
reading -- that this measurement never orders models by size -- and one
family, one quantiser and one framing refutes that:

```
┌────────────────┬────────┬────────┬────────┬────────┬─────┐
│ Qwen3.5 Q4_K_M │  ppl   │  top1  │  top5  │ top10  │ p90 │
├────────────────┼────────┼────────┼────────┼────────┼─────┤
│ 0.8B           │ 6.8417 │ 55.51% │ 80.17% │ 86.53% │ 16  │
├────────────────┼────────┼────────┼────────┼────────┼─────┤
│ 2B             │ 5.4274 │ 59.15% │ 83.68% │ 89.55% │ 10  │
├────────────────┼────────┼────────┼────────┼────────┼─────┤
│ 4B             │ 4.4940 │ 62.12% │ 86.73% │ 91.88% │ 7   │
├────────────────┼────────┼────────┼────────┼────────┼─────┤
│ 9B             │ 3.9347 │ 64.39% │ 88.64% │ 93.37% │ 5   │
└────────────────┴────────┴────────┴────────┴────────┴─────┘
```

Perplexity falls and top-1 rises at every step from 0.8B to 9B, and rank p90
tightens 16 -> 10 -> 7 -> 5. Four sizes, five metrics, no inversions.

So the correct statement is narrower than 9.3 alone suggests. WITHIN one
lineage, one quantiser and one framing, the metric orders cleanly and is the
regime it was built for. ACROSS families and training recipes it measures
proximity to the corpus author, and there the ordering carries no claim about
capability. Note also that the Qwen runs score 24960 target tokens against
gemma's 24958: a different tokenizer under the same per-item 320-token cap, so
cross-family perplexities here are not even scored over the same units.


### 9.9 KL divergence

`--kld` scores a student against a reference model's distribution rather than
the corpus. `--kld-save` writes the reference's top-K log-probs per row plus
the tail as one bucket; `--kld` reads it back. Section 4 says why the default
is teacher-free; this is opt-in.

Q8_0 as reference, 24958 rows, AMD host under Vulkan:

```
reference -> student          checkpoint    kld      top1-agree
26B-A4B Q8_0 -> UD-Q4_K_M     same        0.02621     94.62%
26B-A4B Q8_0 -> QAT q4_0      different   0.16684     85.52%
31B Q8_0 -> Q4_K_M            same        0.05355     93.13%
31B Q8_0 -> QAT q4_0          different   0.43735     81.24%
```

The `checkpoint` column is the point. Same-checkpoint rows are quantisation
damage and are small. The QAT rows are a different training run and their KL
is dominated by that, not by the quantiser. Section 9.12 measures the QAT
files against their own BF16 instead, which is the comparison this section
could not make.

### 9.10 The 26B-A4B and 31B, measured

```
┌─────────┬───────────┬─────────┬────────┬────────┬────────┬─────┐
│ gemma-4 │   quant   │   ppl   │  top1  │  top5  │ top10  │ p90 │
├─────────┼───────────┼─────────┼────────┼────────┼────────┼─────┤
│ 26B-A4B │ Q8_0      │ 7.8885  │ 61.43% │ 85.83% │ 90.91% │ 8   │
├─────────┼───────────┼─────────┼────────┼────────┼────────┼─────┤
│ 26B-A4B │ UD-Q4_K_M │ 7.8334  │ 61.40% │ 86.04% │ 90.98% │ 8   │
├─────────┼───────────┼─────────┼────────┼────────┼────────┼─────┤
│ 26B-A4B │ QAT q4_0  │ 7.0651  │ 60.81% │ 85.50% │ 90.72% │ 8   │
├─────────┼───────────┼─────────┼────────┼────────┼────────┼─────┤
│ 31B     │ Q8_0      │ 10.4170 │ 61.24% │ 84.75% │ 89.41% │ 11  │
├─────────┼───────────┼─────────┼────────┼────────┼────────┼─────┤
│ 31B     │ Q4_K_M    │ 10.0953 │ 61.14% │ 84.57% │ 89.31% │ 11  │
├─────────┼───────────┼─────────┼────────┼────────┼────────┼─────┤
│ 31B     │ QAT q4_0  │ 8.3109  │ 57.36% │ 80.27% │ 86.71% │ 16  │
└─────────┴───────────┴─────────┴────────┴────────┴────────┴─────┘
```

Two of these reproduce numbers this document already carried, on a THIRD
engine: 26B-A4B QAT reads 7.0651 against the 7.0977 recorded in section 9 on
Metal (0.46% apart) and 31B QAT reads 8.3109 against 8.3957 (1.01% apart).
Both sit inside the ~1.04% cross-engine band section 10 measures, so Vulkan on
Vulkan joins Metal and the second Metal engine as an agreeing implementation.

The 31B remains last on every RANKING metric, as section 9.3 recorded, and
section 9.6 still explains it as corpus-author bias rather than weakness.

### 9.12 The shipped 4-bit file is a different model, not a lossy copy

Each Google q4_0 release against a BF16 conversion of THE SAME checkpoint, all
five sizes, one engine, 24958 targets.

```
┌─────────┬────────────────┬───────────────┬───────────────────────┬─────────┬──────────────┐
│ gemma-4 │ full precision │ shipped 4-bit │       difference      │    KL   │ argmax agree │
├─────────┼────────────────┼───────────────┼───────────────────────┼─────────┼──────────────┤
│ E2B     │ 6.2730         │ 6.4478        │ 4-bit worse by 2.79%  │ 0.03867 │ 91.53%       │
├─────────┼────────────────┼───────────────┼───────────────────────┼─────────┼──────────────┤
│ E4B     │ 5.1569         │ 5.1457        │ same within noise     │ 0.02504 │ 93.04%       │
├─────────┼────────────────┼───────────────┼───────────────────────┼─────────┼──────────────┤
│ 12B     │ 7.2460         │ 7.5881        │ 4-bit worse by 4.72%  │ 0.02786 │ 93.77%       │
├─────────┼────────────────┼───────────────┼───────────────────────┼─────────┼──────────────┤
│ 26B-A4B │ 7.0570         │ 7.0651        │ same within noise     │ 0.05049 │ 91.73%       │
├─────────┼────────────────┼───────────────┼───────────────────────┼─────────┼──────────────┤
│ 31B     │ 8.5744         │ 8.3109        │ 4-bit BETTER by 3.07% │ 0.03687 │ 92.37%       │
└─────────┴────────────────┴───────────────┴───────────────────────┴─────────┴──────────────┘

```

Calling the difference a quantisation cost assumes BF16 is ground truth. For a
QAT checkpoint it is not: the quantiser runs inside the training forward pass,
so the 4-bit function is the one being optimised and the BF16 master weights
are scaffolding. There is no reason the 4-bit file must be worse, and at two
of five sizes it is not.

The 31B is the clean case: its q4_0 wins on ALL FIVE metrics, which rules out
a single-metric wobble. Which file is better is not predictable from size,
architecture, or the dense-versus-E-series split of 9.5.

KL says they are genuinely different everywhere, 0.025 to 0.050, argmax
disagreeing 6 to 9 percent of the time. It does not predict the direction: the
26B is the most distant by KL and the closest by perplexity.

NOT ESTABLISHED. E4B at -0.22% and 26B at +0.11% are inside the paired-delta
sampling floor, about 0.45 points at 100 items as measured by companion work
on these corpora, so they mean no measurable difference, not identical. The
effective sample size for a paired log-loss delta is the number of ITEMS, not
tokens: per-token loss is strongly correlated within one response.

RULE. Do not treat a vendor's 4-bit release as a degraded copy of its
full-precision weights, and do not infer one from the other. Measure the file
you intend to ship.

### 9.13 Two 4-bit files, one checkpoint, and what WikiText cannot see

Qwen3.8-27B, our own BF16 from Qwen's weights, our own Q4_K_M from that BF16,
against a third party's UD-Q4_K_M of the same checkpoint.

```
                    WikiText-2   IFBench framed   KL vs BF16   argmax agree
BF16 (reference)      6.3530         5.6195           -             -
our Q4_K_M            6.3877         5.2050        0.03194       92.61%
UD-Q4_K_M             6.3857         5.8820        0.01697       94.48%
Q8_0                     -           5.6074        0.00403       96.94%
```

WikiText puts the two 4-bit files 0.03% apart. Chat-framed text puts them 13%
apart. The standard benchmark does not understate that difference, it cannot
see it.

Perplexity and fidelity disagree about which is better. Perplexity prefers
ours; KL says UD is nearly twice as faithful to the weights it came from. Both
are true and they answer different questions. Q8_0 is the control and behaves
as a near-lossless encoding must: lowest KL, highest agreement, ppl nearest
the reference.

CONSEQUENCE FOR KL BENCHMARKS THAT USE Q8_0 AS REFERENCE. That is common
practice: an independent comparison of Qwen3.8-27B quants
(https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/49) scores
UD-IQ2_XXS at ppl 7.65 / mean KLD 0.146 against bartowski IQ2_XXS at ppl 8.54
/ mean KLD 0.301, both measured against a Q8_0 reference, and there both
metrics agree on the winner.

Q8_0's own divergence from BF16 is 0.00403 here. Negligible against 2-bit
quants at 0.146 and 0.301. About a quarter of the signal against a good 4-bit
quant at 0.017. So a Q8_0 reference is fine at 2 bits and starts to matter at
4, which is the regime this section is in.

SIGN IS NOT GUARANTEED. At 27B the 4-bit file beats its own BF16 on all three
chat-framed corpora by 1.7 to 7.4 percent, as the vendor round trips do in
section 9.12. Better on a corpus is not better, only different.

QUANTISATION COST DOES NOT SHOW UP ON RAW TEXT. Both 4-bit files cost about
+0.5% on WikiText while moving several percent on instruction text in either
direction. A reader taking the WikiText number as a proxy would conclude these
quants are free.

## 10. Verification that was actually done

- Two independent tokenizer implementations agree with HF `tokenizers`:
  50/50 on real chat-templated prompts.
- This tool and an independent Python builder produced the SAME scored token
  count, 24958, from the same corpus and cap -- two tokenizer implementations,
  two cap implementations, identical result.
- CROSS-ENGINE: google's 12B codes read 7.5921 on a SECOND, INDEPENDENT
  Metal engine (after re-headering the file into that engine's key spelling)
  and 7.6710 here on llama.cpp, 1.04% apart. Two engines, two tokenizers, two
  jinja implementations, same codes, same answer. That also validates the
  re-headering.
- The rendered framing matches an independently-derived rendering of the same
  prompts, byte for byte.
- Baselines reproduce run to run: 12B 7.6710 and E4B 5.1417 both re-measured
  identically, the latter through a different path to the same file.
- CROSS-HOST, SAME FILE: on a second Apple-silicon host, on llama.cpp
  `b10712` rather than the `b10680` every section-9 number came from, the two
  QAT E-models reproduce to the last printed digit -- E2B 6.4465 / 57.83% /
  83.26% / 89.02% / p50 0 / p90 11 and E4B 5.1417 / 59.81% / 85.41% / 91.03% /
  p50 0 / p90 8, both over 24958 tokens. The HF revisions are the same ones
  pinned in section 12.2, so this is the same bytes on a different machine
  through a different engine build: neither the 32-commit llama.cpp move nor
  the host changed anything measurable.
- REFACTOR A/B: the single-exit rewrite of `score_item` / `run` into
  `score_chunk` / `score_one` / `score_items` / `score_all` was verified by
  re-running two models before and after. Identical on every metric --
  E2B Q4_K_M 17.2175 / 56.66% / 81.30% / 87.45% / p90 14 and E4B Q4_K_M
  9.7072 / 56.70% / 81.63% / 88.05% / p90 13, both 24958 tokens.
- The section 1 raw-text claim was re-derived from scratch rather than
  inherited: `llama-perplexity` built from the pinned submodule, run over
  `wiki.test.raw` and over the corpus responses stripped of framing, nine
  models each. Section 1.1 has the table. It CONFIRMS section 1 on the 12B
  (507.00 PTQ against 308.79 QAT, a 1.64x gap matching the independently
  measured PTQ penalty) and NARROWS it everywhere else.
- THIRD ENGINE: Vulkan agrees with Metal. The 26B-A4B and
  31B QAT files read 7.0651 and 8.3109 there against 7.0977 and 8.3957 here,
  0.46% and 1.01% apart, inside the cross-engine band above.
- TEMPLATE EQUIVALENCE, by rendering rather than by reading the diff: Google's
  own `chat_template.jinja`, unsloth's Q4_K_M template and unsloth's Q8
  template all render a corpus prompt to BYTE-IDENTICAL output (same sha1,
  ending `<|turn>model\n` with no thought-channel block, as section 6 says an
  E-series prompt should). Their diffs are confined to a tool-call branch a
  single user message never reaches. This is what ruled the template out as
  the cause of the QAT-vs-PTQ gap in section 9.7.


## 11. Limits to carry with any number

- A ppl here is NOT comparable to a WikiText ppl. Different corpus, different
  protocol, response-only scoring.
- Cross-family comparisons carry the framing caveat of section 9.5.
- Cross-engine comparisons carry the ~1% offset of section 10.
- Cross-model comparisons are agreement with the corpus author, section 9.6.
- `--cap` bounds each response, so changing the cap changes the number.
- The tied lm_head means `token_embd`'s quant type dominates scoring cost:
  google keeps it at Q6_K where a Q4_0 build of the same checkpoint does not,
  which makes google's file several times slower to score at identical
  vocabulary.
- On a 48 GB host a 31B-class model wires ~22 GB (17.65 GB weights + KV +
  graph), so run ONE at a time. Wired pages cannot be evicted; two such
  processes fail with
  `kIOGPUCommandBufferCallbackErrorOutOfMemory` at an arbitrary item rather
  than a large one. mmap is already on -- `load_mode` defaults to
  `LLAMA_LOAD_MODE_AUTO`, which means mmap -- and it does not reduce the wired
  figure, it only makes the pages file-backed.


## 12. Credits, sources and licences

### 12.1 The corpora

`pplit-corpus.txt` is derived from

- **IFBench (multi-turn)** — https://huggingface.co/datasets/allenai/IFBench_multi-turn
  split `ifbench_constraints`. Ai2 (Allen Institute for AI).
  Licence **ODC-BY-1.0**, intended for research and educational use in
  accordance with Ai2's Responsible Use Guidelines
  (https://allenai.org/responsible-use).

  The dataset card also states: "This dataset includes output data generated
  from third party models that are subject to separate terms governing their
  use." That applies to every assistant turn reproduced here -- the responses
  are MODEL OUTPUT, not human writing, which is also why section 9.6 warns
  that this metric scores agreement with whatever model produced them.

  Citation, as the dataset card gives it:

      @misc{pyatkin2025generalizing,
      }

  The card supplies the key with an empty body; take the canonical entry from
  the dataset page rather than from here.

- **StructFlowBench** — https://huggingface.co/datasets/Jinnan/StructFlowBench
  MIT. Shipped in this repository in two shapes, `structflow-corpus.txt`
  (first turns only) and `structflow-multiturn-corpus.txt` (40 conversations,
  114 turns), so turn depth can be varied with the source text held fixed.
  Responses are model-written; section 9.6 is the caveat that follows.

- **HealthBench Professional** — OpenAI, MIT, arXiv 2604.27470.
  Physician-written reference answers, no AI assistance, specialty-matched.
  RESEARCH AND VALIDATION ONLY, NOT REDISTRIBUTED HERE. Its README asks that
  examples not be posted in plain text and it ships a canary string, which is
  also what makes it valuable: deliberately uncontaminated.
  `corpora/make-healthbench-corpus.py` rebuilds it. Only the 410 single-turn
  rows are used; the other 115 have in-context assistant turns that are model
  output, and the multi-turn path would score them as physician text.

- **SAD (Strategic Argumentative Dialogue)** — arXiv 2601.07423.
  Human-written r/ChangeMyView, crawled 2016-03 to 2020-09, two years before
  ChatGPT, which is what settles provenance. RESEARCH AND VALIDATION ONLY, NOT
  REDISTRIBUTED HERE. The paper restricts use to academic and NON-COMMERCIAL
  research, overriding the MIT LICENSE in its repo, which covers the authors'
  code and not Reddit comments they do not own.
  `corpora/make-sad-corpus.py` rebuilds it, deterministically (chains sorted
  by leaf id before a seed-0 shuffle). Its prompt is SAD's own figure-10
  instruction, not one invented here, because section 6 shows framing alone
  moves this measurement about fourfold.

  Neither was used to train, tune or select anything, and no text from either
  appears in this repository or the paper. Both licences permit research use
  and neither permits redistribution, so a script ships instead of a corpus.

- Prompt-only sibling, NOT used for responses because it has none:
  https://huggingface.co/datasets/allenai/IFBench_test

- Considered and rejected: **no_robots** —
  https://huggingface.co/datasets/HuggingFaceH4/no_robots, CC-BY-NC-4.0.
  Human-written and therefore a harder corpus (155 raw against IFBench's 45),
  kept out for the licence.

### 12.2 The models measured

Google QAT q4_0 GGUF releases at E2B, E4B, 12B, 26B-A4B and 31B, their
`-qat-q4_0-unquantized` checkpoints, third-party GGUF conversions of the base
`-it` releases, and Qwen's own published weights for Qwen3.5 0.8B/2B/4B/9B and
Qwen3.8-27B. Revisions every number came from:

```
┌─────────────────┬──────────┬──────────────────────────────────────────┐
│ model           │  quant   │                 revision                 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 E2B     │ QAT q4_0 │ 675cff42a74c774d6cb76f76d8eacb49b48c9b93 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 E4B     │ QAT q4_0 │ 4b4a2c1d584be7264f87aac328a1bc739ce81b6c │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 12B     │ QAT q4_0 │ 29d097773436b69ff9feafd636ab4cf873786537 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 26B-A4B │ QAT q4_0 │ d1c082be9cf3c8a514acf63b8761f4b41935842e │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 31B     │ QAT q4_0 │ 59dde24573e7e61570dba08b18a2e1fe246955ed │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 E2B     │ Q4_K_M   │ 0314792d7f1f7e229411f620751375812bb9faf2 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 E4B     │ Q4_K_M   │ bfc15c382204943c3a8fff0c750b94ae2364d7a3 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ gemma-4 12B     │ Q4_K_M   │ fc034cfff751157913579611efad8462ac1be606 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ Qwen3.5 0.8B    │ Q4_K_M   │ 6ab461498e2023f6e3c1baea90a8f0fe38ab64d0 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ Qwen3.5 2B      │ Q4_K_M   │ f6d5376be1edb4d416d56da11e5397a961aca8ae │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ Qwen3.5 4B      │ Q4_K_M   │ e87f176479d0855a907a41277aca2f8ee7a09523 │
├─────────────────┼──────────┼──────────────────────────────────────────┤
│ Qwen3.5 9B      │ Q4_K_M   │ 3885219b6810b007914f3a7950a8d1b469d598a5 │
└─────────────────┴──────────┴──────────────────────────────────────────┘
```

Licence note: each Google file records `general.license = apache-2.0` while
`general.license.link` points at the Gemma Terms of Use, which is not
Apache-2.0. Treat the linked terms as governing.

The QAT files derive from the `-qat-q4_0-unquantized` checkpoint, not from the
base `-it` release. That is why sections 9.7 and 9.12 are different
measurements.

### 12.3 Software

- **llama.cpp** — https://github.com/ggml-org/llama.cpp
  MIT License, Copyright (c) 2023-2026 The ggml authors.

  Credit where it is due: everything hard here is llama.cpp's. It provides the GGUF loader, the Metal backend, the tokenizer,
  the batched decode, and the in-tree **full jinja implementation**
  (`common/jinja`, not minja) that renders gemma-4's macro- and
  channel-heavy template -- which is the single reason a per-model framing is
  even possible here. `common/chat.cpp` already carries first-class gemma-4
  support (`common_chat_params_init_gemma4`,
  `COMMON_CHAT_FORMAT_PEG_GEMMA4`). `tools/perplexity/perplexity.cpp` is the
  protocol reference discussed in sections 1 and 7.

  **Tested against two commits:**

      commit  d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b
      tag     b10680
      date    2026-08-28 14:01:59 -0700

      commit  daef7b6874397a5a7c3d7e38b55e2ee0adf7da38
      tag     b10712
      date    2026-08-31 07:04:34 +0200

  Sections 9.1 to 9.6 were produced by `b10680`; sections 9.7 and 9.8, and the
  cross-host reproduction in section 10, by `b10712`. Both built with
  `-DGGML_METAL=ON`. The two E-model QAT baselines are identical to the last
  printed digit across the pair, so nothing in the 32-commit gap is measurable
  here. Two APIs this tool depends on are recent enough to break on an older
  tree: `llama_load_mode` (which replaced the boolean `use_mmap`) and
  `ggml_set_abort_callback`.

  llama.cpp is now a shallow submodule at `llama.cpp/`, pinned to a commit and
  driven by this repo's own top-level `CMakeLists.txt` via `add_subdirectory`.
  No upstream source file is touched, and nothing is copied into its tree.
- **HF `tokenizers`** — used only to cross-check tokenisation while building
  the corpus, not at scoring time.

For scale, `llama-perplexity` also carries hellaswag,
winogrande, multiple-choice, strided ppl and KL-divergence-vs-base. None of
that is needed to answer "how well does this model predict instruction
responses.

Numbers in this document were measured on one Apple-silicon host, 48 GB
unified memory, Metal backend, one model at a time (section 11).
