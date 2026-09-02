# pplit

Perplexity for instruction-trained models.

`llama-perplexity` scores raw prose. An instruction-tuned model reads raw prose
as something to answer rather than continue, so the number it gives back is not
about the model's quality. gemma-4-12B scores **507** on WikiText-2 and **8.46**
on the same kind of text once that text is wrapped in the model's own chat
template.

pplit does the wrapping. It reads prompt/response pairs, renders each one
through the jinja chat template stored in that model's own GGUF, and scores
only the response tokens.

```
$ pplit -m gemma-4-E4B_q4_0-it.gguf -f pplit-corpus.txt -c 8192 -ngl 99
gemma-4-E4B_q4_0-it.gguf   ppl 5.1417  top1 59.81%  top5 85.41%  top10 91.03%
                           rank p50 0 p90 8   over 24958 tokens, 100 items
```

One corpus file works for every model family, because the framing is resolved
per model at run time instead of being baked into the file. No teacher model,
no reference logits file, no KL dump needed.

The name is `ppl`, the usual abbreviation for perplexity, plus `-it`, the
suffix vendors put on instruction-trained checkpoints.

## What goes wrong without it

`llama-perplexity` cannot measure an instruction-trained model. Its protocol
tokenizes a text file, splits it into `n_ctx` chunks and scores the second half
of each. That is sound and standard, and useless here: an `-it` model reads
raw prose as something to **answer**, not to continue.

Thirteen models, three protocols, one host and one backend. **WikiText-2** is the
standard perplexity benchmark, encyclopedia prose. **raw replies** is this
repo's own corpus responses with the chat framing stripped, so the register is
right and only the framing is missing. **chat-framed** is pplit. All
`llama-perplexity` runs are `-c 512 --chunks 60`.

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

Read the last column, which is what the standard benchmark costs you.

For eight of these nine models it overstates perplexity by 2x to 8x. That is
bad but survivable, you could learn to read it as a biased ruler.

For the dense **gemma-4 12B it is 60x out**: 507.00 against 8.46. A number
like that does not read as "wrong protocol", it reads as a **broken model**,
which is exactly the failure that makes the standard tool unusable rather than
merely inaccurate.

Two separate effects stack up, and the table separates them:

- **Register.** WikiText-2 against raw replies, both unframed. Encyclopedia
  prose costs gemma E2B 4.7x over assistant prose. The corpus you feed it
  matters before any framing question.
- **Framing.** Raw replies against chat-framed, same text both times. This is
  worth 1.2x-1.6x on almost everything, and **9.3x on the dense 12B**.

That second row is the one that matters. The gemma dense line *requires* its
chat frame; the E-series merely benefits from it. OKF.md §9.5 found the same
split from the other direction: stripping the 12B's thought-channel block
costs it 3.69x, while giving that block to E4B costs 1.6%. Framing dependence
is a property of the model family, not of instruction-training in general.

So a raw-text tool has nowhere to put bos, the turn markers, the generation
prompt, or a scoring window that starts at the reply, and for at least one
model family that omission is the difference between 8 and 507. OKF.md §1 and
§2 carry the register ladder across seven corpora and the two approaches that
were tried and refuted.


## Build

llama.cpp is a shallow submodule pinned to the tip of its `master` branch, and
the build drives it directly, nothing is copied into the llama.cpp tree.

```sh
git clone --recurse-submodules https://github.com/leok7v/pplit
cd pplit
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target pplit -j 8
```

Already cloned without submodules:

```sh
git submodule update --init --depth 1
```

To move the submodule to the current upstream tip:

```sh
git submodule update --remote --merge
```

Only `libllama` and `llama-common` are built; llama.cpp's own tools, examples,
server and tests are switched off.


## Use

```sh
./build/pplit -m model.gguf -f pplit-corpus.txt -c 8192 -ngl 99
```

`-c` must exceed the longest prompt+response in the corpus. The bundled corpus
tops out around 3200 tokens, so `-c 4096` is enough; items that do not fit are
skipped with a warning and the surviving count is printed.

```
┌─────────────────┬─────────┬──────────────────────────────────────────┐
│ flag            │ default │               what it does               │
├─────────────────┼─────────┼──────────────────────────────────────────┤
│ --cap N         │ 320     │ max response tokens scored per turn      │
├─────────────────┼─────────┼──────────────────────────────────────────┤
│ --items N       │ 0 (all) │ max conversations, for a quick smoke run │
├─────────────────┼─────────┼──────────────────────────────────────────┤
│ --template FILE │ unset   │ override the chat template               │
├─────────────────┼─────────┼──────────────────────────────────────────┤
│ --kld-save FILE │ unset   │ write a reference distribution dump      │
├─────────────────┼─────────┼──────────────────────────────────────────┤
│ --kld FILE      │ unset   │ score against a reference dump           │
├─────────────────┼─────────┼──────────────────────────────────────────┤
│ --kld-top N     │ 64      │ entries stored per row in the dump       │
└─────────────────┴─────────┴──────────────────────────────────────────┘

```

`--items 35` gets within ~0.02 of the 100-item perplexity in a third of the
time. Everything llama.cpp understands still works: `-m`, `-f`, `-c`, `-ngl`,
`-t` and the rest are passed straight through.

### Checking a backend before you trust it

`--selftest` decodes one prompt twice, once whole and once split into
micro-batches, and compares the two distributions. A correct backend returns
the same answer either way.

```sh
./build/pplit -m model.gguf -f pplit-corpus.txt -ngl 99 --selftest
```

```
selftest  640 tokens, n_ubatch 512 against 640, 40 rows sampled
selftest  argmax differs 0/40, mean KL 0.00000002, worst KL 0.00000026
selftest  PASS: splitting the batch does not change the distribution
```

Exit status is 0 on pass and 2 on fail, so it can gate a script. Run it once
on any new machine, GPU or backend build. A backend that fails this will still
produce fluent text and plausible-looking perplexity, so nothing else in the
output will warn you.

### Comparing two models directly

The five headline metrics are teacher-free. `--kld` is the opt-in exception:
it answers "how far did this quantisation move from the weights it came from",
which no corpus-agreement metric can. Two passes, reference first:

```sh
./build/pplit -m reference.gguf -f pplit-corpus.txt -ngl 99 --kld-save ref.kld
./build/pplit -m candidate.gguf -f pplit-corpus.txt -ngl 99 --kld ref.kld
```

The second run prints its usual line plus:

```
kld  0.02621  median  0.00346  p90  0.05737  p99  0.32053  top1-agree 94.62%
```

`--kld-top` sets K (default 64). The dump stores the reference's top-K
log-probs per scored row plus the tail as a single lumped bucket, which makes
the reported KL a lower bound on the true value that tightens as K grows. It
is ~13 MB for the bundled corpus; storing all 262144 logits per row would be
~26 GB.

The header records vocabulary and row count, so a student that tokenizes
differently, or skips a different set of items, is reported as a row mismatch
rather than silently compared against misaligned rows. **KL is only meaningful
between models sharing a vocabulary**, and only interpretable as quantisation
damage when both files come from the same checkpoint.


## Reading the output

```
┌──────────┬────────────────────────────────────────────────────────────────────────┐
│ field    │                                meaning                                 │
├──────────┼────────────────────────────────────────────────────────────────────────┤
│ ppl      │ how much probability the model put on the right token; lower is better │
├──────────┼────────────────────────────────────────────────────────────────────────┤
│ top1     │ how often the corpus token was the model's single best guess           │
├──────────┼────────────────────────────────────────────────────────────────────────┤
│ top5     │ how often it was among the top 5                                       │
├──────────┼────────────────────────────────────────────────────────────────────────┤
│ top10    │ how often it was among the top 10                                      │
├──────────┼────────────────────────────────────────────────────────────────────────┤
│ rank p50 │ median position of the right token in the model's ranking              │
├──────────┼────────────────────────────────────────────────────────────────────────┤
│ rank p90 │ 90th-percentile position of the right token                            │
└──────────┴────────────────────────────────────────────────────────────────────────┘
```

### What p90 actually means, in plain English

Imagine the model is asked "what word comes next?" and, instead of answering
once, it lines up **every word it knows**, 262,144 of them for gemma-4,
248,320 for Qwen3.5, in order of how likely it thinks each one is.

Now check where the word the corpus *actually* used ended up in that queue.

- `rank p50 0`, half the time, the right word was at the very front. The
  model's first guess was correct.
- `p90 5`, nine times out of ten, the right word was somewhere in the first
  five. Only in the remaining tenth was it further back than that.

So **p90 is a "how bad does it get" number**, not an average. It deliberately
ignores the easy cases and reports how far down the list the answer sits when
the model is having a hard time. A p90 of 5 means the model is almost never
badly lost. A p90 of 16 means that one time in ten it has the right answer
buried well down its list. Lower is better.

`ppl` is the odd one out of the five, because it is not about position at all.
It asks how much **probability** the model committed to the right word, not
merely whether it ranked it first. A model can rank the right word first and
still score a mediocre perplexity by giving it only 30% and scattering the rest
, confident-and-correct scores better than barely-preferred-and-correct.

All five are **teacher-free**: they measure agreement with the corpus, not with
a reference model, so nothing needs to be run twice.


## The corpus

`pplit-corpus.txt`, 100 prompt/response pairs, 329 KB.

```
<|@PROMPT@|>
why might a company initiate a price drop on a product
<|@RESPONSE@|>
A company might initiate a price drop for several strategic reasons...
<|@PROMPT@|>
...
```

Sentinels alone on their line; `#` lines before the first sentinel are
comments. Bring your own corpus by following the same shape.

Responses come from [IFBench multi-turn](https://huggingface.co/datasets/allenai/IFBench_multi-turn)
(Ai2, ODC-BY-1.0). They are *model-written*, which is deliberate: text no model
under test produced, so the number is a real perplexity rather than the ~1.3 a
model scores against its own output.

### Multi-turn

A `<|@CONV@|>` line starts a conversation, and every prompt/response pair after
it is one turn of it. Turn *t* is scored with turns 0..t-1 as history plus the
generation prompt, so later turns test whether the model still tracks context.
A file with no `<|@CONV@|>` behaves exactly as before, every pair its own
single-turn conversation.

`structflow-corpus.txt` and `structflow-multiturn-corpus.txt` are the same
source in both shapes, from
[StructFlowBench](https://huggingface.co/datasets/Jinnan/StructFlowBench) (MIT),
so turn depth can be varied with the text held fixed.

### Two corpora that ship as scripts, not text

Both corpora below are **human-written**, which matters because everything
above is model-written and the metric partly measures closeness to the response
author's style. Neither can be redistributed, so `corpora/` holds a script that
rebuilds each one from its canonical source.

**Both were used here solely for research and validation**, to check whether
results obtained on model-written corpora also hold on human-written text. No
text from either is reproduced in this repository or in the paper, and neither
was used to train, tune or select anything. The scripts check for sentinel
collisions and preserve each dataset's canary string.

`corpora/make-healthbench-corpus.py` builds from **HealthBench Professional**
(OpenAI, MIT, [arXiv 2604.27470](https://arxiv.org/abs/2604.27470)), whose
reference answers were written by physicians without AI assistance. Its README
asks that examples not be posted in plain text online, and it ships a canary
string so trainers can exclude it; that request is also what makes it valuable,
since it is deliberately uncontaminated. The script keeps only the 410
single-turn rows, because the in-context assistant turns of the rest are
model output rather than physician text.

`corpora/make-sad-corpus.py` builds from **SAD**, Strategic Argumentative
Dialogue ([arXiv 2601.07423](https://arxiv.org/abs/2601.07423)), human-written
r/ChangeMyView debate crawled 2016-03 to 2020-09, two years before ChatGPT,
which is what settles its provenance. Its paper restricts use to academic and
**non-commercial research**, which overrides the MIT file in its repository:
that file covers the authors' code, not Reddit comments they do not own. The
prompt is SAD's own generation instruction from the paper rather than one of
ours, because framing alone moves this measurement severalfold.


## Results

All runs: 100 conversations, cap 320, `-c 8192 -ngl 99`, Release build, one
model resident at a time, each model rendered through its **own** chat
template. Two hosts, one Metal and one AMD; the AMD host runs Vulkan.

### gemma-4

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
├─────────┼────────────┼─────────┼────────┼────────┼────────┼─────┤
│ 12B     │ PTQ Q4_K_M │ 8.4623  │ 58.37% │ 82.55% │ 87.79% │ 14  │
└─────────┴────────────┴─────────┴────────┴────────┴────────┴─────┘
```

### Qwen3.5

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

Within one family, one quantiser and one framing, the Qwen ladder is ordered
on every metric, perplexity falls and top-1 rises monotonically from 0.8B to
9B. That is the regime this instrument is sound in.

### The 4-bit file is a different model, not a lossy copy

Google ships two things for each gemma-4 size: a 4-bit file you can run, and
the full-precision weights it came from. The natural assumption is that the
4-bit file is a slightly damaged copy of the full-precision one.

That assumption is backwards for these models. Quantisation-aware training
runs the 4-bit conversion *inside* the training loop, so the 4-bit version is
the one that was actually being optimised. The full-precision weights are
left-over scaffolding, not a version anyone intended you to run.

Measured at all five sizes, scoring each shipped 4-bit file against a BF16
conversion of the very same checkpoint:

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

The 4-bit file is worse at two sizes, better at one, and indistinguishable at
two. The spread runs from 4.72% worse to 3.07% better, and nothing about a
model's size or design tells you where it will land. The 31B is the clearest
case: its 4-bit file does not merely edge ahead, it wins on all five metrics
at once, which is not what rounding error looks like.

The KL column shows these really are different models everywhere. The two
files disagree about the single most likely next word 6 to 9 percent of the
time. But KL does not predict the direction either: the 26B pair is the most
different by KL and the closest by perplexity.

**What to take from this.** Do not treat a vendor's 4-bit release as a
degraded version of its full-precision weights, and do not use one to predict
the other. They are different models. Measure the one you actually intend to
ship.

### Third-party quantisations of the same two models

The rows above use Google's own files throughout. For contrast, here are
third-party quantisations of the base instruction-tuned 26B and 31B, scored
against a Q8_0 of the same third-party conversion so the checkpoint is held
fixed. Q8_0 is not a true full-precision reference, but it is close enough
that these read as ordinary post-training quantisation costs:

```
┌─────────┬──────────┬─────────┬────────────────────┬─────────┬──────────────┐
│ gemma-4 │ Q8_0 ref │  Q4_K_M │     difference     │    KL   │ argmax agree │
├─────────┼──────────┼─────────┼────────────────────┼─────────┼──────────────┤
│ 26B-A4B │ 7.8885   │ 7.8334  │ Q4 better by 0.70% │ 0.02621 │ 94.62%       │
├─────────┼──────────┼─────────┼────────────────────┼─────────┼──────────────┤
│ 31B     │ 10.4170  │ 10.0953 │ Q4 better by 3.09% │ 0.05355 │ 93.13%       │
└─────────┴──────────┴─────────┴────────────────────┴─────────┴──────────────┘

```

Both come out slightly *better* than their Q8_0 reference, which is the same
pattern the Qwen ladder shows at smaller sizes and at larger ones than any
model in it. Note these are not comparable to the vendor table above: a
different checkpoint, a different quantiser, and a Q8_0 rather than a BF16
reference.

### Quantisation-aware training predicts this corpus better

`E2B` and `E4B` were each measured twice: once from Google's **QAT** release,
once from a **post-training** Q4_K_M conversion. Framing, corpus and token
count are identical, 24958 target tokens in all four runs.

```
┌─────────┬──────────┬────────────┬─────────────┐
│ gemma-4 │ QAT q4_0 │ PTQ Q4_K_M │ ppl penalty │
├─────────┼──────────┼────────────┼─────────────┤
│ E2B     │ 6.4465   │ 17.2175    │ 2.67x       │
├─────────┼──────────┼────────────┼─────────────┤
│ E4B     │ 5.1417   │ 9.7072     │ 1.89x       │
└─────────┴──────────┴────────────┴─────────────┘
```

Note what happens to the two kinds of metric: top-1 barely moves
(E2B 57.83% to 56.66%) while perplexity nearly triples, the
calibration-versus-ranking split described in OKF.md §9.4, and why this tool
prints all five numbers instead of one.

**These two files are not the same checkpoint.** Google's QAT releases derive
from the `-qat-q4_0-unquantized` weights, while the third-party files quantise
the base `-it` release, so the ratio bounds the *combined* effect of a
different training run and a different quantiser. The vendor round-trip table
above isolates the encoding with the checkpoint held fixed, and finds it worth
at most a few percent in either direction. Treat 2.67x as "the QAT release
predicts this corpus better", not as "QAT recovers 2.67x of quantisation
loss".

The practical consequence: a perplexity quoted for "gemma-4 E2B" is meaningless
without the quantisation, because on this corpus that name spans 6.4465 to
17.2175.

### Models measured

gemma-4, Google QAT q4_0:

- [google/gemma-4-E2B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf)
- [google/gemma-4-E4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf)

gemma-4, Google QAT q4_0, large:

- [google/gemma-4-26B-A4B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)
- [google/gemma-4-31B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf)

gemma-4, unsloth Q8_0 and Q4_K_M:

- [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF)
- [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
- [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [unsloth/gemma-4-31B-it-GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)

Qwen3.5, unsloth Q4_K_M:

- [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)

### What these numbers do not say

Comparing **across families** is the one thing this measurement is weakest at.
It scores agreement with whoever wrote the corpus, and the corpus responses
were produced by some specific model, so a model resembling that author is
flattered. Use it to compare **quantisations or sizes within one lineage**,
where it is precise and correctly ordered. OKF.md §9.6 and §11 give the full
list of caveats to carry with any quoted number.


## Files

```
┌──────────────────┬───────────────────────────────────────────────────────────┐
│ file             │                         what it is                        │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ pplit.cpp        │ the tool                                                  │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ pplit-corpus.txt │ the corpus, 100 prompt/response pairs                     │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ OKF.md           │ why it exists, what was refuted, results, limits, credits │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ CMakeLists.txt   │ one target                                                │
├──────────────────┼───────────────────────────────────────────────────────────┤
│ llama.cpp/       │ shallow submodule, tip of master                          │
└──────────────────┴───────────────────────────────────────────────────────────┘
```

Comments live in OKF.md rather than in the source, by house rule. If you are
about to change the code, read OKF.md §6 (why the template is per-model) and
§8 (what each function does and where it stops) first.


## Credits

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, © 2023-2026 The
  ggml authors. Everything hard is theirs: GGUF loading, Metal, the tokenizer,
  batched decode, and the in-tree jinja implementation that makes per-model
  chat templating possible.
- [IFBench](https://huggingface.co/datasets/allenai/IFBench_multi-turn), Ai2,
  ODC-BY-1.0. Includes output from third-party models subject to separate
  terms.
- gemma-4, Google, governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/docs/gemma_4_license).
- Qwen3.5, Alibaba, Apache-2.0.

pplit itself is MIT; see LICENSE. Full credits, licences, pinned revisions and
citation in OKF.md §12.
