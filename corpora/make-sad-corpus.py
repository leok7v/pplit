#!/usr/bin/env python3
"""Build a gemma-perplexity corpus from SAD (Strategic Argumentative Dialogue).

SAD is human-written r/ChangeMyView debate, crawled 2016-03..2020-09 -- two
years before ChatGPT, which is what settles its provenance.  Licence: the
paper's "Data Release and Intended Use" says academic and NON-COMMERCIAL
research only, which overrides the repo's MIT LICENSE file.  Do not
redistribute the text; rebuild it.

    sent.json      https://drive.usercontent.google.com/download?id=10iw-LgkZCj9ipfCE4wbcu24mdVoMgVLs&export=download&confirm=t
    argument.json  https://drive.usercontent.google.com/download?id=1moEtRwouy7ruBNmXeM9tdqWVyKHBuc2v&export=download&confirm=t

    usage: make-sad-corpus.py sent.json argument.json out.txt [n_items]

The prompt is SAD's OWN generation instruction (paper figure 10, no-strategy
variant), not one of ours: framing swings this measurement by 4x, so the
template must come from the corpus authors.  Speaker labels come from the
`attitude` field -- paper 3.2, 1 = supports the topic, 0 = opposes.
"""
import json, random, sys

SENTINELS = {"<|@PROMPT@|>", "<|@RESPONSE@|>", "<|@CONV@|>"}
MAX_PROMPT_CHARS = 12000
SEED = 0

TEMPLATE = """# Topic
%s

# Debate History
%s

You are an expert of debate, based on this #Topic and #Debate History, please \
conduct a discussion. Your responses will be used for research purposes only, \
so please have a definite reply."""


def build(sent, args, n_items):
    ok = [x for x in args
          if len(x["context"]) >= 3
          and len(x["context"]) == len(x["attitude"])
          and all(c in sent for c in x["context"])]
    ok.sort(key=lambda x: x["context"][-1])
    random.Random(SEED).shuffle(ok)
    items, skipped = [], 0
    for x in ok:
        ids, att = x["context"][:-1], x["attitude"][:-1]
        hist = "\n".join(
            "%s:%s" % ("support" if a else "oppose", sent[c].strip())
            for c, a in zip(ids, att))
        prompt = TEMPLATE % (x["topic"].strip(), hist)
        response = sent[x["context"][-1]].strip()
        if len(prompt) > MAX_PROMPT_CHARS or not response:
            skipped += 1
            continue
        if any(ln.strip() in SENTINELS
               for t in (prompt, response) for ln in t.split("\n")):
            skipped += 1
            continue
        items.append((prompt, response))
        if len(items) == n_items:
            break
    return items, skipped, len(ok)


def main(sent_p, args_p, dst, n_items):
    sent = json.load(open(sent_p))
    args = json.load(open(args_p))
    items, skipped, pool = build(sent, args, n_items)
    out = ["# SAD (Strategic Argumentative Dialogue) -- HUMAN-written",
           "# r/ChangeMyView debate, crawled 2016-03..2020-09.",
           "# arXiv 2601.07423. Licence: NON-COMMERCIAL research only per the",
           "# paper, which overrides the repo's MIT file. DO NOT REDISTRIBUTE.",
           "# %d items, seed %d, from a pool of %d chains (>=3 turns, ids"
           % (len(items), SEED, pool),
           "# resolvable); %d candidates skipped as over %d prompt chars."
           % (skipped, MAX_PROMPT_CHARS),
           "# Prompt is SAD's own figure-10 generation instruction."]
    for prompt, response in items:
        out += ["<|@PROMPT@|>", prompt, "<|@RESPONSE@|>", response]
    open(dst, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d items, %d skipped, pool %d"
          % (dst, len(items), skipped, pool))


main(sys.argv[1], sys.argv[2], sys.argv[3],
     int(sys.argv[4]) if len(sys.argv) > 4 else 400)
