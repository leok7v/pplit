#!/usr/bin/env python3
import json, sys, urllib.request, io, zipfile

URL = ("https://openaipublic.blob.core.windows.net/simple-evals/"
       "healthbench_professional/assets.zip")
SENTINELS = {"<|@PROMPT@|>", "<|@RESPONSE@|>", "<|@CONV@|>"}

def main(dst):
    with urllib.request.urlopen(URL) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = "healthbench_professional_eval.jsonl"
    rows = [json.loads(l) for l in z.read(name).decode().split("\n") if l]
    keep = [x for x in rows if len(x["conversation"]["messages"]) == 1]
    for x in keep:
        for t in (x["conversation"]["messages"][0]["content"],
                  x["physician_response"]):
            for ln in t.split("\n"):
                if ln.strip() in SENTINELS:
                    raise SystemExit("sentinel collision in source text")
    out = [
        "# HealthBench Professional -- physician-written responses.",
        "# OpenAI, MIT. arXiv 2604.27470. %d of %d rows: the %d multi-turn rows"
        % (len(keep), len(rows), len(rows) - len(keep)),
        "# are EXCLUDED because their in-context assistant turns are",
        "# ChatGPT-for-Clinicians output, not physician text.",
        "# DO NOT PUBLISH THIS FILE OR ITS CONTENTS. Canary: %s"
        % keep[0]["canary_string"],
    ]
    for x in keep:
        out += ["<|@PROMPT@|>",
                x["conversation"]["messages"][0]["content"].strip(),
                "<|@RESPONSE@|>",
                x["physician_response"].strip()]
    open(dst, "w").write("\n".join(out) + "\n")
    print("wrote %s: %d items" % (dst, len(keep)))

main(sys.argv[1] if len(sys.argv) > 1 else "healthbench-corpus.txt")
