import argparse
import json
import random
import re

from datasets import load_dataset


def clean_hellaswag(t):
    t = t.strip().replace(" [title]", ". ")
    t = re.sub("\\[.*?\\]", "", t)
    return t.replace("  ", " ")


def sample(d, n, seed=0):
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    return [d[i] for i in sorted(idx[:n])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/downstream_tasks.jsonl")
    ap.add_argument("--n", type=int, default=1000, help="examples per task")
    a = ap.parse_args()
    out = []
    for ex in sample(load_dataset("EleutherAI/lambada_openai", "default", split="test"), a.n):
        ctx, last = ex["text"].rsplit(" ", 1)
        out.append({"task": "lambada", "ctx": ctx, "conts": [" " + last], "label": 0})
    for ex in sample(load_dataset("Rowan/hellaswag", split="validation"), a.n):
        ctx = clean_hellaswag(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
        out.append({"task": "hellaswag", "ctx": ctx, "conts": [" " + clean_hellaswag(e) for e in ex["endings"]],
                    "label": int(ex["label"])})
    for ex in sample(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test"), a.n):
        out.append({"task": "arc_easy", "ctx": "Question: " + ex["question"] + "\nAnswer:",
                    "conts": [" " + t for t in ex["choices"]["text"]],
                    "label": ex["choices"]["label"].index(ex["answerKey"])})
    for ex in sample(load_dataset("baber/piqa", split="validation"), a.n):
        out.append({"task": "piqa", "ctx": "Question: " + ex["goal"] + "\nAnswer:",
                    "conts": [" " + ex["sol1"], " " + ex["sol2"]], "label": int(ex["label"])})
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
