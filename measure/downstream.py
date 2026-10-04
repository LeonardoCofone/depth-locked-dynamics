import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from measure.common import STATE_SEED, base_parser, for_each_run

TASKS = ("lambada", "hellaswag", "arc_easy", "piqa")


def build_requests(tok, path, T):
    examples = [json.loads(l) for l in open(path, encoding="utf-8")]
    requests = []
    for i, ex in enumerate(examples):
        ctx = tok.encode(ex["ctx"]).ids
        for j, cont in enumerate(ex["conts"]):
            ids = tok.encode(cont).ids
            requests.append({"ex": i, "j": j, "ids": (ctx + ids)[-T:], "n": len(ids),
                             "bytes": len(cont.encode("utf-8"))})
    return examples, requests


@torch.no_grad()
def score_all(m, requests, K, gamma, dev, bs):
    logprob = np.zeros((len(requests), K), dtype=np.float32)
    greedy = np.zeros((len(requests), K), dtype=bool)
    order = sorted(range(len(requests)), key=lambda i: len(requests[i]["ids"]))
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        x = torch.zeros((len(idx), max(len(requests[i]["ids"]) for i in idx)), dtype=torch.long)
        for row, i in enumerate(idx):
            x[row, :len(requests[i]["ids"])] = torch.tensor(requests[i]["ids"])
        x = x.to(dev)
        e = m.embed(x)
        s = torch.cat([m.init_state(e[row:row + 1], torch.Generator(device=dev).manual_seed(STATE_SEED))
                       for row in range(len(idx))])
        pos, tgt, own = [], [], []
        for row, i in enumerate(idx):
            ln, n = len(requests[i]["ids"]), requests[i]["n"]
            for p in range(ln - n - 1, ln - 1):
                pos.append((row, p))
                tgt.append(requests[i]["ids"][p + 1])
                own.append(row)
        pr = torch.tensor([p[0] for p in pos], device=dev)
        pc = torch.tensor([p[1] for p in pos], device=dev)
        tg = torch.tensor(tgt, device=dev)
        own_t = torch.tensor(own, device=dev)
        for k in range(K):
            nxt = m.step(s, e)
            s = nxt if gamma == 1.0 else s + gamma * (nxt - s)
            logits = m.head(m.run_blocks(m.coda, "coda", m.C(s))[pr, pc]).float()
            tok_lp = F.log_softmax(logits, -1).gather(1, tg[:, None])[:, 0]
            ok = (logits.argmax(-1) == tg).float()
            sum_lp = torch.zeros(len(idx), device=dev).index_add_(0, own_t, tok_lp)
            n_ok = torch.zeros(len(idx), device=dev).index_add_(0, own_t, ok)
            n_tok = torch.zeros(len(idx), device=dev).index_add_(0, own_t, torch.ones_like(ok))
            logprob[idx, k] = sum_lp.cpu().numpy()
            greedy[idx, k] = (n_ok == n_tok).cpu().numpy()
    return logprob, greedy


def accuracies(examples, requests, logprob, greedy):
    K = logprob.shape[1]
    by_example = {}
    for q, r in enumerate(requests):
        by_example.setdefault(r["ex"], []).append(q)
    hits = {t: np.zeros(K) for t in TASKS}
    count = {t: 0 for t in TASKS}
    for i, ex in enumerate(examples):
        t, qs = ex["task"], by_example[i]
        count[t] += 1
        if t == "lambada":
            hits[t] += greedy[qs[0]]
        else:
            norm = logprob[qs] / np.array([requests[q]["bytes"] for q in qs])[:, None]
            hits[t] += (norm.argmax(0) == ex["label"])
    acc = {t: (100 * hits[t] / count[t]).tolist() for t in TASKS}
    acc["mean"] = np.mean([acc[t] for t in TASKS], 0).tolist()
    return acc, count


def main():
    ap = base_parser()
    ap.add_argument("--tasks", default="data/downstream_tasks.jsonl", help="written by measure.prepare_downstream")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--min_r", type=int, default=4)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    examples, requests = build_requests(tok, a.tasks, a.T)

    def measure(m, cfg, dev):
        r, fixed = cfg["mean_rec"], cfg["sampling"] == "fixed"
        K = 3 * r if fixed else 4 * r
        res = {"r": r, "fixed": fixed, "K": K, "T": a.T}
        res["trained"], res["n"] = accuracies(examples, requests, *score_all(m, requests, K, 1.0, dev, a.bs))
        if fixed:
            res["scaled_half"], _ = accuracies(examples, requests, *score_all(m, requests, K, 0.5, dev, a.bs))
        return res

    for_each_run(a, "downstream.json", measure,
                 keep=lambda cfg, name: cfg["sampling"] != "fixed" or cfg["mean_rec"] >= a.min_r)


if __name__ == "__main__":
    main()
