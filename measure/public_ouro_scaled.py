import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from measure.common import device
from measure.public_ouro import load_sequences, make_step


@torch.no_grad()
def trajectory(step, head, e, y, K, gamma, k_trained=0):
    s, losses = e, []
    for k in range(K):
        nxt = step(s)
        s = nxt if (gamma == 1.0 or k < k_trained) else s + gamma * (nxt - s)
        losses.append(float(F.cross_entropy(head(s).float().flatten(0, 1), y.flatten())))
    return losses


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local folder or hub id of an Ouro model")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/public")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    dev = device()
    t0 = time.time()
    m, x, y = load_sequences(a.model, a.data, a.seqs, a.T, dev)
    r = int(getattr(m.config, "total_ut_steps", 4))
    step = make_step(m, x.shape[1], dev)
    e = m.model.embed_tokens(x).float()
    hidden = m.model(input_ids=x, use_cache=False)[1]
    s, check = e, []
    for h in hidden:
        s = step(s)
        check.append(float((s - h.float()).norm() / h.float().norm()))
    res = {"r": r, "check_vs_forward": check, "trained": trajectory(step, m.lm_head, e, y, 32, 1.0)}
    for rp in (6, 8, 16):
        res["scaled_rp%d" % rp] = trajectory(step, m.lm_head, e, y, max(3 * rp, 24), r / rp)
    res["scaled1_rp8"] = trajectory(step, m.lm_head, e, y, 24, r / 8, k_trained=1)
    name = os.path.basename(a.model.rstrip("/"))
    res.update(model=name, secs=time.time() - t0, seqs=a.seqs, T=a.T)
    dst = os.path.join(a.out, name, "ouro_scaled.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, "w"))


if __name__ == "__main__":
    main()
