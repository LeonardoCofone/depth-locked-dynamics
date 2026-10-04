import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from loopedlm.checkpoint import load_model
from measure.common import device, initial_state, sequences

BASE = {"8s0": "fixed_r8_b8_s0", "8s1": "fixed_r8_b8_s1", "8s2": "fixed_r8_b8_s2", "12": "fixed_r12_b12_s0",
        "16": "fixed_r16_b16_s0", "4s0": "fixed_r4_b4_s0"}
BASE_CROSS = [("8s0", "12"), ("8s0", "16"), ("12", "16"), ("8s1", "16"), ("8s2", "16"), ("4s0", "8s0")]
BASE_SAME = [("8s0", "8s1"), ("8s0", "8s2"), ("8s1", "8s2")]
DEPTHS = (6, 8, 10, 12, 16, 20, 24)
GRID = {"%ds%d" % (r, sd): "fixed_r%d_b%d_s%d" % (r, r, sd) for r in DEPTHS for sd in (0, 1)}
GRID_CROSS = [("%ds%d" % (a, sd), "%ds%d" % (b, sd)) for sd in (0, 1) for a, b in zip(DEPTHS[:-1], DEPTHS[1:])]
GRID_CROSS += [("8s0", "24s0"), ("6s0", "24s0")]
GRID_SAME = [("%ds0" % r, "%ds1" % r) for r in DEPTHS]
TIMES = (0.25, 0.5, 0.75)


@torch.no_grad()
def collect(m, x, y, r):
    e = m.embed(x)
    s = initial_state(m, e)
    loops = {int(round(t * r)): t for t in TIMES}
    out = {}
    for k in range(r):
        nxt = m.step(s, e)
        if k in loops:
            out[loops[k]] = (s.flatten(0, 1).double().cpu().numpy(), (nxt - s).flatten(0, 1).double().cpu().numpy())
        s = nxt
    return out, float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten()))


def fit_affine(X, Y):
    Xa = np.hstack([X, np.ones((len(X), 1))])
    G = Xa.T @ Xa
    G += 1e-3 * np.trace(G) / X.shape[1] * np.eye(G.shape[0])
    B = np.linalg.solve(G, Xa.T @ Y)
    return B[:-1], B[-1]


def r2(Yhat, Y):
    return float(1 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def cos_rows(A, B):
    return np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30)


def compare(A, B, ra, rb, train, test, rng):
    res = {}
    for t in TIMES:
        Sa, Da = A[t]
        Sb, Db = B[t]
        W, c = fit_affine(Sa[train], Sb[train])
        fa, fb = ra * Da[test] @ W, rb * Db[test]
        cs = cos_rows(fa, fb)
        null = cos_rows(fa[rng.permutation(len(test))], fb)
        ratio = np.linalg.norm(fa, axis=1) / (np.linalg.norm(fb, axis=1) + 1e-30)
        other = {str(t2): float(np.median(cos_rows(fa, rb * B[t2][1][test]))) for t2 in TIMES if t2 != t}
        res[str(t)] = {"R2": r2(Sa[test] @ W + c, Sb[test]), "cos_med": float(np.median(cs)),
                       "cos_q": [float(np.quantile(cs, q)) for q in (.1, .25, .75, .9)],
                       "cos_null": float(np.median(null)), "ratio_med": float(np.median(ratio)),
                       "ratio_raw_med": float(np.median(ratio) * rb / ra), "cos_other_t": other}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/field_alignment")
    ap.add_argument("--set", default="base", choices=["base", "grid"], help="grid = depths 6..24, two seeds")
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    models, cross, same = (BASE, BASE_CROSS, BASE_SAME) if a.set == "base" else (GRID, GRID_CROSS, GRID_SAME)
    dev = device()
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    data, depth, info = {}, {}, {}
    for key, name in models.items():
        run = os.path.join(a.runs, name)
        if not os.path.exists(os.path.join(run, "run_info.json")):
            print("missing", name, flush=True)
            continue
        m, cfg = load_model(run, dev)
        r = cfg["mean_rec"]
        data[key], L_r = collect(m, x, y, r)
        depth[key] = r
        info[key] = {"run": name, "r": r, "L_r": L_r}
        del m
    n = next(iter(data.values()))[TIMES[0]][0].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    train, test = idx[: n // 2], idx[n // 2:]
    res = {"set": a.set, "models": info, "n_tokens": n, "cross": {}, "same": {}}
    for group, pairs in (("cross", cross), ("same", same)):
        for ka, kb in pairs:
            if ka in data and kb in data:
                res[group]["%s->%s" % (ka, kb)] = compare(data[ka], data[kb], depth[ka], depth[kb], train, test, rng)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "field_alignment_%s.json" % a.set), "w"))


if __name__ == "__main__":
    main()
