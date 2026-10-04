import numpy as np
import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state, sequences


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    K = min(8 * r, a.Kmax)
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    e = m.embed(x)
    s = initial_state(m, e)
    S, losses, lens = [], [], []
    for _ in range(K):
        s = m.step(s, e)
        S.append(s.float())
        losses.append(float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten())))
        lens.append(float(F.cross_entropy(m.head(s).float().flatten(0, 1), y.flatten())))
    sr = S[r - 1]
    scale = float(sr.norm(dim=-1).mean())
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = D[j, i] = float((S[i] - S[j]).norm(dim=-1).mean()) / scale
    cos = [float(F.cosine_similarity(S[k], sr, dim=-1).mean()) for k in range(K)]
    return {"r": r, "K": K, "loss_coda": losses, "loss_lens": lens, "D": np.round(D, 5).tolist(), "cos_r": cos,
            "scale": scale}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--Kmax", type=int, default=128)
    a = ap.parse_args()
    for_each_run(a, "state_distance.json", lambda m, cfg, dev: measure(m, cfg, dev, a))


if __name__ == "__main__":
    main()
