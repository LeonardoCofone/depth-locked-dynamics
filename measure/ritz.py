import numpy as np
import torch

from measure.common import base_parser, for_each_run, initial_state, jvp, sequences


@torch.no_grad()
def arnoldi(f, s, m, gen):
    n = s.numel()
    Q = torch.zeros(n, m + 1, dtype=s.dtype, device=s.device)
    H = torch.zeros(m + 1, m, dtype=s.dtype, device=s.device)
    q = torch.randn(n, dtype=s.dtype, device=s.device, generator=gen)
    Q[:, 0] = q / q.norm()
    for j in range(m):
        w = jvp(f, s, Q[:, j].view(s.shape)).reshape(-1)
        for _ in range(2):
            h = Q[:, :j + 1].T @ w
            w = w - Q[:, :j + 1] @ h
            H[:j + 1, j] += h
        H[j + 1, j] = w.norm()
        if H[j + 1, j] < 1e-12:
            m = j + 1
            break
        Q[:, j + 1] = w / H[j + 1, j]
    ev, V = np.linalg.eig(H[:m, :m].cpu().numpy())
    res = np.abs(float(H[m, m - 1]) * V[-1, :])
    o = np.argsort(-np.abs(ev))
    return {"ritz": [[float(z.real), float(z.imag)] for z in ev[o]], "res": [float(x) for x in res[o]]}


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    x = sequences(a.data)[0][: a.seqs, : a.T].to(dev)
    e = m.embed(x)
    s = initial_state(m, e)
    gen = torch.Generator(device=dev).manual_seed(0)
    ks = sorted({max(1, r // 2), r, 2 * r, 4 * r})
    out = {"r": r, "m": a.m, "k": {}}
    for k in range(1, max(ks) + 1):
        s = m.step(s, e)
        if k in ks:
            out["k"][str(k)] = arnoldi(lambda z: m.step(z, e), s, a.m, gen)
    return out


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--m", type=int, default=48)
    a = ap.parse_args()
    for_each_run(a, "ritz.json", lambda m, cfg, dev: measure(m, cfg, dev, a), dtype=torch.float64)


if __name__ == "__main__":
    main()
