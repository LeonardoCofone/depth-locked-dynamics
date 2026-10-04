import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state, sequences, window_geometry


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    K = 4 * r
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    e = m.embed(x)
    s = initial_state(m, e)
    S, losses = [s.float()], []
    for _ in range(K):
        s = m.step(s, e)
        S.append(s.float())
        losses.append(float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten())))
    St = torch.stack(S).reshape(K + 1, -1, S[0].shape[-1])
    D = St[1:] - St[:-1]
    scale = float((St[r].double() ** 2).sum(-1).mean())
    Dd = D.double()
    G = torch.einsum("and,bnd->ab", Dd, Dd) / Dd.shape[1] / scale
    return {"r": r, "K": K, "scale_sr2": scale, "loss_coda": losses, "G": G.cpu().numpy().round(8).tolist(),
            "post": window_geometry(D[r:]), "pre": window_geometry(D[:r])}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    for_each_run(a, "step_geometry.json", lambda m, cfg, dev: measure(m, cfg, dev, a))


if __name__ == "__main__":
    main()
