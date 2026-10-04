import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state, sequences
from measure.loop_decoder import ridge_decoder


def unit(z):
    return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def fit_decoder(m, e, r, sub, lam):
    s = initial_state(m, e)
    Z = []
    for _ in range(r):
        s = m.step(s, e)
        Z.append(unit(s[:, ::sub].float()))
    X = torch.stack(Z).reshape(-1, Z[0].shape[-1])
    y = torch.arange(1, r + 1, device=X.device, dtype=X.dtype).repeat_interleave(Z[0].numel() // Z[0].shape[-1])
    W, mu, ym = ridge_decoder(X, y, lam)
    pred = (X - mu) @ W + ym
    r2 = float(1 - ((pred - y) ** 2).mean() / ((y - y.mean()) ** 2).mean())
    return W, mu, ym, r2


def decode(s, W, mu, ym):
    return (unit(s.float()) - mu) @ W + ym


def moved(s, a, v):
    return s + a * s.norm(dim=-1, keepdim=True) * v


@torch.no_grad()
def calibrate(s, v, shift, W, mu, ym):
    base = float(decode(s, W, mu, ym).mean())
    gain = float(v @ W)
    a = shift / gain if abs(gain) > 1e-9 else 0.0
    for _ in range(3):
        got = float(decode(moved(s, a, v), W, mu, ym).mean()) - base
        if abs(got) < 1e-9:
            break
        a *= shift / got
    return a, float(decode(moved(s, a, v), W, mu, ym).mean()) - base


@torch.no_grad()
def losses_from(m, e, y, s, k0, K):
    out = []
    for _ in range(k0, K):
        s = m.step(s, e)
        out.append(float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten())))
    return out


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    K, k0, shift = 4 * r, max(1, r // 2), max(1, r // 4)
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    B = x.shape[0]
    e_train, e_test, y_test = m.embed(x[: B // 2]), m.embed(x[B // 2:]), y[B // 2:]
    W, mu, ym, r2 = fit_decoder(m, e_train, r, a.sub, a.lam)
    u = (W / W.norm()).to(e_test.dtype)
    s = initial_state(m, e_test)
    pre = []
    for _ in range(k0):
        s = m.step(s, e_test)
        pre.append(float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y_test.flatten())))
    out = {"r": r, "K": K, "k0": k0, "D": shift, "r2_fit": r2, "khat_k0": float(decode(s, W, mu, ym).mean()),
           "loss_pre": pre, "clean": losses_from(m, e_test, y_test, s, k0, K), "edits": []}
    gen = torch.Generator(device=u.device).manual_seed(11)
    directions = [("coord", u)]
    for i in range(a.ndir):
        v = torch.randn(u.shape, generator=gen, device=u.device, dtype=u.dtype)
        v = v - (v @ u) * u
        directions.append(("rand%d" % i, v / v.norm()))
    a_coord = {}
    for name, v in directions:
        for sign in (1, -1):
            if name == "coord":
                amount, got = calibrate(s, v, sign * shift, W, mu, ym)
                a_coord[sign] = amount
            else:
                amount = abs(a_coord.get(sign, 0.0)) * sign
                got = float(decode(moved(s, amount, v), W, mu, ym).mean()) - out["khat_k0"]
            s2 = moved(s, amount, v)
            rel = float((s2 - s).norm(dim=-1).mean() / s.norm(dim=-1).mean())
            out["edits"].append({"dir": name, "sign": sign, "a": amount, "rel_shift": rel, "khat_change": got,
                                 "loss": losses_from(m, e_test, y_test, s2, k0, K)})
    return out


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--sub", type=int, default=2, help="keep one token every sub for the decoder")
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge, relative to trace(X^T X)/d")
    ap.add_argument("--ndir", type=int, default=2, help="random control directions")
    a = ap.parse_args()
    for_each_run(a, "temporal_edit.json", lambda m, cfg, dev: measure(m, cfg, dev, a))


if __name__ == "__main__":
    main()
