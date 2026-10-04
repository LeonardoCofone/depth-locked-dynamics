import torch

from measure.common import base_parser, for_each_run, initial_state, sequences


def ridge_decoder(X, y, lam):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    G = Xc.T @ Xc
    lam_eff = lam * float(torch.trace(G)) / G.shape[0]
    W = torch.linalg.solve(G + lam_eff * torch.eye(G.shape[0], device=G.device, dtype=G.dtype), Xc.T @ (y - ym))
    return W, mu, ym


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    K = 4 * r
    fit = a.fitK or r
    x = sequences(a.data)[0][: a.seqs, : a.T].to(dev)
    e = m.embed(x)
    s = initial_state(m, e)
    Z, step, prev = [], [], s
    for _ in range(K):
        s = m.step(s, e)
        step.append(float((s - prev).norm(dim=-1).mean() / prev.norm(dim=-1).mean().clamp_min(1e-12)))
        prev = s
        z = s[:, ::a.sub].float()
        Z.append(z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    Z = torch.stack(Z)
    B = Z.shape[1]
    train, test = slice(0, B // 2), slice(B // 2, B)
    X = Z[:fit, train].reshape(-1, Z.shape[-1])
    target = torch.arange(1, fit + 1, device=Z.device, dtype=Z.dtype)
    target = target.repeat_interleave(Z[0, train].numel() // Z.shape[-1])
    W, mu, ym = ridge_decoder(X, target, a.lam)
    decoded = [(Z[k, test].reshape(-1, Z.shape[-1]) - mu) @ W + ym for k in range(K)]
    truth = torch.cat([torch.full_like(decoded[k], k + 1) for k in range(fit)])
    pred = torch.cat(decoded[:fit])
    r2 = float(1 - ((pred - truth) ** 2).mean() / ((truth - truth.mean()) ** 2).mean())
    return {"r": r, "K": K, "fitK": fit, "r2": r2, "khat_mean": [float(h.mean()) for h in decoded],
            "khat_q25": [float(h.quantile(0.25)) for h in decoded],
            "khat_q75": [float(h.quantile(0.75)) for h in decoded], "step": step}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--sub", type=int, default=2, help="keep one token every sub")
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge, relative to trace(X^T X)/d")
    ap.add_argument("--fitK", type=int, default=0, help="0 = fit on loops 1..r, otherwise on loops 1..fitK")
    a = ap.parse_args()
    name = "loop_decoder.json" if not a.fitK else "loop_decoder_k%d.json" % a.fitK
    for_each_run(a, name, lambda m, cfg, dev: measure(m, cfg, dev, a))


if __name__ == "__main__":
    main()
