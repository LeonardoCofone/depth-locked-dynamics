import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state, lyapunov, sequences, window_geometry

NEW_SEED = 2027
MODELS = ["fixed_r%d_b%d_s0" % (r, r) for r in (6, 8, 10, 12, 16, 20, 24)] + [
    "fixed_r8_b8_s1", "fixed_r12_b12_s1", "fixed_r8_b4_s1", "fixed_r8_b8_all_s0", "poisson_r8_b4_s0",
    "poisson_r8_b32_s0"]


@torch.no_grad()
def lambda_profile(m, cfg, x, ndir, dev):
    r = cfg["mean_rec"]
    e = m.embed(x)
    steps = [lambda s: m.step(s, e)] * max(2 * r, r + 8)
    exps, _ = lyapunov(steps, initial_state(m, e), torch.Generator(device=dev).manual_seed(0), ndir)
    return exps.mean(1).tolist()


@torch.no_grad()
def trajectory(m, cfg, x, y):
    r = cfg["mean_rec"]
    K = 6 * r
    e = m.embed(x)
    s = initial_state(m, e)
    S, losses = [s.float()], []
    for _ in range(K):
        s = m.step(s, e)
        S.append(s.float())
        losses.append(float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten())))
    St = torch.stack(S).reshape(K + 1, -1, S[0].shape[-1])
    D = St[1:] - St[:-1]
    sr = St[r]
    dist = [float(((St[k] - sr).norm(dim=-1) / sr.norm(dim=-1).clamp_min(1e-30)).median()) for k in range(K + 1)]
    windows = {"r_3r": window_geometry(D[r:3 * r]), "r_4r": window_geometry(D[r:4 * r]),
               "r_6r": window_geometry(D[r:6 * r])}
    return {"loss_coda": losses, "dist_med": dist, "post": windows}


def measure(m, cfg, dev, a):
    xn, yn = sequences(a.data, seed=NEW_SEED)
    x_ly = xn[: a.ly_seqs, : a.ly_T].to(dev)
    res = {"r": cfg["mean_rec"], "new_seed": NEW_SEED,
           "lyap16_new": lambda_profile(m, cfg, x_ly, 16, dev), "lyap32_new": lambda_profile(m, cfg, x_ly, 32, dev)}
    m.float()
    res.update(trajectory(m, cfg, xn[: a.seqs, : a.T].to(dev), yn[: a.seqs, : a.T].to(dev)))
    return res


def main():
    ap = base_parser()
    ap.add_argument("--ly_seqs", type=int, default=8)
    ap.add_argument("--ly_T", type=int, default=256)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    for_each_run(a, "robustness.json", lambda m, cfg, dev: measure(m, cfg, dev, a), dtype=torch.float64,
                 keep=lambda cfg, name: name in MODELS or bool(a.only))


if __name__ == "__main__":
    main()
