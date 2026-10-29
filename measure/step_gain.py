import numpy as np
import torch

from measure.common import base_parser, for_each_run, initial_state, sequences
from measure.plane_dynamics import eig_rho_theta, phi, window_dmd

TS = (1e-3, 1e-2, 0.1, 0.3, 1.0)


def quantiles(t, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    t = t[torch.isfinite(t)].double()
    return {str(p): float(t.quantile(p)) for p in ps} if t.numel() else None


@torch.no_grad()
def measure(m, cfg, xb):
    r = cfg["mean_rec"]
    K = 4 * r + 1
    m.double()
    e = m.embed(xb)
    S = [initial_state(m, e)]
    for _ in range(K):
        S.append(phi(m, S[-1], e))
    shape, dm = S[-1].shape, S[-1].shape[-1]
    flat = lambda z: z.reshape(-1, dm)
    St = torch.stack([flat(z) for z in S])
    D = St[1:] - St[:-1]
    Q = window_dmd(D[r:4 * r], 2)[3]
    f = lambda z: flat(phi(m, z, e))
    res = {"r": r, "K": K, "points": {}}
    for k in sorted(set([r, r + r // 2, 2 * r, 3 * r])):
        sk, dk = S[k], D[k]
        nd = dk.norm(dim=-1).clamp_min(1e-300)
        f0 = f(sk)

        def cdiff(v, t):
            V = v.reshape(shape)
            return (f(sk + t * V) - f(sk - t * V)) / (2 * t)

        q = Q.transpose(1, 2) @ dk[:, :, None]
        Jin = cdiff((Q @ q)[:, :, 0], 1e-3)
        a = (Q.transpose(1, 2) @ Jin[:, :, None])[:, :, 0].norm(dim=-1) / nd
        Jd3, Jd4 = cdiff(dk, 1e-3), cdiff(dk, 1e-4)
        b, b4 = Jd3.norm(dim=-1) / nd, Jd4.norm(dim=-1) / nd
        cc = D[k + 1].norm(dim=-1) / nd
        g = {t: (f(sk + t * dk.reshape(shape)) - f0).norm(dim=-1) / (t * nd) for t in TS}
        JQ = torch.stack([cdiff(Q[:, :, j] * nd[:, None], 1e-3) / nd[:, None] for j in range(2)], -1)
        rhoB, thB, _ = eig_rho_theta(Q.transpose(1, 2) @ JQ)
        lba, lcb = torch.log(b / a.clamp_min(1e-300)), torch.log(cc / b.clamp_min(1e-300))
        res["points"]["k%d" % k] = {
            "k": k,
            "check_phi_step": float((f0 - flat(m.step(sk, e))).norm() / f0.norm()),
            "check_g1_vs_c": float(((g[1.0] - cc).abs() / cc.clamp_min(1e-300)).median()),
            "check_b3_vs_b4": float(((b - b4).abs() / b4.clamp_min(1e-300)).median()),
            "a": quantiles(a), "b": quantiles(b), "c": quantiles(cc), "rhoB": quantiles(rhoB), "thetaB": quantiles(thB),
            "log_b_over_a_med": float(lba.median()), "log_c_over_b_med": float(lcb.median()),
            "g_med": {str(t): float(v.median()) for t, v in g.items()},
            "step_rel_med": float((nd / St[k].norm(dim=-1)).median()),
            "step_in_plane_med": float((q[:, :, 0].norm(dim=-1) / nd).median()),
        }
    allk = {}
    for k in range(r, 3 * r):
        sk, dk = S[k], D[k]
        nd = dk.norm(dim=-1).clamp_min(1e-300)
        cd = lambda v, t=1e-3: (f(sk + t * v.reshape(shape)) - f(sk - t * v.reshape(shape))) / (2 * t)
        q = Q.transpose(1, 2) @ dk[:, :, None]
        a = (Q.transpose(1, 2) @ cd((Q @ q)[:, :, 0])[:, :, None])[:, :, 0].norm(dim=-1) / nd
        b = cd(dk).norm(dim=-1) / nd
        cc = D[k + 1].norm(dim=-1) / nd
        JQ = torch.stack([cd(Q[:, :, j] * nd[:, None]) / nd[:, None] for j in range(2)], -1)
        rhoB = eig_rho_theta(Q.transpose(1, 2) @ JQ)[0]
        allk[str(k)] = {"a": float(a.median()), "b": float(b.median()), "c": float(cc.median()),
                        "rhoB": float(rhoB.median()), "in_plane": float((q[:, :, 0].norm(dim=-1) / nd).median())}
    res["all_k"] = allk
    res["geo_mean_r_3r"] = {x: float(np.exp(np.mean([np.log(max(v[x], 1e-300)) for v in allk.values()])))
                            for x in ("a", "b", "c", "rhoB")}
    P = list(res["points"].values())
    Mba = float(np.mean([p["log_b_over_a_med"] for p in P]))
    Mcb = float(np.mean([p["log_c_over_b_med"] for p in P]))
    gap = Mba + Mcb
    res["summary"] = {"M_ba": Mba, "M_cb": Mcb, "gap": gap,
                      "share_plane_exit": Mba / gap if gap != 0 else None,
                      "share_nonlinear": Mcb / gap if gap != 0 else None,
                      "g1_gt_g001": bool(np.median([p["g_med"]["1.0"] for p in P]) >
                                         np.median([p["g_med"]["0.01"] for p in P])),
                      "max_check_phi_step": max(p["check_phi_step"] for p in P),
                      "max_check_g1_vs_c": max(p["check_g1_vs_c"] for p in P),
                      "max_check_b3_vs_b4": max(p["check_b3_vs_b4"] for p in P)}
    m.float()
    return res


def main():
    ap = base_parser()
    ap.add_argument("--jac_seqs", type=int, default=2)
    ap.add_argument("--jac_T", type=int, default=256)
    a = ap.parse_args()
    xb, _ = sequences(a.data)
    xj = xb[: a.jac_seqs, : a.jac_T]

    def run(m, cfg, dev):
        res = measure(m, cfg, xj.to(dev))
        res.update(jac_seqs=a.jac_seqs, jac_T=a.jac_T)
        return res

    for_each_run(a, "step_gain.json", run,
                 keep=lambda cfg, name: cfg["sampling"] == "fixed" and cfg["mean_bptt"] >= cfg["mean_rec"] >= 8)


if __name__ == "__main__":
    main()
