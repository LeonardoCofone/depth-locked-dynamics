import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, initial_state, quantiles, sequences
from measure.plane_dynamics import part_D, phi


@torch.no_grad()
def loss_of(m, s, y):
    return float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten()))


def tok_rel(a, b, ref):
    return quantiles((a - b).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30))


@torch.no_grad()
def scaled_run(m, e, y, s0, r, rp, times):
    g = r / rp
    s, states, losses = s0, {}, [loss_of(m, s0, y)]
    for k in range(rp):
        s = s + g * (m.step(s, e) - s) if rp != r else m.step(s, e)
        for t in times:
            if abs(t - (k + 1) / rp) < 1e-9:
                states[t] = s.clone()
        losses.append(loss_of(m, s, y))
    return states, losses


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    res = {"r": r, "seqs": a.seqs, "T": a.T}
    e = m.embed(x)
    s0 = initial_state(m, e)
    s1 = m.step(s0, e)
    res["check_rp_eq_r"] = float(((s0 + 1.0 * (m.step(s0, e) - s0)) - s1).norm() / s1.norm())
    res["core_check_f32"] = float((phi(m, s0, e) - s1).norm() / s1.norm())
    times = [j / 4 for j in range(1, 5) if (j * r) % 4 == 0]
    if 1.0 not in times:
        times.append(1.0)
    res["ts"] = times
    rps = [r, 2 * r, 4 * r, 8 * r]
    runs, curves = {}, {}
    for rp in rps:
        runs[rp], curves[rp] = scaled_run(m, e, y, s0, r, rp, times)
    ref = runs[r][1.0]
    conv = {"rp%d" % rp: {"loss": {str(t): loss_of(m, runs[rp][t], y) for t in times}, "curve": curves[rp]}
            for rp in rps}
    conv["dist_r_to_8r"] = {str(t): tok_rel(runs[r][t], runs[8 * r][t], ref) for t in times}
    for p, q in zip(rps[:-1], rps[1:]):
        conv["diff_%d_%d" % (p, q)] = {str(t): tok_rel(runs[q][t], runs[p][t], ref) for t in times}
    res["conv"] = conv
    res["richardson"] = {}
    for p, q in zip(rps[:-1], rps[1:]):
        sR = 2 * runs[q][1.0] - runs[p][1.0]
        res["richardson"]["R_%d_%d" % (p, q)] = {"loss": loss_of(m, sR, y), "dist_from_sr": tok_rel(sR, ref, ref)}
    res["L_r"] = conv["rp%d" % r]["loss"]["1.0"]
    del runs
    parts = {"inj": [], "core": [], "tot": [], "cos": []}
    s = s0
    for _ in range(2 * r):
        u = m.inject(s, e)
        nxt = phi(m, s, e)
        inj, core = u - s, nxt - u
        ns = s.norm(dim=-1).clamp_min(1e-30)
        parts["inj"].append(float((inj.norm(dim=-1) / ns).median()))
        parts["core"].append(float((core.norm(dim=-1) / ns).median()))
        parts["tot"].append(float(((inj + core).norm(dim=-1) / ns).median()))
        parts["cos"].append(float(F.cosine_similarity(inj, core, dim=-1).median()))
        s = m.step(s, e)
    res["step_parts"] = parts
    res["D"] = part_D(m)
    return res


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    a = ap.parse_args()
    for_each_run(a, "scaled_update.json", lambda m, cfg, dev: measure(m, cfg, dev, a),
                 keep=lambda cfg, name: cfg["mean_rec"] >= 4)


if __name__ == "__main__":
    main()
