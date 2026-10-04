import torch

from measure.common import base_parser, for_each_run, initial_state, quantiles, sequences
from measure.plane_dynamics import phi
from measure.scaled_update import loss_of


@torch.no_grad()
def run_variant(m, e, y, s0, k_start, K, keep):
    s, losses, states = s0, [loss_of(m, s0, y)], {}
    for k in range(K):
        nxt = m.step(s, e)
        s = nxt if k < k_start else s + 2.0 * (nxt - s)
        losses.append(loss_of(m, s, y))
        if k + 1 in keep:
            states[k + 1] = s.clone()
    return losses, states


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    res = {"r": r, "seqs": a.seqs, "T": a.T}
    e = m.embed(x)
    s0 = initial_state(m, e)
    s1 = m.step(s0, e)
    res["core_check_f32"] = float((phi(m, s0, e) - s1).norm() / s1.norm())
    trained, states = run_variant(m, e, y, s0, 10 ** 9, 2 * r, {r, r // 2})
    sr = states[r]
    res["trained"] = {"loss": trained, "L_r": trained[r]}
    check, _ = run_variant(m, e, y, s0, 10 ** 9, 3, set())
    res["check_trained"] = max(abs(p - q) for p, q in zip(check, trained[:4]))
    variants = {}
    for ks in sorted(set([0, 1, 2, 4, r // 2])):
        arrive = ks + (r - ks) // 2
        losses, st = run_variant(m, e, y, s0, ks, max(2 * arrive, arrive + 2), {arrive, arrive + 1})
        variants["double_from%d" % ks] = {
            "k_start": ks, "arrive": arrive, "odd": (r - ks) % 2 == 1, "loss": losses, "L_arrive": losses[arrive],
            "L_arrive_plus1": losses[arrive + 1],
            "L_trained_same_loop": trained[arrive] if arrive < len(trained) else None,
            "jump_first_double": losses[ks + 1] - trained[ks + 1],
            "dist_arrive_from_sr": {str(k): quantiles((st[k] - sr).norm(dim=-1) / sr.norm(dim=-1).clamp_min(1e-30))
                                    for k in st},
            "max_loss_double_phase": max(losses[ks + 1: arrive + 1]) if arrive > ks else None}
    res["variants"] = variants
    return res


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--min_r", type=int, default=8)
    a = ap.parse_args()
    for_each_run(a, "doubled_update.json", lambda m, cfg, dev: measure(m, cfg, dev, a),
                 keep=lambda cfg, name: cfg["sampling"] == "fixed" and cfg["mean_rec"] >= a.min_r)


if __name__ == "__main__":
    main()
