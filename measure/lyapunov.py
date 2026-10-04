import torch

from measure.common import base_parser, for_each_run, initial_state, lyapunov, sequences


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    x = sequences(a.data)[0][: a.seqs, : a.T].to(dev)
    e = m.embed(x)
    K = max(2 * r, r + 8)
    steps = [lambda s: m.step(s, e)] * K
    exps, _ = lyapunov(steps, initial_state(m, e), torch.Generator(device=dev).manual_seed(0), a.ndir)
    return {"r": r, "ndir": a.ndir, "lyap": exps.tolist()}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--ndir", type=int, default=16)
    a = ap.parse_args()
    for_each_run(a, "lyapunov.json", lambda m, cfg, dev: measure(m, cfg, dev, a), dtype=torch.float64)


if __name__ == "__main__":
    main()
