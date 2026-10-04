import torch

from measure.common import base_parser, for_each_run, initial_state, non_normality, sequences


def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    if a.dense:
        loops = list(range(1, 4 * r + 1))
    else:
        loops = sorted({2, max(2, r // 2), max(2, r - 1), r, 2 * r})
    gen = torch.Generator(device=dev).manual_seed(0)
    x = sequences(a.data)[0][: a.seqs, : a.T].to(dev)
    with torch.no_grad():
        e = m.embed(x)
        s = initial_state(m, e)
    out = {"r": r, "n": a.n, "nu": {}, "parts": {}}
    for k in range(1, max(loops) + 1):
        with torch.no_grad():
            s = m.step(s, e)
        if k in loops:
            parts = []
            out["nu"][str(k)] = non_normality(lambda z: m.step(z, e), s, a.n, gen, parts)
            out["parts"][str(k)] = parts
    return out


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=2)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--dense", type=int, default=1, help="1 = every loop 1..4r, 0 = loops 2, r/2, r-1, r, 2r")
    ap.add_argument("--n", type=int, default=0, help="Hutchinson probes; 0 = 16 if dense, 8 otherwise")
    a = ap.parse_args()
    a.n = a.n or (16 if a.dense else 8)
    name = "non_normality.json" if a.dense else "non_normality_coarse.json"
    for_each_run(a, name, lambda m, cfg, dev: measure(m, cfg, dev, a), dtype=torch.float64)


if __name__ == "__main__":
    main()
