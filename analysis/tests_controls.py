import numpy as np

from analysis.common import (NEW_FIXED, cv, is_base_full, load_public, paper_models, share, summaries,
                             two_seeds)

GROUPS = {"random depth, full backprop": ["poisson_r8_b32_s%d" % s for s in (0, 1)],
          "fixed r=8 truncated to 4": ["fixed_r8_b4_s%d" % s for s in (0, 1)],
          "loss at every loop": ["fixed_r8_b8_all_s%d" % s for s in (0, 1)],
          "random depth, decay 0.9": ["poisson_r8_b4_dec0.9_s%d" % s for s in (0, 1)]}
BOOT = 2000


def recipe_tests(S):
    have = {g: [S[d] for d in ds if d in S and "brake" in S[d]] for g, ds in GROUPS.items()}
    g = have["random depth, full backprop"]
    kb, ko = sum(o["brake"] for o in g), sum(o["osc"] for o in g)
    print("Poisson through every loop: brake %d of 2 -> %s; oscillation %d of 2 -> %s" % (
        kb, two_seeds(kb, len(g)), ko, two_seeds(ko, len(g))))
    g = have["fixed r=8 truncated to 4"]
    kb = sum(o["argmin_ok"] for o in g)
    print("fixed r=8 truncated to 4: lowest lambda within one loop of r in %d of 2 -> %s" % (kb, two_seeds(kb, len(g))))
    g = have["loss at every loop"]
    kb, ko = sum(o["brake"] for o in g), sum(o["osc"] for o in g)
    verdict = "YES" if kb == len(g) == 2 and ko == 0 else ("NO" if kb == 0 or ko == len(g) else "grey")
    print("loss at every loop: brake %d of 2 and no oscillation (%d oscillate) -> %s" % (kb, ko, verdict))
    g = [S[d] for d in GROUPS["random depth, decay 0.9"] if d in S]
    kq = sum(o["q"] >= 0.76 for o in g)
    print("Poisson with decay 0.9 reaches q >= 0.76: q = %s -> %s"
          % ([round(o["q"], 3) for o in g], two_seeds(kq, len(g))))


def new_fixed_tests(S):
    nf = [S[d] for d in NEW_FIXED if d in S]
    print("new fixed: lowest lambda within one loop of r: %s; relative brake <= -0.05 in %d" % (
        share(sum(o["argmin_ok"] for o in nf), len(nf), 0.9), sum(o["brake"] for o in nf)))
    print("  outside: %s" % [o["name"] for o in nf if not o["argmin_ok"]])
    with_pt = [o for o in nf if o.get("PT_sg")]
    k = sum(1.6 <= o["PT_sg"] / o["r"] <= 2.5 for o in with_pt)
    print("new fixed: P_Theta/r in [1.6, 2.5]: %s" % share(k, len(with_pt), 2 / 3))
    base = [o for d, o in S.items() if is_base_full(d) and 4 <= o["r"] <= 24 and o.get("PT_sg")]
    c = cv([o["PT_sg"] / o["r"] for o in base])
    print("all baseline fixed r=4..24: CV of P_Theta/r = %.3f over %d models -> %s"
          % (c, len(base), "YES" if c <= 0.25 else "NO"))
    scaled = [o for o in nf if o.get("scaled2r") is not None]
    print("new fixed: update scaled by 1/2, loss at 2r within 0.02 of L_r: %s" % share(
        sum(o["scaled2r"] <= 0.02 for o in scaled), len(scaled), 2 / 3))


def exponents(S):
    by_depth = {}
    for d, o in S.items():
        if is_base_full(d) and 6 <= o["r"] <= 24 and all(k in o for k in ("n_u", "al_u", "lam_u")):
            by_depth.setdefault(o["r"], []).append(o)
    rs = sorted(by_depth)
    keys = {"step": "n_u", "angle": "al_u", "log-stretch": "lam_u"}

    def mean_of(o, key):
        return float(np.mean(np.abs(o[key]))) if key == "lam_u" else float(np.mean(o[key]))

    def fit(vals):
        x = np.log(np.array(rs, dtype=float))
        y = np.log(np.array([np.mean(vals[r]) for r in rs]))
        return float(-np.polyfit(x, y, 1)[0])

    rng = np.random.default_rng(0)
    for q, key in keys.items():
        vals = {r: [mean_of(o, key) for o in by_depth[r]] for r in rs}
        b0 = fit(vals)
        boot = [fit({r: list(rng.choice(vals[r], len(vals[r]))) for r in rs}) for _ in range(BOOT)]
        lo, hi = np.quantile(boot, [0.05, 0.95])
        print("exponent of the %-11s %.2f [%.2f, %.2f] -> interval contains one: %s" % (
            q, b0, lo, hi, "YES" if lo <= 1 <= hi else "NO"))
    for q, key in keys.items():
        raw = np.array([np.mean([o[key] for o in by_depth[r]], 0) for r in rs])
        scaled = raw * np.array(rs, dtype=float)[:, None]
        cs = float(np.mean(np.abs(scaled.std(0) / scaled.mean(0))))
        cr = float(np.mean(np.abs(raw.std(0) / raw.mean(0))))
        print("collapse of r*%-11s CV %.3f against %.3f -> %s"
              % (q, cs, cr, "YES" if cs <= 0.25 and cs <= 0.5 * cr else "NO"))


def rotation_engine(S):
    base = [o for d, o in S.items() if is_base_full(d) and 4 <= o["r"] <= 24 and o.get("b") is not None]
    c = cv([o["b"] for o in base])
    print("r*phi_r constant over %d baseline fixed models: CV %.3f -> %s" % (
        len(base), c, "YES" if c <= 0.20 else ("NO" if c > 0.30 else "grey")))
    nf = [S[d] for d in NEW_FIXED if d in S and S[d].get("b") is not None]
    print("new fixed: r*phi_r in [2.4, 4.0]: %s" % share(sum(2.4 <= o["b"] <= 4.0 for o in nf), len(nf), 2 / 3))
    with_p = [o for o in nf if o.get("PT_pd")]
    print("new fixed: one turn at phi_r within 30%% of P_Theta: %s" % share(
        sum(abs(2 * np.pi / o["phi"] / o["PT_pd"] - 1) <= 0.30 for o in with_p), len(with_p), 0.70))
    print("new fixed: without attention and without MLP, each leaves 0.2 to 0.7 of phi_r: %s" % share(
        sum(0.2 <= o["noattn"] <= 0.7 and 0.2 <= o["nomlp"] <= 0.7 for o in nf), len(nf), 2 / 3))


def ouro_scaled():
    out = {}
    for name in ("Ouro-1.4B", "Ouro-2.6B"):
        o = load_public(name, "ouro_scaled.json")
        trained, half, r = o["trained"], o["scaled_rp8"], o["r"]
        out[name] = dict(c1=half[2 * r - 1] - trained[r - 1], c2=half[2 * r - 1] < trained[2 * r - 1],
                         c3=max(abs(half[2 * k - 1] - trained[k - 1]) for k in (2, 3, 4)))
        print("  %s: L_8 at step 1/2 minus L_4 %+.3f; below the trained loop at 8: %s; max |L_2k - L_k| %.3f" % (
            name, out[name]["c1"], out[name]["c2"], out[name]["c3"]))
    n = len(out)
    print("Ouro, update scaled by 1/2: loss at loop 8 within 0.02 of L_4 -> %s"
          % two_seeds(sum(v["c1"] <= 0.02 for v in out.values()), n))
    print("Ouro, same: loss at loop 8 below the trained loop at 8 -> %s"
          % two_seeds(sum(v["c2"] for v in out.values()), n))
    print("Ouro, same: L_2k within 0.05 of L_k for k = 2, 3, 4 -> %s"
          % two_seeds(sum(v["c3"] <= 0.05 for v in out.values()), n))


def main():
    S = summaries(paper_models())
    print("== Recipes at r = 8")
    recipe_tests(S)
    print("\n== New fixed models (r = 6, 10, 20, 24 and second seeds at 12, 16)")
    new_fixed_tests(S)
    print("\n== Exponents and collapse, baseline fixed r = 6..24")
    exponents(S)
    print("\n== Rotation at landing")
    rotation_engine(S)
    print("\n== Ouro with the scaled update")
    ouro_scaled()


if __name__ == "__main__":
    main()
