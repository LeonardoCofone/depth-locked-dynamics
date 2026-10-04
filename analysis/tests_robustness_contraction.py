import numpy as np

from analysis.common import contraction_rate, depth, has, lambda_profile, load, loss_curve, minima, phase_period, rise

FULL = ["fixed_r%d_b%d_s0" % (r, r) for r in (6, 8, 10, 12, 16, 20, 24)] + ["fixed_r8_b8_s1", "fixed_r12_b12_s1"]
CONTROLS = ["fixed_r8_b8_all_s0", "poisson_r8_b4_s0", "poisson_r8_b32_s0"]
ROBUST = FULL + ["fixed_r8_b4_s1"] + CONTROLS
BRAKE = [d for d in FULL if depth(d) <= 16]
PLANAR = ["fixed_r6_b6_s0", "fixed_r8_b8_s0", "fixed_r8_b8_s1", "fixed_r10_b10_s0", "fixed_r12_b12_s1"]
PILOT = [("P0 fixed r=8, reference", "fixed_r8_b8_D100M_s0"),
         ("P1 Poisson full backprop, update 0.25", "poisson_r8_b32_us0.25_D100M_s0"),
         ("P2 Poisson full backprop, update 0.1", "poisson_r8_b32_us0.1_D100M_s0"),
         ("P3 Poisson full backprop, decay frozen at 0.95", "poisson_r8_b32_dec0.95_dfz_D100M_s0"),
         ("P4 loss at every loop, update 0.25", "fixed_r8_b8_all_us0.25_D100M_s0")]


def verdict(k, n, yes, no):
    f = k / n
    return ("YES" if f >= yes else ("NO" if f <= no else "grey")) + " (%d of %d)" % (k, n)


def lowest(v, r):
    return int(np.argmin(v[:3 * r])) + 1


def robustness():
    info = {}
    for d in ROBUST:
        o = load(d, "robustness.json")
        r = o["r"]
        x = dict(r=r, am16=lowest(o["lyap16_new"], r), am32=lowest(o["lyap32_new"], r),
                 osc_new=len(minima(np.array([np.nan] + o["loss_coda"]), r)) >= 2,
                 osc_orig=len(minima(loss_curve(d), r)) >= 2,
                 PT_sg=phase_period(load(d, "step_geometry.json")["post"]["cumphase_med"]))
        for w in ("r_3r", "r_4r", "r_6r"):
            x["PT_" + w] = phase_period(o["post"][w]["cumphase_med"])
            x["e2_" + w], x["e3_" + w] = o["post"][w]["e2_med"], o["post"][w]["e3_med"]
        info[d] = x
    k = sum(abs(info[d]["am16"] - info[d]["r"]) <= 1 for d in BRAKE)
    kc = sum(abs(info[d]["am16"] - info[d]["r"]) > 1 for d in CONTROLS)
    print("new sequences, lowest lambda within one loop of r: %s; controls without it: %d of 3" % (
        verdict(k, 7, 6 / 7, 4 / 7), kc))
    k = sum(abs(info[d]["am32"] - info[d]["am16"]) <= 1 for d in ROBUST)
    print("32 directions, lowest lambda within one loop of that with 16: %s" % verdict(k, 13, 11 / 13, 8 / 13))
    k1 = sum(info[d]["e2_r_4r"] >= 0.80 for d in PLANAR)
    k2 = sum(info[d]["e3_r_4r"] - info[d]["e2_r_4r"] <= 0.10 for d in PLANAR)
    print("new sequences, two modes >= 80%% in %d of 5, third mode <= 10%% in %d of 5 -> %s" % (
        k1, k2, "YES" if k1 >= 4 and k2 >= 4 else "NO"))
    for w in ("r_3r", "r_6r"):
        ok = [d for d in FULL if info[d]["PT_r_4r"] is not None and info[d]["PT_" + w] is not None]
        k = sum(abs(info[d]["PT_" + w] / info[d]["PT_r_4r"] - 1) <= 0.15 for d in ok)
        print("P_Theta from the window %s within 15%% of the window to 4r: %s" % (w, verdict(k, len(ok), 2 / 3, 1 / 3)))
    ok = [d for d in FULL if info[d]["PT_r_4r"] is not None and info[d]["PT_sg"] is not None]
    k = sum(abs(info[d]["PT_r_4r"] / info[d]["PT_sg"] - 1) <= 0.15 for d in ok)
    print("P_Theta on new sequences within 15%% of the original ones: %s" % verdict(k, len(ok), 2 / 3, 1 / 3))
    k = sum(info[d]["osc_new"] == info[d]["osc_orig"] for d in ROBUST)
    print("new sequences, same oscillation verdict: %s; controls oscillating: %d of 3" % (
        verdict(k, 13, 10 / 13, 9 / 13), sum(info[d]["osc_new"] for d in CONTROLS)))


def describe(d):
    r = depth(d)
    L = loss_curve(d)
    lam = lambda_profile(d)
    am = lowest(lam, r)
    sg = load(d, "step_geometry.json")
    PT = phase_period(sg["post"]["cumphase_med"])
    print("  %-32s q %.3f | L_r %.3f, rise r..2r %+.3f, minima %s | lowest lambda at %d | phase at 4r %.2f"
          " | P_Theta %s" % (d, contraction_rate(d), L[r], rise(L, r), minima(L, r), am, sg["post"]["cumphase_med"][-1],
                             "-" if PT is None else "%.1f" % PT))
    return dict(q=contraction_rate(d), brake=abs(am - r) <= 1, rise=rise(L, r), osc=len(minima(L, r)) >= 2)


def pilot():
    best = None
    for label, d in PILOT:
        print(label)
        q = describe(d)["q"]
        if not label.startswith("P0") and q >= 0.85 and (best is None or q > best[1] + 0.01):
            best = (label, q)
    print("chosen: %s" % ("none reaches q >= 0.85" if best is None else "%s (q %.3f)" % best))


def matched(gamma, q_rule, rise_bound, oscillation_test):
    med = ["poisson_r8_b32_us%g_s%d" % (gamma, s) for s in (0, 1)]
    fix = ["fixed_r8_b8_us%g_s%d" % (gamma, s) for s in (0, 1)]
    if not all(has(d, "curves.json") and has(d, "lyapunov.json") and has(d, "step_geometry.json") for d in med + fix):
        print("  results missing")
        return
    M, F = [describe(d) for d in med], [describe(d) for d in fix]
    q_rule(gamma, M, F)
    a1 = sum((not v["brake"]) and v["rise"] <= rise_bound for v in M)
    print("damped by %g, Poisson without brake and rise <= %g: %s" % (gamma, rise_bound, "YES" if a1 == 2 else (
        "NO" if all(v["brake"] and v["rise"] > rise_bound for v in M) else "grey")))
    a2 = sum(v["brake"] and v["rise"] >= 0.02 for v in F)
    print("damped by %g, fixed with brake and rise >= 0.02: %s" % (gamma, {2: "YES", 1: "grey", 0: "NO"}[a2]))
    if oscillation_test:
        a3 = sum(v["osc"] for v in F)
        print("damped by %g, fixed oscillates: %s" % (gamma, {2: "YES", 1: "grey", 0: "NO"}[a3]))


def q_absolute(gamma, M, F):
    k = sum(v["q"] >= 0.85 for v in M + F)
    print("damped by %g, q >= 0.85 in all four: %s (%d of 4)" % (gamma, "YES" if k == 4 else "NO", k))


def q_relative(gamma, M, F):
    gap = abs(np.mean([v["q"] for v in M]) - np.mean([v["q"] for v in F]))
    ok = all(v["q"] >= 0.80 for v in M + F) and gap <= 0.03
    print("damped by %g, q >= 0.80 in all four and mean gap <= 0.03 (gap %.3f): %s"
          % (gamma, gap, "YES" if ok else "NO"))


def main():
    print("== Robustness on new sequences and with 32 directions")
    robustness()
    print("\n== Contraction pilot at 0.1B tokens (selection rule)")
    pilot()
    print("\n== Update damped by 0.1, Poisson against fixed, 0.5B tokens")
    matched(0.1, q_absolute, 0.005, False)
    print("\n== Update damped by 0.25, Poisson against fixed, 0.5B tokens")
    matched(0.25, q_relative, 0.02, True)


if __name__ == "__main__":
    main()
