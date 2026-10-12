from analysis.common import SCALE, brake_of, lambda_profile, load, loss_curve, minima, rise, two_seeds

FIXED = SCALE[:2]
POISSON = SCALE[2]
REFERENCE = {FIXED[0]: "fixed_r8_b8_s0", FIXED[1]: "fixed_r8_b8_s0", POISSON: "poisson_r8_b32_s0"}
R = 8


def main():
    info = {}
    for d in SCALE:
        L = loss_curve(d)
        am, rel = brake_of(lambda_profile(d), R)
        info[d] = dict(L_r=float(L[R]), am=am, rel=rel, minima=minima(L, R), rise=rise(L, R),
                       e2=load(d, "step_geometry.json")["post"]["e2_med"])
        print("  %-26s L_r %.4f | lowest lambda at %d, relative brake %+.3f | minima %s | rise r..2r %+.3f | two modes "
              "%.3f" % (d, info[d]["L_r"], am, rel, info[d]["minima"][:6], info[d]["rise"], info[d]["e2"]))
    ref = {d: float(loss_curve(REFERENCE[d])[R]) for d in SCALE}
    k = sum(info[d]["L_r"] < ref[d] for d in SCALE)
    print("472M: L_r below the 114M model (3 of 3): %s against %s, %d of 3 -> %s" % (
        ", ".join("%.4f" % info[d]["L_r"] for d in SCALE), ", ".join("%.4f" % ref[d] for d in SCALE), k,
        "YES" if k == 3 else "NO"))
    k = sum(abs(info[d]["am"] - R) <= 1 for d in FIXED)
    print("472M fixed: lowest lambda within one loop of r (2 of 2): %d of 2 -> %s" % (k, two_seeds(k, 2)))
    print("472M Poisson: lowest lambda not within one loop of r: loop %d -> %s" % (
        info[POISSON]["am"], "YES" if abs(info[POISSON]["am"] - R) > 1 else "NO"))
    k = sum(len(info[d]["minima"]) >= 2 for d in FIXED)
    print("472M fixed: oscillation (2 of 2): %d of 2 -> %s" % (k, two_seeds(k, 2)))
    kf = sum(info[d]["rise"] >= 0.02 for d in FIXED)
    pm = info[POISSON]["rise"] <= 0.02
    v = "YES" if kf == 2 and pm else ("NO" if kf == 0 and not pm else "grey")
    print("472M: rise in (r, 2r] >= 0.02 in both fixed and <= 0.02 in Poisson: fixed %d of 2, Poisson %+.4f -> %s" % (
        kf, info[POISSON]["rise"], v))
    k = sum(info[d]["e2"] >= 0.80 for d in FIXED)
    print("472M fixed: two leading modes >= 80%% after r (2 of 2): %d of 2 -> %s" % (k, two_seeds(k, 2)))


if __name__ == "__main__":
    main()
