import numpy as np

from analysis.common import depth, load

MEASURED = ["fixed_r8_b8_s0", "fixed_r8_b8_s1", "fixed_r12_b12_s0", "fixed_r12_b12_s1", "fixed_r16_b16_s0",
            "fixed_r16_b16_s1", "fixed_r8_b8_D1500M_s0", "fixed_r16_b16_D1500M_s0", "fixed_r8_b8_d768_s0"]
PRECISION = 1e-3   # J d_k with step 1e-3 |d_k| against 1e-4 |d_k|, median relative difference


def main():
    print("== Gain along the step after r (descriptive)")
    kept = []
    for d in MEASURED:
        j = load(d, "step_gain.json")
        r, s, p, g = depth(d), j["summary"], j["points"], j["geo_mean_r_3r"]
        ok = s["max_check_b3_vs_b4"] <= PRECISION
        late = [p["k%d" % (2 * r)]["step_in_plane_med"], p["k%d" % (3 * r)]["step_in_plane_med"]]
        worst = max(abs(q["log_c_over_b_med"]) for q in p.values())
        print("  %-24s r=%-2d precision %.1e %s | per loop over r..3r: true %.3f, full Jacobian %.3f, plane %.3f | "
              "full Jacobian against true: mean %+.4f, worst loop %+.4f (log) | in the plane at r %.3f, at 2r and 3r "
              "%.3f %.3f" % (d, r, s["max_check_b3_vs_b4"], "ok" if ok else "FAILED", g["c"], g["b"], g["a"],
                             s["M_cb"], worst, p["k%d" % r]["step_in_plane_med"], *late))
        if ok:
            kept.append((g, s, p, r, late, worst))
    G = lambda x: [k[0][x] for k in kept]
    print("models with the precision check passed: %d of %d" % (len(kept), len(MEASURED)))
    print("true gain per loop over r..3r (geometric mean): %.2f to %.2f" % (min(G("c")), max(G("c"))))
    print("planar restriction: %.2f to %.2f" % (min(G("a")), max(G("a"))))
    print("full Jacobian along the step against the true gain: mean |log ratio| at most %.4f (%.1f%%), worst loop %.1f%%"
          % (max(abs(k[1]["M_cb"]) for k in kept), 100 * (np.exp(max(abs(k[1]["M_cb"]) for k in kept)) - 1),
             100 * (np.exp(max(k[5] for k in kept)) - 1)))
    print("step in the plane at r: at least %.2f; at 2r and 3r: %.2f to %.2f" % (
        min(k[2]["k%d" % k[3]]["step_in_plane_med"] for k in kept), min(min(k[4]) for k in kept),
        max(max(k[4]) for k in kept)))


if __name__ == "__main__":
    main()
