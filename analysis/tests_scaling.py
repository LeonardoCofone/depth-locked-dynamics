import numpy as np
from scipy.stats import spearmanr

from analysis.common import (U, contraction_rate, depth, first_grid, has, is_fixed, lambda_profile, load, loss_curve,
                             minima, phase_period, step_profiles)

B = "fixed_r%d_b%d_s%d"
BASE = [[B % (8, 8, s) for s in range(3)], [B % (12, 12, 0)], [B % (16, 16, 0)]]


def cv_rows(rows):
    rows = np.array(rows)
    m = np.abs(rows.mean(0))
    ok = m >= 0.01
    return float(np.mean(rows.std(0)[ok] / m[ok])) if rows.ndim > 1 else float(rows.std() / abs(rows.mean()))


def profiles(groups):
    out = []
    for runs in groups:
        ps = [step_profiles(x) for x in runs]
        r = ps[0][0]
        n = np.mean([np.interp(U, (np.arange(r) + 0.5) / r, p[1]) for p in ps], 0)
        a = np.mean([np.interp(U, (np.arange(r - 1) + 1) / r, p[2]) for p in ps], 0)
        lam = np.mean([np.interp(U, (np.arange(1, r + 1) - 0.5) / r, lambda_profile(x)[:r]) for x in runs], 0)
        sums = [lambda_profile(x)[r // 2 - 1:r].sum() for x in runs]
        last4 = [lambda_profile(x)[r - 4:r].sum() for x in runs]
        out.append(dict(r=r, n=n, a=a, lam=lam, S=np.mean(sums), S4=np.mean(last4)))
    return out


def collapse(label, scaled, raw, yes=0.25, grey=0.40):
    cs, cr = cv_rows(scaled), cv_rows(raw)
    v = "YES" if cs <= yes and cs <= 0.5 * cr else ("grey" if cs <= grey and cs <= 0.5 * cr else "NO")
    print("%s: CV %.3f against %.3f unscaled -> %s" % (label, cs, cr, v))


def before_landing():
    P = profiles(BASE)
    collapse("turning angle before r as r*alpha(k/r)", [p["r"] * p["a"] for p in P], [p["a"] for p in P])
    collapse("step before r as r*n(k/r)", [p["r"] * p["n"] for p in P], [p["n"] for p in P])
    collapse("log-stretch before r as r*lambda(k/r)", [p["r"] * p["lam"] for p in P], [p["lam"] for p in P])
    c, c4 = cv_rows([p["S"] for p in P]), cv_rows([p["S4"] for p in P])
    v = "YES" if c <= 0.15 and c <= 0.5 * c4 else ("grey" if c <= 0.25 and c < c4 else "NO")
    print("log-stretch summed over loops r/2..r constant in r: CV %.3f against %.3f over the last four loops -> %s"
          % (c, c4, v))


def period_from_phase():
    rows = []
    for d in first_grid():
        if not is_fixed(d) or not has(d, "step_geometry.json") or depth(d) < 4:
            continue
        r = depth(d)
        L = loss_curve(d)
        m = minima(L, r)
        rows.append((d, r, contraction_rate(d), phase_period(load(d, "step_geometry.json")["post"]["cumphase_med"]),
                     m[0] - r if m else None))
    defined = [x for x in rows if x[3] is not None]
    f = len(defined) / len(rows)
    print("period from the phase defined: %d of %d -> %s" % (len(defined), len(rows), "YES" if f >= 0.8 else (
        "grey" if f >= 0.6 else "NO")))
    print("  undefined in %s" % [x[0] for x in rows if x[3] is None])
    base = [x for x in rows if x[0] in sum(BASE, []) or x[0] in [B % (4, 4, s) for s in range(3)]]
    v = [x[3] / x[1] for x in base if x[3]]
    c = np.std(v) / np.mean(v)
    print("phase period scales with r in the baseline: CV of P_Theta/r %.3f over %d models -> %s" % (
        c, len(v), "YES" if c <= 0.2 else ("grey" if c <= 0.3 else "NO")))
    rho = spearmanr([x[2] for x in defined], [x[3] / x[1] for x in defined])[0]
    print("q and P_Theta/r associated: Spearman %.2f over %d models -> %s" % (
        rho, len(defined), "YES" if rho >= 0.6 else ("grey" if rho >= 0.4 else "NO")))


def main():
    print("== Before landing (r = 8, 12, 16)")
    before_landing()
    print("\n== Period from the phase (first grid)")
    period_from_phase()


if __name__ == "__main__":
    main()
