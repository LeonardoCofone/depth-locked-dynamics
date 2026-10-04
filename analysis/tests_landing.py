import os

import numpy as np
from scipy.stats import spearmanr

from analysis.common import MODELS, U, depth, first_grid, has, is_fixed, lambda_profile, load, loss_curve, phase_period
from analysis.tests_decoder import wave_models
from analysis.tests_scaling import cv_rows
from analysis.tests_wave import FIRST_POISSON, full_fixed

LONG = ["fixed_r12_b12_s0", "fixed_r16_b16_s0", "fixed_r16_b16_D1500M_s0", "fixed_r8_b8_D1500M_s0"]
DEEP_FULL = (["fixed_r8_b8_s%d" % s for s in range(3)] + ["fixed_r12_b12_s0", "fixed_r16_b16_s0",
             "fixed_r8_b8_D1500M_s0", "fixed_r16_b16_D1500M_s0"] + ["fixed_r8_b8_muon_s%d" % s for s in range(3)]
             + ["fixed_r8_b8_dec0.7_s%d" % s for s in range(2)] + ["fixed_r8_b8_dec0.9_s%d" % s for s in range(2)])


def arc_strength(L, r):
    dL = L[:-1] - L[1:]
    seg = list(range(max(2, r // 4), r - 1))
    kmin = min(seg, key=lambda k: dL[k - 1])
    return kmin, dL[kmin - 1], dL[kmin:r].max()


def deep_full():
    return [d for d in first_grid() if full_fixed(d) and depth(d) >= 8]


def slow_phase():
    ok = n = 0
    for d in wave_models():
        j = load(d, "state_distance.json")
        r, K = j["r"], j["K"]
        D = np.array(j["D"])
        n += 1
        ok += float(np.median([int(np.argmin(D[k - 1][:r])) + 1 for k in range(3 * r, min(4 * r, K) + 1)])) <= r - 2
    print("state returns to the slow phase before the final arc: %d of %d" % (ok, n))
    hits = 0
    for d in LONG:
        r = depth(d)
        kmin, low, after = arc_strength(loss_curve(d), r)
        hits += after >= 1.5 * low and kmin < r - 1
    print("loss decrease accelerates before r (slow phase, then x1.5): %d of %d" % (hits, len(LONG)))


def arc_and_amplitude():
    xs, ys = [], []
    for d in deep_full():
        L = loss_curve(d)
        r = depth(d)
        _, low, after = arc_strength(L, r)
        xs.append(after / max(low, 1e-9))
        ys.append(L[r + 1:2 * r + 1].max() - L[r])
    rho = spearmanr(xs, ys).correlation
    print("amplitude follows the acceleration of the loss: Spearman %.2f (n = %d) -> %s" % (
        rho, len(xs), "YES" if rho >= 0.6 else ("NO" if rho < 0.3 else "grey")))
    step, excursion = [], []
    for d in wave_models():
        r = depth(d)
        step.append(load(d, "loop_decoder.json")["step"][r - 1])
        Dr = np.array(load(d, "state_distance.json")["D"])[r - 1]
        excursion.append(Dr[r:4 * r].max())
    rho = spearmanr(step, excursion).correlation
    print("excursion after r ranks with the step at loop r: Spearman %.2f (n = %d) -> %s" % (
        rho, len(step), "YES" if rho >= 0.6 else ("NO" if rho < 0.3 else "grey")))


def step_growth():
    ok_f = n_f = ok_p = 0
    for d in deep_full():
        st = np.array([np.nan] + load(d, "loop_decoder.json")["step"])
        r = depth(d)
        kmin = 2 + int(np.argmin(st[2:r]))
        n_f += 1
        ok_f += kmin < r and st[kmin:r + 1].max() / st[kmin] - 1 >= 0.10
    for d in FIRST_POISSON:
        st = np.array([np.nan] + load(d, "loop_decoder.json")["step"])
        seg = st[2:9]
        ok_p += max(seg[i + 1] / seg[i] - 1 for i in range(len(seg) - 1)) <= 0.05
    print("step grows before r in fixed models: %d of %d; Poisson monotone %d of %d"
          % (ok_f, n_f, ok_p, len(FIRST_POISSON)))


def non_normality():
    poisson = [load(d, "non_normality_coarse.json")["nu"]["7"] for d in FIRST_POISSON]
    fixed, force = [], []
    for d in deep_full():
        r = depth(d)
        fixed.append(load(d, "non_normality_coarse.json")["nu"][str(r - 1)])
        _, low, after = arc_strength(loss_curve(d), r)
        force.append(after / max(low, 1e-9))
    print("fixed more non-normal than every Poisson at r - 1: %d of %d"
          % (sum(v > max(poisson) for v in fixed), len(fixed)))
    print("non-normality at r - 1 follows the acceleration of the loss: Spearman %.2f (n = %d)" % (
        spearmanr(fixed, force).correlation, len(fixed)))
    rng = np.random.default_rng(0)

    def boot(parts, B=2000):
        p = np.array(parts)
        idx = rng.integers(0, len(p), (B, len(p)))
        return np.sqrt(p[idx, 0].sum(1) / p[idx, 1].sum(1))

    hits = 0
    for d in DEEP_FULL:
        j = load(d, "non_normality.json")
        r = j["r"]
        nu = {int(k): v for k, v in j["nu"].items()}
        ks = list(range(2, 2 * r + 1))
        k_star = ks[int(np.argmin([nu[k] for k in ks]))]
        near = abs(k_star - r) <= max(2, r / 4)
        inner = nu[k_star] <= nu[2 * r] - 0.05 and nu[k_star] <= nu[2] - 0.2
        sep = np.quantile(boot(j["parts"][str(k_star)]), 0.95) < np.quantile(boot(j["parts"][str(2 * r)]), 0.05)
        hits += near and inner and sep
    flat = 0
    for d in FIRST_POISSON:
        nu = np.array([load(d, "non_normality.json")["nu"][str(k)] for k in range(4, 17)])
        flat += nu.max() - nu.min() < 0.1
    print("non-normality minimum within max(2, r/4) of r: %d of %d; Poisson flat %d of %d" % (
        hits, len(DEEP_FULL), flat, len(FIRST_POISSON)))
    follows = 0
    group = DEEP_FULL + ["fixed_r8_b8_d768_s0"]
    for d in group:
        r = depth(d)
        nu = load(d, "non_normality.json")["nu"]
        lam = np.array(load(d, "lyapunov.json")["lyap"]).mean(1)
        ks = range(1, 2 * r)
        follows += spearmanr([nu[str(k)] for k in ks], [lam[k] for k in ks]).correlation >= 0.6
    print("non-normality follows the log-stretch loop by loop: %d of %d" % (follows, len(group)))


def truncated():
    b, A = [], []
    for d in [x for x in first_grid() if x.startswith(("fixed_r12_b8", "fixed_r16_b8"))]:
        r = depth(d)
        lam = lambda_profile(d)
        L = loss_curve(d)
        b.append(lam[r - 1] - np.median(lam[:r]))
        A.append(L[r + 1:2 * r + 1].max() - L[r])
    print("brake and amplitude in the truncated models: Spearman %+.2f (n = %d)" % (spearmanr(b, A)[0], len(b)))


def after_landing_collapse():
    groups = [["fixed_r8_b8_s%d" % s for s in range(3)], ["fixed_r12_b12_s0"], ["fixed_r16_b16_s0"]]
    raw, scaled = [], []
    for runs in groups:
        r = depth(runs[0])
        grid = (np.arange(1, 2 * r + 1) - 0.5) / r
        p = np.mean([np.interp(U + 0.8, grid, lambda_profile(x)[:2 * r]) for x in runs], 0)
        raw.append(p)
        scaled.append(r * p)
    cs, cr = cv_rows(scaled), cv_rows(raw)
    v = "YES" if cs <= 0.25 and cs <= 0.5 * cr else ("grey" if cs <= 0.40 and cs <= 0.5 * cr else "NO")
    print("log-stretch after r collapses as r*lambda(k/r): CV %.3f against %.3f -> %s" % (cs, cr, v))


def landing_velocity():
    rows = []
    for d in first_grid():
        if not is_fixed(d) or not has(d, "step_geometry.json") or depth(d) < 4:
            continue
        sg = load(d, "step_geometry.json")
        r = sg["r"]
        P = phase_period(sg["post"]["cumphase_med"])
        if not P:
            continue
        G = np.array(sg["G"])
        n = np.sqrt(np.diag(G))
        alpha = np.array([np.arccos(np.clip(G[k, k + 1] / (n[k] * n[k + 1]), -1, 1)) for k in range(r)])
        rows.append((d, 2 * np.pi / alpha[[r - 3, r - 2]].mean() / P))
    hits = sum(abs(x - 1) <= 0.3 for _, x in rows)
    base = [x for d, x in rows if d in ["fixed_r%d_b%d_s%d" % (r, r, s) for r in (4, 8, 12, 16) for s in range(3)]]
    print("period from the angular velocity at landing: %d of %d (median ratio %.2f in the baseline)" % (
        hits, len(rows), np.median(base)))


def held_out():
    for d, window in (("fixed_r8_b8_D1500M_s0", (7, 8, 9)), ("fixed_r16_b16_D1500M_s0", (15, 16, 17)),
                      ("fixed_r8_b8_dec0.9_s1", (7, 8, 9))):
        lam = lambda_profile(d)
        k = int(np.argmin(lam)) + 1
        print("held-out %s: lowest lambda at loop %d -> %s" % (d, k, k in window))
    for d in ("fixed_r8_b8_D1500M_s0", "fixed_r16_b16_D1500M_s0"):
        st = load(d, "curves.json")["step_norm_median"]
        r = depth(d)
        q = (st[r] / st[2]) ** (1 / (r - 2))
        print("contraction rate constant across data: %s q = %.3f, in [0.89, 0.94]: %s" % (d, q, 0.89 <= q <= 0.94))
    path = {}
    for d in ("fixed_r8_b8_s0", "fixed_r16_b16_s0", "fixed_r8_b8_D1500M_s0", "fixed_r16_b16_D1500M_s0"):
        sn = np.load(os.path.join(MODELS, d, "curves_tokens.npz"))["token_step_norm"].astype(np.float64)
        sn = sn.reshape(sn.shape[0], -1)
        path[d] = sum(float(sn[k - 1].mean()) for k in range(2, depth(d) + 1))
    ratio = path["fixed_r16_b16_D1500M_s0"] / path["fixed_r8_b8_D1500M_s0"]
    print("same path length at 1.5B tokens: ratio r=16 / r=8 %.2f (0.5B: %.2f) -> %s" % (
        ratio, path["fixed_r16_b16_s0"] / path["fixed_r8_b8_s0"], 0.85 <= ratio <= 1.15))


def main():
    print("== Before landing")
    slow_phase()
    arc_and_amplitude()
    step_growth()
    print("\n== Non-normality")
    non_normality()
    print("\n== Brake")
    truncated()
    held_out()
    print("\n== Rotation and contraction after landing")
    after_landing_collapse()
    landing_velocity()


if __name__ == "__main__":
    main()
