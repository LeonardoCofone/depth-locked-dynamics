import os

import numpy as np

from analysis.common import MODELS, cv, first_grid, has, is_fixed, load, phase_period, plane_angle

BASE_DEPTHS = (4, 8, 12, 16)
PATHS = ("noattn", "nomlp", "frozenP", "frozennorm")
PATH_NAMES = {"noattn": "attention", "nomlp": "MLP", "frozenP": "attention pattern", "frozennorm": "normalisation"}


def verdict(frac, yes, grey):
    return "YES" if frac >= yes else ("grey" if frac >= grey else "NO")


def period(v):
    return v["A"]["win"].get("w1", {}).get("P_theta")


def theta_star(v, mode="full"):
    return v["B"]["k%d" % (v["K"] - 1)][mode]["theta"]["0.5"]


def base_groups(fixed):
    return {r: [k for k in fixed if k in ("fixed_r%d_b%d_s%d" % (r, r, s) for s in range(3))] for r in BASE_DEPTHS}


def eight_against_four(fixed):
    ok = n = 0
    for k, v in fixed.items():
        if not has(k, "step_geometry.json") or v["r"] < 4:
            continue
        p4 = phase_period(load(k, "step_geometry.json")["post"]["cumphase_med"])
        if p4 is None:
            continue
        n += 1
        ok += period(v) is not None and abs(period(v) / p4 - 1) <= 0.15
    print("period from the phase on 8 sequences within 15%% of 4 sequences: %d of %d -> %s" % (
        ok, n, "YES" if ok / n >= 0.8 else "NO"))


def slowest_mode(fixed):
    rows = [v for v in fixed.values() if period(v)]
    hit = sum(abs(2 * np.pi / theta_star(v) / period(v) - 1) <= 0.30 for v in rows if theta_star(v) > 0)
    print("period from the slowest mode of the Jacobian at the fixed point: %d of %d -> %s" % (
        hit, len(rows), verdict(hit / len(rows), 0.7, 0.5)))


def fixed_point_angle(fixed, G):
    th = {r: np.mean([theta_star(fixed[k]) for k in G[r]]) for r in BASE_DEPTHS if G[r]}
    cs, cr = cv([r * th[r] for r in th]), cv([th[r] for r in th])
    print("rotation angle at the fixed point collapses as r*theta: CV %.3f against %.3f -> %s" % (
        cs, cr, "YES" if cs <= 0.2 and cs <= 0.5 * cr else ("grey" if cs <= 0.3 and cs <= 0.5 * cr else "NO")))


def paths(fixed, G):
    med = {c: float(np.median([theta_star(v) - theta_star(v, c) for v in fixed.values()])) for c in PATHS}
    src = max(med, key=med.get)
    halved = [theta_star(v, src) <= 0.5 * theta_star(v) for v in fixed.values()]
    print("switching off the %s path halves the angle at the fixed point: %d of %d -> %s" % (
        PATH_NAMES[src], sum(halved), len(halved), verdict(np.mean(halved), 0.7, 0.5)))
    removed = {r: np.mean([theta_star(fixed[k]) - theta_star(fixed[k], src) for k in G[r]])
               for r in BASE_DEPTHS if G[r]}
    c = cv([r * removed[r] for r in removed])
    print("angle removed with the same path collapses as r*dtheta: CV %.3f -> %s" % (
        c, "YES" if c <= 0.25 else ("grey" if c <= 0.35 else "NO")))


def relaxed_after(fixed):
    hit = n = 0
    for v in fixed.values():
        if not period(v):
            continue
        n += 1
        vals = []
        for g in (0.5, 0.75):
            c = v["C"].get("relax_after_%g" % g, {})
            pg = c.get("win", {}).get("w1", {}).get("P_theta") if "error" not in c else None
            vals.append(pg * g / period(v) if pg else None)
        hit += all(x is not None and 0.75 <= x <= 1.25 for x in vals)
    print("update scaled by gamma after r stretches the period by 1/gamma: %d of %d -> %s" % (
        hit, n, verdict(hit / n, 0.7, 0.5)))


def rescaled_depth(fixed):
    res = {"a": [], "b": [], "c": []}
    for v in fixed.values():
        if v["r"] < 8:
            continue
        r, L = v["r"], v["A"]["loss"]
        for tag, rp in (("a", 2 * r), ("b", r // 2), ("c", (3 * r) // 2)):
            c = v["C"].get("relax_all_rp%d" % rp, {})
            if "error" in c or "loss" not in c:
                res[tag].append(False)
                continue
            res[tag].append(c["loss"][rp] < L[rp] if tag == "b" else c["loss"][rp] <= L[r] + 0.02)
    labels = {"a": "update scaled by 1/2 from the start: loss at 2r within 0.02 of L_r",
              "b": "update scaled by 2 from the start: loss at r/2 below the trained loop",
              "c": "update scaled by 2/3 from the start: loss at 3r/2 within 0.02 of L_r"}
    for tag in "abc":
        print("%s: %d of %d -> %s" % (labels[tag], sum(res[tag]), len(res[tag]), verdict(np.mean(res[tag]), 0.7, 0.5)))


def frozen_attention(fixed):
    hit = n = 0
    for v in fixed.values():
        base = v["A"]["win"].get("w1", {}).get("cumphase_med")
        c = v["C"].get("freezeP_after", {})
        frozen = c.get("win", {}).get("w1", {}).get("cumphase_med") if "error" not in c else None
        if not base:
            continue
        n += 1
        hit += frozen is not None and frozen[-1] <= 0.5 * base[-1]
    print("freezing the attention weights after r halves the rotation: %d of %d -> %s" % (
        hit, n, verdict(hit / n, 0.7, 0.5)))


def landing_rotation(fixed):
    hit = n = 0
    for k, v in fixed.items():
        path = os.path.join(MODELS, k, "plane_dynamics_tokens.npz")
        if not period(v) or not os.path.exists(path):
            continue
        z = np.load(path)
        if "B_k%d_full" % v["r"] not in z:
            continue
        phi = np.nanmedian(plane_angle(z["B_k%d_full" % v["r"]].astype(np.float64)))
        n += 1
        hit += phi > 0 and abs(2 * np.pi / phi / period(v) - 1) <= 0.30
    print("period is one turn at the Jacobian's rotation at landing: %d of %d -> %s" % (
        hit, n, verdict(hit / n, 0.7, 0.5)))


def main():
    fixed = {d: load(d, "plane_dynamics.json") for d in first_grid() if is_fixed(d) and has(d, "plane_dynamics.json")}
    G = base_groups(fixed)
    eight_against_four(fixed)
    slowest_mode(fixed)
    fixed_point_angle(fixed, G)
    paths(fixed, G)
    relaxed_after(fixed)
    rescaled_depth(fixed)
    frozen_attention(fixed)
    landing_rotation(fixed)


if __name__ == "__main__":
    main()
