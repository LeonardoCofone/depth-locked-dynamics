import cmath

import numpy as np

from analysis.common import first_grid, has, is_fixed, load

BASE_DEPTHS = (4, 8, 12, 16)


def verdict(frac, yes, no):
    return "YES" if frac >= yes else ("NO" if frac < no else "grey")


def base(models, r):
    return [k for k in models if k in ("fixed_r%d_b%d_s%d" % (r, r, s) for s in range(3))]


def decay_scale(fixed):
    xs, ys = [], []
    for r in BASE_DEPTHS:
        ks = base(fixed, r)
        xs.append(r)
        ys.append(float(np.mean([fixed[k]["D"]["decay"]["0.5"] for k in ks])))
    slope = float(np.polyfit(np.log(xs), np.log(1 - np.array(ys)), 1)[0])
    rising = all(y2 > y1 for y1, y2 in zip(ys, ys[1:]))
    v = "YES" if (-1.3 <= slope <= -0.7 and rising) else ("NO" if slope > -0.3 else "grey")
    print("decay carries the 1/r scale: slope of log(1 - decay) against log r %.2f -> %s" % (slope, v))


def exact_phase(fixed):
    win = total = 0
    errors = []
    for v in fixed.values():
        p1 = v["A"]["win"].get("w1", {}).get("P_theta")
        bk = v["B"].get("k%d" % v["r"], {}).get("full", {})
        if not p1 or not bk:
            continue
        mu = bk["rho"]["0.5"] * cmath.exp(1j * bk["theta"]["0.5"])
        for g in (0.5, 0.75):
            c = v["C"].get("relax_after_%g" % g, {})
            pg = c.get("win", {}).get("w1", {}).get("P_theta") if "error" not in c else None
            phase = cmath.phase(1 + g * (mu - 1))
            if not pg or phase <= 0:
                continue
            measured, predicted = pg * g / p1, g * bk["theta"]["0.5"] / phase
            total += 1
            win += abs(predicted - measured) < abs(1 - measured)
            errors.append(abs(predicted - measured))
    em = float(np.median(errors))
    v = "YES" if win / total >= 2 / 3 and em <= 0.08 else ("NO" if win / total < 0.5 else "grey")
    print("exact phase arg(1 + gamma(mu - 1)) beats 1: %d of %d, median error %.3f -> %s" % (win, total, em, v))


def doubled_step_stability(fixed):
    fail = unstable = 0
    for v in fixed.values():
        r = v["r"]
        if r < 8:
            continue
        c = v["C"].get("relax_all_rp%d" % (r // 2))
        bk = v["B"].get("k%d" % (r // 2), {}).get("full")
        if not c or "loss" not in c or not bk:
            continue
        failed = c["loss"][-1] > v["C"]["relax_after_0.5"]["loss"][r] + 0.1
        mu = bk["rho"]["0.5"] * cmath.exp(1j * bk["theta"]["0.5"])
        fail += failed
        unstable += failed and abs(2 * mu - 1) > 1
    v = "YES" if unstable / fail >= 2 / 3 else ("NO" if (fail - unstable) / fail >= 2 / 3 else "grey")
    print("doubled step unstable in the plane at r/2 where the halved loop fails: %d of %d -> %s" % (unstable, fail, v))


def continuous_limit(scaled):
    deep = {k: v for k, v in scaled.items() if v["r"] >= 8}
    hit = sum(abs(v["L_r"] - v["conv"]["rp%d" % (8 * v["r"])]["loss"]["1.0"]) <= 0.02 for v in deep.values())
    print("trained loop within 0.02 of its continuous limit (r' = 8r): %d of %d -> %s" % (
        hit, len(deep), verdict(hit / len(deep), 2 / 3, 0.5)))
    profiles = {}
    for r in (8, 12, 16):
        ps = []
        for k in base(scaled, r):
            c = scaled[k]["conv"]["rp%d" % (8 * r)]["curve"]
            n = len(c) - 1
            ps.append([(c[int(round(t * n))] - c[n]) / (c[0] - c[n]) for t in (0.25, 0.5, 0.75)])
        profiles[r] = np.mean(ps, axis=0)
    M = np.array(list(profiles.values()))
    dev = float(np.max(np.abs(M - M.mean(axis=0))))
    print("same limit curve for r = 8, 12, 16: max deviation %.3f -> %s" % (
        dev, "YES" if dev <= 0.05 else ("NO" if dev > 0.15 else "grey")))
    hit = sum(v["richardson"]["R_%d_%d" % (v["r"], 2 * v["r"])]["loss"] - v["L_r"] <= -0.01 for v in deep.values())
    print("Richardson extrapolation lowers the loss by >= 0.01: %d of %d -> %s" % (
        hit, len(deep), verdict(hit / len(deep), 2 / 3, 1 / 3)))
    cancel = none = 0
    for v in deep.values():
        r, sp = v["r"], v["step_parts"]
        loops = range(r // 2, r)
        ratio = float(np.median([sp["tot"][j] / sp["inj"][j] for j in loops]))
        cos = float(np.median([sp["cos"][j] for j in loops]))
        cancel += ratio <= 0.5 and cos <= -0.5
        none += ratio >= 0.9
    n = len(deep)
    print("small step from cancellation (net/first part <= 0.5, cosine <= -0.5): %d of %d -> %s" % (
        cancel, n, "YES" if cancel / n >= 2 / 3 else ("NO" if none / n >= 2 / 3 else "grey")))


def main():
    grid = [d for d in first_grid() if is_fixed(d)]
    plane = {d: load(d, "plane_dynamics.json") for d in grid if has(d, "plane_dynamics.json")}
    scaled = {d: load(d, "scaled_update.json") for d in grid if has(d, "scaled_update.json")}
    decay_scale(plane)
    exact_phase(plane)
    doubled_step_stability(plane)
    continuous_limit(scaled)


if __name__ == "__main__":
    main()
