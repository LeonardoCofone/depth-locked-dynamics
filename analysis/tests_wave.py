import itertools

import numpy as np
from scipy.stats import pearsonr, spearmanr

from analysis.common import WAVE15, contraction_rate, depth, first_grid, is_fixed, lambda_profile, load, loss_curve

SEEN = {"fixed_r8_b8_s0", "fixed_r4_b4_s0", "fixed_r16_b16_s0", "fixed_r8_b8_D1500M_s0", "poisson_r8_b4_s0"}
FIRST_POISSON = ["poisson_r8_b4_s%d" % s for s in range(3)] + ["poisson_r8_b4_muon_s%d" % s for s in range(3)]


def separated_minima(L, r):
    L = np.asarray(L)
    idx = [k for k in range(r, len(L) - 1) if L[k] <= L[k - 1] and L[k] <= L[k + 1]]
    keep = []
    for k in idx:
        if not keep:
            keep.append(k)
        elif k - keep[-1] >= 6 and L[keep[-1]:k + 1].max() - max(L[keep[-1]], L[k]) >= 0.01:
            keep.append(k)
    return keep


def full_fixed(name):
    return is_fixed(name) and depth(name) == int(name.split("_")[2][1:]) and "_d768" not in name


def oscillation_prediction():
    fixed, poisson = [], []
    for d in first_grid():
        if d in SEEN or "_d768" in d:
            continue
        L = loss_curve(d)
        r = depth(d)
        if full_fixed(d):
            fixed.append(len(separated_minima(L, r)) >= 2)
        elif not is_fixed(d):
            k = int(np.argmin(L))
            poisson.append(L[k:].max() - L.min() < 0.005)
    ok = "YES" if sum(fixed) / len(fixed) >= 0.8 and all(poisson) else "NO"
    print("oscillation: uninspected full-backprop fixed models with two loss minima: %d of %d -> %s;"
          " flat Poisson %d of %d" % (sum(fixed), len(fixed), ok, sum(poisson), len(poisson)))
    for small, large in (("fixed_r8_b8_s0", "fixed_r8_b8_D1500M_s0"), ("fixed_r16_b16_s0", "fixed_r16_b16_D1500M_s0")):
        a, b = [max(loss_curve(x)[depth(x):2 * depth(x) + 1]) - loss_curve(x)[depth(x)] for x in (small, large)]
        print("amplitude grows with data: %s %.3f -> %.3f" % (small, a, b))


def first_block():
    rows = {}
    for d in first_grid():
        r = depth(d)
        if not full_fixed(d) or r < 4:
            continue
        L = loss_curve(d)
        m = separated_minima(L, r)
        rows[d] = dict(r=r, P=m[1] - m[0] if len(m) >= 2 else None, A=L[r + 1:2 * r + 1].max() - L[r],
                       q=contraction_rate(d), brake=lambda_profile(d)[r - 1], L=L)
    family = [n for n in rows if n.startswith("fixed_r8_b8") and "_dec0.9" not in n and rows[n]["P"]]
    rho = spearmanr([rows[n]["q"] for n in family], [rows[n]["P"] for n in family]).correlation
    print("q and P associated within r = 8 (n = %d): Spearman %.2f -> %s" % (
        len(family), rho, "YES" if rho >= 0.7 else ("NO" if rho <= 0.3 else "grey")))
    names = list(rows)
    rho2 = spearmanr([rows[n]["brake"] for n in names], [rows[n]["A"] for n in names]).correlation
    print("brake and amplitude, all full-backprop fixed r >= 4 (n = %d): Spearman %.2f -> %s" % (
        len(names), rho2, "YES" if rho2 <= -0.6 else ("NO" if abs(rho2) < 0.3 else "grey")))
    at8 = [n for n in names if rows[n]["r"] == 8]
    rho8 = spearmanr([rows[n]["brake"] for n in at8], [rows[n]["A"] for n in at8]).correlation
    print("  exploratory, within r = 8 (n = %d): Spearman %.2f" % (len(at8), rho8))
    base = {4: "fixed_r4_b4_s0", 8: "fixed_r8_b8_s0", 12: "fixed_r12_b12_s0", 16: "fixed_r16_b16_s0"}
    u = np.linspace(0.05, 3, 40)
    curves = {}
    for r, d in base.items():
        L = rows[d]["L"]
        k = np.arange(len(L))
        y = L - L[r]
        curves[r] = (np.interp(r + u * r, k, y), np.interp(r + u * 4, k, y))
    pairs = list(itertools.combinations(base, 2))
    scaled = np.mean([pearsonr(curves[a][0], curves[b][0])[0] for a, b in pairs])
    absolute = np.mean([pearsonr(curves[a][1], curves[b][1])[0] for a, b in pairs])
    print("curves collapse in (k - r)/r: mean pairwise correlation %.2f against %.2f in loops -> %s" % (
        scaled, absolute, "YES" if scaled >= 0.8 and scaled >= absolute + 0.15 else "NO"))


def detrend(x, w):
    w = max(3, int(w))
    t = np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")
    return x - t


def state_wave():
    counts = dict(A=0, B=0, C=0, D=0, E=0)
    n = 0
    for d in WAVE15:
        j = load(d, "state_distance.json")
        r, K = j["r"], j["K"]
        L = np.array([np.nan] + j["loss_coda"])
        lens = np.array([np.nan] + j["loss_lens"])
        Dm = np.array(j["D"])
        Dr = np.array([np.nan] + list(Dm[r - 1]))
        m = separated_minima(L, r)
        if len(m) < 2:
            continue
        P = m[1] - m[0]
        n += 1
        dmin = [k for k in range(r + 1, K) if Dr[k] <= Dr[k - 1] and Dr[k] <= Dr[k + 1]]
        counts["A"] += any(abs(k - m[1]) <= 2 for k in dmin)
        maxima = [k for k in range(r + 1, K) if L[k] >= L[k - 1] and L[k] >= L[k + 1]]
        back = lambda k: min(Dm[k - 1][jj] for jj in range(r))
        counts["B"] += np.mean([back(k) for k in m[1:]]) < (np.mean([back(k) for k in maxima]) if maxima else np.nan)
        counts["C"] += (r + P <= K) and Dr[r + P] < Dr[r + P // 2]
        seg = slice(r + 1, K + 1)
        counts["D"] += pearsonr(L[seg], lens[seg])[0] >= 0.7
        x, y = detrend(L[seg], P), detrend(Dr[seg], P)
        best = 0.0
        for lag in range(-P // 2, P // 2 + 1):
            u, v = (x[lag:], y[:len(y) - lag]) if lag >= 0 else (x[:lag], y[-lag:])
            if len(u) > 5:
                c = pearsonr(u, v)[0]
                best = c if abs(c) > abs(best) else best
        counts["E"] += abs(best) >= 0.6
    labels = {"A": "second loss minimum on a minimum of |s_k - s_r|",
              "B": "closer to loops 1..r at loss minima than at maxima",
              "C": "closer to s_r after one period than after half",
              "E": "detrended loss and distance oscillate together (|corr| >= 0.6)",
              "D": "a linear readout of the state follows the loss (Pearson >= 0.7)"}
    for key in "ABCED":
        k = counts[key]
        v = "YES" if k >= 0.7 * n else ("NO" if key != "D" else "grey")
        print("state: %s: %d of %d -> %s" % (labels[key], k, n, v))
    pois = [np.max(np.array(load(d, "state_distance.json")["D"])[7][8:]) for d in FIRST_POISSON]
    print("Poisson state stays within 5%% of s_r: %d of %d" % (sum(p < 0.05 for p in pois), len(pois)))


def main():
    print("== Oscillation and amplitude")
    oscillation_prediction()
    print("\n== Period, contraction, brake and collapse")
    first_block()
    print("\n== Where the wave lives: distances between states")
    state_wave()


if __name__ == "__main__":
    main()
