import os

import numpy as np
from scipy.stats import spearmanr

from analysis.common import MODELS, WAVE15, load


def first_minimum(L, r):
    for k in range(r + 1, len(L)):
        if L[k - 1] < L[k - 2] and L[k - 1] <= L[k]:
            return k - r
    return None


def planar_rotation():
    g1 = g2 = g3 = 0
    for d in WAVE15:
        j = load(d, "step_geometry.json")
        r = j["r"]
        P = first_minimum(j["loss_coda"], r)
        cum = j["post"]["cumphase_med"]
        turn = cum[P] if P is not None and P < len(cum) else float("nan")
        before = j["pre"]["cumphase_med"][-1]
        predicted = 2 * np.pi * r / before if before > 0 else float("nan")
        g1 += j["post"]["e2_med"] >= 0.8
        g2 += P is not None and abs(turn / (2 * np.pi) - 1) <= 0.3
        g3 += P is not None and abs(predicted / P - 1) <= 0.3
    n = len(WAVE15)
    print("steps after r planar (two modes >= 80%% of the energy): %d of %d" % (g1, n))
    print("first loss minimum after one full turn of the step: %d of %d" % (g2, n))
    print("period from the angular velocity before r: %d of %d" % (g3, n))


def damping():
    predicted, measured = [], []
    for d in WAVE15:
        g = load(d, "step_geometry.json")
        r = g["r"]
        n = np.sqrt(np.diag(np.array(g["G"])))
        post = n[r:3 * r]
        rho = np.exp(np.polyfit(np.arange(len(post)), np.log(post), 1)[0])
        P = first_minimum(g["loss_coda"], r)
        w = load(d, "state_distance.json")
        K = w["K"]
        D = np.array(w["D"])[r - 1]
        if P is None:
            continue
        floor = D[K - r:].mean()
        maxima = []
        for k in range(r + 1, K - 1):
            if D[k - 1] > D[k - 2] and D[k - 1] >= D[k] and (not maxima or k - maxima[-1] >= P / 2):
                maxima.append(k)
        if len(maxima) < 2:
            continue
        measured.append((D[maxima[1] - 1] - floor) / (D[maxima[0] - 1] - floor))
        predicted.append(rho ** P)
    print("damping of the state oscillation per period equals rho^P: Spearman %.2f (n = %d)" % (
        spearmanr(predicted, measured)[0], len(predicted)))


def quadratic_landing():
    ok = no = 0
    for d in WAVE15:
        w = load(d, "state_distance.json")
        r, K = w["r"], w["K"]
        L = np.array(w["loss_coda"])
        D = np.array(w["D"])[r - 1]
        ks = np.arange(r + 1, min(4 * r, K) + 1)
        y, x = L[ks - 1] - L[r - 1], D[ks - 1] ** 2
        c = (x @ y) / (x @ x)
        r2 = 1 - ((y - c * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ok += r2 >= 0.8
        no += r2 <= 0.5
    print("loss after r quadratic in the displacement (R^2 >= 0.8): %d of %d (R^2 <= 0.5 in %d)"
          % (ok, len(WAVE15), no))


def per_token():
    t1 = t2 = t2_80 = 0
    for d in WAVE15:
        z = np.load(os.path.join(MODELS, d, "curves_tokens.npz"))
        r = load(d, "curves.json")["r"]
        Lt = z["token_loss"].astype(np.float64).reshape(z["token_loss"].shape[0], -1)
        w = Lt[r + 1:4 * r + 1] - Lt[r]
        W = w.mean(1)
        kp = int(np.argmax(W[:r]))
        wc, Wc = w - w.mean(0), W - W.mean()
        den = np.sqrt((wc ** 2).sum(0) * (Wc ** 2).sum())
        ok = den > 0
        corr = (wc[:, ok] * Wc[:, None]).sum(0) / den[ok]
        c = w[kp]
        share = np.sort(c)[::-1][: len(c) // 10].sum() / c.sum()
        t1 += np.median(corr) >= 0.5
        t2 += share <= 0.5
        t2_80 += share >= 0.8
    n = len(WAVE15)
    print("per token: median correlation with the average oscillation >= 0.5: %d of %d" % (t1, n))
    print("per token: top 10%% of tokens carry <= 50%% of the amplitude: %d of %d (>= 80%%: %d of %d)"
          % (t2, n, t2_80, n))


def main():
    print("== Geometry of the steps after r")
    planar_rotation()
    damping()
    quadratic_landing()
    print("\n== Per token")
    per_token()


if __name__ == "__main__":
    main()
