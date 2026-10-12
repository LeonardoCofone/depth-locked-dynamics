import numpy as np

from analysis.common import depth, first_grid, is_fixed, load, loss_curve
from analysis.tests_wave import FIRST_POISSON, separated_minima


def complex_ritz(v):
    return [complex(a, b) for (a, b), res in zip(v["ritz"], v["res"]) if b > 1e-9 and res < 0.05 * abs(complex(a, b))]


def bridge():
    hit = n = 0
    for d in first_grid():
        r = depth(d)
        if not is_fixed(d) or r < 4:
            continue
        m = separated_minima(loss_curve(d), r)
        if len(m) < 2:
            continue
        P = m[1] - m[0]
        z = complex_ritz(load(d, "ritz.json")["k"][str(2 * r)])
        theta = abs(np.angle(max(z, key=abs))) if z else None
        n += 1
        ok = theta is not None and abs(theta / (2 * np.pi / P) - 1) <= 0.30
        hit += ok
        print("  %-26s P %2d  phase of the leading complex Ritz pair at 2r %s, 2 pi / P %.2f" % (
            d, P, "-" if theta is None else "%.2f" % theta, 2 * np.pi / P))
    print("phase of the leading complex Ritz pair at 2r within 30%% of 2 pi / P: %d of %d -> %s" % (
        hit, n, "YES" if hit / n >= 0.7 else "NO"))


def poisson():
    top = []
    for d in FIRST_POISSON:
        z = [c for v in load(d, "ritz.json")["k"].values() for c in complex_ritz(v)]
        top.append(max(abs(c) for c in z) if z else 0.0)
    k = sum(t <= 0.9 for t in top)
    print("Poisson: no complex Ritz value with |mu| > 0.9: %d of %d, largest %.2f to %.2f -> %s" % (
        k, len(top), min(top), max(top), "YES" if k == len(top) else "NO"))


def main():
    print("== Local spectrum of the loop (Arnoldi, Ritz values with residual below 5% of their modulus)")
    bridge()
    poisson()


if __name__ == "__main__":
    main()
