import numpy as np

from analysis.common import WAVE15, first_grid, has, is_fixed, load, loss_curve, minima


def verdict(n, N, hi=2 / 3, lo=1 / 3):
    f = n / N
    return "YES" if f >= hi else ("NO" if f <= lo else "grey")


def doubled_update():
    rows = []
    for d in first_grid():
        if not has(d, "doubled_update.json"):
            continue
        j = load(d, "doubled_update.json")
        r, V, Lr = j["r"], j["variants"], j["trained"]["L_r"]
        rows.append((Lr, V["double_from2"]["L_arrive"], V["double_from2"]["L_trained_same_loop"],
                     V["double_from0"]["loss"][r // 2]))
    N = len(rows)
    n1 = sum(arrive <= Lr + 0.10 for Lr, arrive, _, _ in rows)
    n2 = sum(arrive < same for _, arrive, same, _ in rows)
    n3 = sum(arrive < from0 for _, arrive, _, from0 in rows)
    print("update doubled after two trained loops, loss at t = 1 within 0.10 of L_r: %d of %d -> %s"
          % (n1, N, verdict(n1, N)))
    print("same, below the trained loop stopped at the same loop: %d of %d -> %s" % (n2, N, verdict(n2, N)))
    print("same, below the update doubled from the start: %d of %d -> %s" % (n3, N, verdict(n3, N)))


def downstream():
    fixed, poisson = [], []
    for d in first_grid():
        if not has(d, "downstream.json"):
            continue
        j = load(d, "downstream.json")
        r = j["r"]
        A = np.array(j["trained"]["mean"])
        if j["fixed"]:
            L = loss_curve(d)
            peak = r + 1 + int(np.argmax(L[r + 1:2 * r + 1]))
            m = minima(L, r)
            low = m[0] if m else None
            S = np.array(j["scaled_half"]["mean"])
            fixed.append(dict(d=d, r=r, A_r=A[r - 1], A_peak=A[peak - 1],
                              A_low=A[low - 1] if low and low <= len(A) else None, S_2r=S[2 * r - 1]))
        else:
            poisson.append(A[7:32].max() - A[7:32].min())
    deep = [x for x in fixed if x["r"] >= 8]
    n1 = sum(x["A_r"] - x["A_peak"] >= 1.0 for x in deep)
    print("accuracy at the loss peak in (r, 2r] at least one point below loop r: %d of %d -> %s" % (
        n1, len(deep), verdict(n1, len(deep))))
    wave = [x for x in fixed if x["d"] in WAVE15]
    n2 = sum(x["A_low"] is not None and x["A_low"] >= x["A_peak"] + 0.5 for x in wave)
    print("accuracy at the next loss minimum 0.5 points above the peak: %d of %d -> %s"
          % (n2, len(wave), verdict(n2, len(wave))))
    n3 = sum(x["S_2r"] >= x["A_r"] - 1.0 for x in deep)
    print("update scaled by 1/2 keeps the accuracy at 2r within one point of loop r: %d of %d -> %s" % (
        n3, len(deep), verdict(n3, len(deep))))
    n4 = sum(x <= 1.0 for x in poisson)
    print("Poisson accuracy varies by at most one point over loops 8 to 32: %d of %d -> %s" % (
        n4, len(poisson), "YES" if n4 >= 5 else "NO"))


def scaled_states():
    rows = []
    for d in first_grid():
        if not (is_fixed(d) and has(d, "scaled_update.json") and has(d, "plane_dynamics.json")):
            continue
        su, pd = load(d, "scaled_update.json"), load(d, "plane_dynamics.json")
        r = su["r"]
        if r < 8:
            continue
        D = su["conv"]["diff_%d_%d" % (r, 2 * r)]["1.0"]["0.5"]
        E = pd["A"]["dist_med"][2 * r]
        S = pd["A"]["step_med"][r - 1]
        P1 = pd["A"]["win"].get("w1", {}).get("P_theta")
        c = pd["C"].get("relax_all_rp%d" % (2 * r), {})
        P2 = c.get("win", {}).get("w1", {}).get("P_theta") if "win" in c else None
        rows.append((D, E, S, (P2 / (2 * r)) / (P1 / r) if P1 and P2 else None))
    n = len(rows)
    k1 = sum(D < E for D, E, _, _ in rows)
    k2 = sum(D <= S for D, _, S, _ in rows)
    ph = [x for _, _, _, x in rows if x is not None]
    k3 = sum(0.75 <= x <= 1.25 for x in ph)
    v = lambda k, n: "YES" if k / n >= 0.7 else ("NO" if k / n < 0.5 else "grey")
    print("update scaled by 1/2: state at 2r closer to s_r than the trained loop at 2r: %d of %d -> %s"
          % (k1, n, v(k1, n)))
    print("same: state at 2r within one trained step of s_r: %d of %d -> %s" % (k2, n, v(k2, n)))
    print("same: period after landing in units of r' within 25%% of the trained one: %d of %d -> %s" % (
        k3, len(ph), v(k3, len(ph))))


def main():
    print("== Update doubled after the first loops")
    doubled_update()
    print("\n== Downstream tasks, loop by loop")
    downstream()
    print("\n== States of the scaled loop")
    scaled_states()


if __name__ == "__main__":
    main()
