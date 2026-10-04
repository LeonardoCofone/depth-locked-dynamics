import numpy as np

from analysis.common import load_public

RANDOM_DEPTH = ["huginn-0125", "parcae-1.3b", "parcae-140m", "parcae-370m", "parcae-770m"]


def random_depth():
    totals = [0, 0, 0, 0]
    for name in RANDOM_DEPTH:
        j = load_public(name, "public_dynamics.json")
        r = j["r"]
        L = np.array(j["loss"])
        lam = np.array(j["lyap"]).mean(1)
        D = np.array(j["dist_from_sr"])
        nu = {int(k): v for k, v in j["nu"].items()}
        ks = np.arange(2, min(2 * r, len(lam)) + 1)
        am = ks[np.argmin(lam[ks - 1])]
        rel = lam[r - 1] - np.median(lam[r // 2 - 1: 3 * r // 2])
        no_brake = not (abs(am - r) <= 1 and rel <= -0.05)
        excursion = L[r:].max() - L[r - 1:].min()
        dmax = D[r + 1:].max()
        window = [nu[k] for k in nu if r // 2 <= k <= 3 * r // 2]
        nu_range = max(window) - min(window)
        checks = (no_brake, excursion < 0.02, dmax < 0.05, nu_range < 0.1)
        totals = [t + c for t, c in zip(totals, checks)]
        print("  %-12s r=%d relative brake %+.3f, loss moves %.4f, state moves %.4f, nu range %.3f" % (
            name, r, rel, excursion, dmax, nu_range))
    labels = ("no brake at the mean depth", "loss after the mean depth moves < 0.02",
              "state within 5% of the state at the mean depth", "non-normality varies < 0.1 around the mean depth")
    for label, t in zip(labels, totals):
        print("%s: %d of 5 -> %s" % (label, t, "YES" if t >= 4 else "NO"))


def ouro():
    counts = [0, 0, 0, 0, 0]
    for name in ("Ouro-1.4B", "Ouro-2.6B"):
        j = load_public(name, "ouro_dynamics.json")
        r = j["r"]
        L = np.concatenate([[np.nan], j["loss"]])
        lam = np.array(j["lyap"]).mean(1)
        nu = {int(k): v for k, v in j["nu"].items()}
        am = int(np.argmin(lam[:12])) + 1
        rel = lam[r - 1] - np.median(lam[r // 2 - 1: 3 * r // 2])
        lowest_loss = int(np.nanargmin(L[1:17])) + 1
        dips = [k for k in range(9, 14) if L[k] < L[k - 1] and L[k] <= L[k + 1] and np.nanmax(L[r:k]) - L[r] >= 0.01]
        k_nu = min(range(2, 9), key=lambda k: nu[k])
        checks = (am in (3, 4, 5), lowest_loss in (3, 4, 5) and L[8] > L[4], bool(dips),
                  2 < k_nu < 8 and abs(k_nu - 4) <= 2 and nu[k_nu] <= nu[8] - 0.05, j["post"]["e2_med"] >= 0.8)
        counts = [c + v for c, v in zip(counts, checks)]
        print("  %s: lowest lambda at loop %d (relative brake %+.3f); lowest loss at loop %d, L_8 - L_4 %+.3f; local"
              " minimum in loops 9..13: %s; lowest nu in 2..8 at loop %d; two modes after r carry %.2f" % (
                  name, am, rel, lowest_loss, L[8] - L[4], ["%d (dip %.3f)" % (k, L[k - 1] - L[k]) for k in dips],
                  k_nu, j["post"]["e2_med"]))
    labels = ("brake within one loop of 4", "loss lowest at 3 to 5 and higher at 8", "loss minimum in loops 9 to 13",
              "non-normality minimum within 2 loops of 4", "steps after 4 planar (two modes >= 80%)")
    for label, c in zip(labels, counts):
        print("%s: %d of 2" % (label, c))


def main():
    print("== Public random-depth models (Parcae 140M to 1.3B, Huginn 3.5B)")
    random_depth()
    print("\n== Ouro (fixed depth 4, loss at every loop)")
    ouro()


if __name__ == "__main__":
    main()
