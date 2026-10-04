import numpy as np

from analysis.common import depth, first_grid, has, load
from analysis.tests_wave import FIRST_POISSON, full_fixed, separated_minima

EDITED = (["fixed_r8_b8_s%d" % s for s in range(3)] + ["fixed_r8_b8_D1500M_s0", "fixed_r12_b12_s0", "fixed_r16_b16_s0",
          "fixed_r16_b16_D1500M_s0"] + ["fixed_r8_b8_muon_s%d" % s for s in range(3)]
          + ["fixed_r8_b8_dec0.7_s%d" % s for s in range(2)])


def wave_models():
    out = []
    for d in first_grid():
        r = depth(d)
        if not full_fixed(d) or r < 4 or not has(d, "loop_decoder.json"):
            continue
        L = np.array([np.nan] + load(d, "state_distance.json")["loss_coda"])
        if len(separated_minima(L, r)) >= 2:
            out.append(d)
    return out


def decoder():
    rows = []
    for d in wave_models():
        j = load(d, "loop_decoder.json")
        r, K = j["r"], j["K"]
        kh = np.array([np.nan] + j["khat_mean"])
        drop = np.nanmax(kh[r:]) - np.nanmin(kh[r + int(np.nanargmax(kh[r:])):])
        L = np.array([np.nan] + load(d, "state_distance.json")["loss_coda"])
        loss_minima = separated_minima(L, r)
        kh_minima = [k for k in range(r + 1, K) if kh[k] <= kh[k - 1] and kh[k] <= kh[k + 1]]
        k4 = any(abs(a - b) <= 2 for a in kh_minima for b in loss_minima[1:])
        rows.append(dict(r2=j["r2"], drop=round(drop / r, 2), k4=k4))
    n = len(rows)
    print("loop decoder R^2 >= 0.9: %d of %d" % (sum(x["r2"] >= 0.9 for x in rows), n))
    print("decoded index restarts (drops by >= r/4 after r): %d of %d" % (sum(x["drop"] >= 0.25 for x in rows), n))
    print("minima of the decoded index match the loss minima: %d of %d" % (sum(x["k4"] for x in rows), n))
    fixed = [load(d, "loop_decoder_k4.json")["r2"] for d in first_grid()
             if full_fixed(d) and depth(d) >= 4 and has(d, "loop_decoder_k4.json")]
    poisson = [load(d, "loop_decoder_k4.json")["r2"] for d in FIRST_POISSON]
    gap = np.mean(fixed) - np.mean(poisson)
    print("decoder on loops 1..4: fixed %.2f against Poisson %.2f -> %s" % (
        np.mean(fixed), np.mean(poisson), "YES" if gap >= 0.3 else ("NO" if gap < 0.1 else "grey")))


def step_profile_repeats():
    hits_f = hits_p = 0
    names = wave_models()
    for d in names + FIRST_POISSON:
        j = load(d, "loop_decoder.json")
        r = j["r"]
        p = np.diff(np.log(np.clip(np.array(j["step"]), 1e-12, None)))
        cs = [np.corrcoef(p[0:r - 1], p[r + s:2 * r - 1 + s])[0, 1] for s in range(0, r + 1) if 2 * r - 1 + s <= len(p)]
        hit = cs[0] >= np.percentile(cs[1:], 90) and cs[0] >= 0.5
        if d in FIRST_POISSON:
            hits_p += hit
        else:
            hits_f += hit
    print("step profile repeats with period r: fixed %d of %d, Poisson %d of %d" % (
        hits_f, len(names), hits_p, len(FIRST_POISSON)))


def temporal_edit():
    moved = still = 0
    random_still = []
    for d in EDITED:
        j = load(d, "temporal_edit.json")
        r, k0, D = j["r"], j["k0"], j["D"]
        land = lambda Lc: k0 + 1 + int(np.argmin(Lc[: 2 * r - k0]))
        l0 = land(j["clean"])
        shift = {(e["dir"], e["sign"]): land(e["loss"]) - l0 for e in j["edits"]}
        moved += shift[("coord", 1)] <= -D / 2 and shift[("coord", -1)] >= D / 2
        still += abs(shift[("coord", 1)]) < D / 4 and abs(shift[("coord", -1)]) < D / 4
        random_still += [abs(v) < D / 4 for (direction, _), v in shift.items() if direction != "coord"]
    n = len(EDITED)
    print("editing the temporal coordinate moves the landing: %d of %d (no shift in %d of %d)" % (moved, n, still, n))
    print("random edits of the same size leave the landing in place: %d of %d" % (sum(random_still), len(random_still)))


def main():
    print("== Loop decoder")
    decoder()
    step_profile_repeats()
    print("\n== Editing the decoded loop index")
    temporal_edit()


if __name__ == "__main__":
    main()
