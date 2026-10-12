import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MODELS = os.path.join(ROOT, "models")
PUBLIC = os.path.join(ROOT, "public")
U = np.round(np.arange(0.3, 0.95, 0.1), 2)

CONTROL_GRID = (["fixed_r%d_b%d_s%d" % (r, r, s) for r in (6, 10, 20, 24) for s in (0, 1)]
                + ["fixed_r12_b12_s1", "fixed_r16_b16_s1"]
                + ["poisson_r8_b32_s%d" % s for s in (0, 1)] + ["fixed_r8_b4_s%d" % s for s in (0, 1)]
                + ["fixed_r8_b8_all_s%d" % s for s in (0, 1)] + ["poisson_r8_b4_dec0.9_s%d" % s for s in (0, 1)])
NEW_FIXED = CONTROL_GRID[:10]
CONTRACTION = ["fixed_r8_b8_D100M_s0", "poisson_r8_b32_us0.25_D100M_s0", "poisson_r8_b32_us0.1_D100M_s0",
               "poisson_r8_b32_dec0.95_dfz_D100M_s0", "fixed_r8_b8_all_us0.25_D100M_s0"]
CONTRACTION += ["poisson_r8_b32_us0.1_s%d" % s for s in (0, 1)] + ["fixed_r8_b8_us0.1_s%d" % s for s in (0, 1)]
CONTRACTION += ["poisson_r8_b32_us0.25_s%d" % s for s in (0, 1)] + ["fixed_r8_b8_us0.25_s%d" % s for s in (0, 1)]
SCALE = ["fixed_r8_b8_d1792_s0", "fixed_r8_b8_d1792_s1", "poisson_r8_b32_d1792_s0"]


def model_names():
    return sorted(d for d in os.listdir(MODELS) if os.path.isdir(os.path.join(MODELS, d)))


def paper_models():
    return [d for d in model_names() if d not in CONTRACTION and d not in SCALE]


def first_grid():
    return [d for d in paper_models() if d not in CONTROL_GRID]


def has(name, f):
    return os.path.exists(os.path.join(MODELS, name, f))


def load(name, f):
    return json.load(open(os.path.join(MODELS, name, f)))


def load_public(name, f):
    return json.load(open(os.path.join(PUBLIC, name, f)))


def depth(name):
    return int(name.split("_")[1][1:])


def bptt(name):
    return int(name.split("_")[2][1:])


def is_fixed(name):
    return name.startswith("fixed")


def is_base_full(name):
    return (is_fixed(name) and depth(name) == bptt(name)
            and not any(t in name for t in ("_D", "_dec", "_muon", "_d768", "_all", "_us", "_dfz")))


def loss_curve(name):
    return np.array([e["loss"] for e in load(name, "curves.json")["per_k"]])


def contraction_rate(name):
    c = load(name, "curves.json")
    r, st = c["r"], c["step_norm_median"]
    return float((st[r] / st[2]) ** (1 / (r - 2))) if r > 2 else None


def lambda_profile(name):
    return np.array(load(name, "lyapunov.json")["lyap"]).mean(1)


def minima(L, r):
    K = min(8 * r, len(L) - 2)
    loc = [k for k in range(r + 1, K + 1) if L[k] < L[k - 1] and L[k] <= L[k + 1]]
    out = []
    for k in loc:
        if not out:
            out.append(k)
        elif k - out[-1] >= 6 and L[out[-1]:k + 1].max() - max(L[out[-1]], L[k]) >= 0.01:
            out.append(k)
    return out


def phase_period(cum, turns=1):
    cum = np.array(cum)
    for j in range(1, len(cum)):
        if cum[j] >= 2 * np.pi * turns:
            return float(j - 1 + (2 * np.pi * turns - cum[j - 1]) / (cum[j] - cum[j - 1]))
    return None


def rise(L, r):
    return float(L[r + 1:2 * r + 1].max() - L[r])


def brake_of(lam, r):
    am = int(np.argmin(lam[:min(len(lam), 3 * r)])) + 1
    rel = float(lam[r - 1] - np.median(lam[max(0, r // 2 - 1):3 * r // 2]))
    return am, rel


def plane_angle(B):
    tr = B[:, 0, 0] + B[:, 1, 1]
    det = B[:, 0, 0] * B[:, 1, 1] - B[:, 0, 1] * B[:, 1, 0]
    c = np.clip(tr / (2 * np.sqrt(np.clip(det, 1e-30, None))), -1, 1)
    return np.where(det > 0, np.arccos(c), np.nan)


def step_profiles(name):
    g = load(name, "step_geometry.json")
    r = g["r"]
    G = np.array(g["G"])
    n = np.sqrt(np.diag(G))[:r]
    alpha = np.array([np.arccos(np.clip(G[k, k + 1] / (n[k] * np.sqrt(G[k + 1, k + 1])), -1, 1)) for k in range(r - 1)])
    return r, n, alpha


def two_seeds(k, n):
    if n == 0:
        return "not testable"
    if n == 1:
        return "incomplete"
    if n == 2:
        return {2: "YES", 1: "grey", 0: "NO"}[k]
    return "YES" if k / n >= 2 / 3 else ("NO" if k == 0 else "grey")


def share(k, n, yes):
    return ("YES" if n and k / n >= yes else "NO") + " (%d of %d)" % (k, n)


def cv(v):
    v = np.array(v, dtype=float)
    return float(np.std(v) / abs(np.mean(v)))


def summary(name):
    out = {"name": name}
    c = load(name, "curves.json")
    r = c["r"]
    L = np.array([e["loss"] for e in c["per_k"]])
    m = minima(L, r)
    out.update(r=r, L=L, minima=m, osc=len(m) >= 2, rise=rise(L, r), q=contraction_rate(name))
    if has(name, "lyapunov.json"):
        lam = lambda_profile(name)
        am, rel = brake_of(lam, r)
        out.update(lam=lam, am=am, rel=rel, brake=abs(am - r) <= 1 and rel <= -0.05, argmin_ok=abs(am - r) <= 1)
        out["lam_u"] = np.interp(U, (np.arange(1, len(lam) + 1) - 0.5) / r, lam)
    if has(name, "step_geometry.json"):
        out["PT_sg"] = phase_period(load(name, "step_geometry.json")["post"]["cumphase_med"]) if r >= 4 else None
        _, n, alpha = step_profiles(name)
        out["n_u"] = np.interp(U, (np.arange(r) + 0.5) / r, n)
        out["al_u"] = np.interp(U, (np.arange(r - 1) + 1) / r, alpha)
    if has(name, "plane_dynamics.json"):
        v = load(name, "plane_dynamics.json")
        out["PT_pd"] = v["A"]["win"].get("w1", {}).get("P_theta")
        La = np.array(v["A"]["loss"])
        key = "relax_all_rp%d" % (2 * r)
        if key in v["C"] and "loss" in v["C"][key]:
            out["scaled2r"] = float(v["C"][key]["loss"][2 * r] - La[r])
            out["trained2r"] = float(La[2 * r] - La[r])
    if has(name, "plane_dynamics_tokens.npz"):
        z = np.load(os.path.join(MODELS, name, "plane_dynamics_tokens.npz"))
        if "B_k%d_full" % r in z:
            ph = {mo: float(np.nanmedian(plane_angle(z["B_k%d_%s" % (r, mo)].astype(np.float64))))
                  for mo in ("full", "noattn", "nomlp", "decayonly")}
            out.update(phi=ph["full"], b=r * ph["full"], noattn=ph["noattn"] / ph["full"],
                       nomlp=ph["nomlp"] / ph["full"], phi_decay=ph["decayonly"])
    return out


def summaries(names=None):
    return {n: summary(n) for n in (names if names is not None else model_names()) if has(n, "curves.json")}


WAVE15 = (["fixed_r4_b4_s%d" % s for s in range(3)] + ["fixed_r8_b8_s%d" % s for s in range(3)]
          + ["fixed_r12_b12_s0", "fixed_r16_b16_s0", "fixed_r8_b8_D1500M_s0", "fixed_r16_b16_D1500M_s0"]
          + ["fixed_r8_b8_muon_s%d" % s for s in range(3)] + ["fixed_r8_b8_dec0.7_s%d" % s for s in range(2)])
