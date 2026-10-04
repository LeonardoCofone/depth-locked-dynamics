import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from analysis.common import load, load_public, loss_curve

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
RUN = {"r4": "fixed_r4_b4_s0", "r8": "fixed_r8_b8_s0", "r12": "fixed_r12_b12_s0", "r16": "fixed_r16_b16_s0",
       "r8_15": "fixed_r8_b8_D1500M_s0", "r16_15": "fixed_r16_b16_D1500M_s0", "pois": "poisson_r8_b4_s0"}
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e6ea"
DEPTHS = (6, 8, 10, 12, 16, 20, 24)
DCOL = dict(zip(DEPTHS, plt.cm.viridis(np.linspace(0.92, 0.08, len(DEPTHS)))))
C = {"r4": plt.cm.viridis(1.0), "r8": DCOL[8], "r12": DCOL[12], "r16": DCOL[16], "pois": "#7a8794"}
NAME = {"r4": "fixed $r{=}4$", "r8": "fixed $r{=}8$", "r12": "fixed $r{=}12$", "r16": "fixed $r{=}16$",
        "pois": "Poisson (mean 8)"}
RECIPES = [
    ("fixed, through every loop", ["fixed_r8_b8_s%d" % s for s in range(3)], DCOL[8], "-"),
    ("loss at every loop", ["fixed_r8_b8_all_s%d" % s for s in range(2)], "#8e5bc9", "-"),
    ("Poisson, through every loop", ["poisson_r8_b32_s%d" % s for s in range(2)], INK, "--"),
    ("Poisson, truncated to 4", ["poisson_r8_b4_s%d" % s for s in range(3)], C["pois"], "--"),
]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6, "legend.frameon": False,
    "pdf.fonttype": 42, "font.family": "serif",
    "figure.constrained_layout.w_pad": 0.09, "figure.constrained_layout.h_pad": 0.06,
})


def lam(run):
    return np.array(load(run, "lyapunov.json")["lyap"]).mean(1)


def base_runs(r):
    return ["fixed_r%d_b%d_s%d" % (r, r, s) for s in ((0, 1, 2) if r == 8 else (0, 1))]


def step_profiles(run):
    j = load(run, "step_geometry.json")
    r = j["r"]
    G = np.array(j["G"])
    n = np.sqrt(np.diag(G))
    alpha = np.array([np.arccos(np.clip(G[k, k + 1] / (n[k] * n[k + 1]), -1, 1)) for k in range(r - 1)])
    return r, n[:r], alpha, np.array(j["post"]["cumphase_med"])


def check_bounds(fig, name, margin=0.012):
    fig.canvas.draw()
    R = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    m = margin * W
    bad = []
    boxes = [ax.get_tightbbox(R) for ax in fig.axes]
    for i, b in enumerate(boxes):
        if b.x0 < m or b.x1 > W - m or b.y0 < 0 or b.y1 > H:
            bad.append("panel %d outside the figure" % i)
    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if (i < j and b1.overlaps(b2) and min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1
                    and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1):
                bad.append("panels %d and %d overlap" % (i, j))
    titles = [ax.title.get_window_extent(R) for ax in fig.axes if ax.get_title()]
    for t1 in titles:
        for t2 in titles:
            if t1 is not t2 and abs(t1.y0 - t2.y0) < 2 and t1.x0 < t2.x0 and t1.x1 > t2.x0 - m:
                bad.append("titles touch")
    print(name, "OK" if not bad else "CUT: %s" % bad)
    fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"CreationDate": None})


def fig_profile():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.3), constrained_layout=True)
    for r in DEPTHS:
        ns = np.array([step_profiles(x)[1] for x in base_runs(r)])
        u = (np.arange(r) + 0.5) / r
        axs[0].plot(u, ns.mean(0), color=DCOL[r], lw=1.4, marker="o", ms=2.2, label="$r{=}%d$" % r)
        axs[1].fill_between(u, r * ns.min(0), r * ns.max(0), color=DCOL[r], alpha=0.18, lw=0)
        axs[1].plot(u, r * ns.mean(0), color=DCOL[r], lw=1.4, marker="o", ms=2.2, label="$r{=}%d$" % r)
    axs[0].set_ylabel("$\\|d_k\\|/\\|s_r\\|$")
    axs[0].set_title("(a) deeper loops take smaller steps")
    axs[1].set_ylabel("$r\\,\\|d_k\\|/\\|s_r\\|$")
    axs[1].set_title("(b) times $r$: one profile")
    axs[1].legend(loc="upper right", fontsize=6, ncol=2)
    for ax in axs:
        ax.set_xlabel("rescaled loop $k/r$ (before $r$)")
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.axvspan(0.3, 0.9, color=GRID, alpha=0.6, lw=0)
    check_bounds(fig, "fig_profile")


def fig_brake():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True)
    ax = axs[0]
    for r in DEPTHS:
        profile = np.mean([lam(x)[:2 * r] for x in base_runs(r)], 0)
        u = np.arange(1, len(profile) + 1) / r
        ax.plot(u, profile, color=DCOL[r], lw=1.4, label="$r{=}%d$" % r)
        ax.plot([1], [profile[r - 1]], "o", ms=4, color=DCOL[r], mec="white", mew=0.8, zorder=5)
    ax.axvline(1, color=MUTED, lw=0.8, alpha=0.6)
    ax.set_xlabel("rescaled loop $k/r$")
    ax.set_ylabel("typical log-stretch $\\lambda_k$")
    ax.set_title("(a) the brake at $r$, depths 6 to 24")
    ax.legend(loc="upper right", fontsize=6, ncol=4, columnspacing=0.8, handlelength=1.2)
    ax.set_xlim(0, 2.05)
    ax = axs[1]
    for small, large in (("r8", "r8_15"), ("r16", "r16_15")):
        for run, ls, tokens in ((RUN[small], ":", "0.5B"), (RUN[large], "-", "1.5B")):
            profile = lam(run)
            r = load(run, "lyapunov.json")["r"]
            u = np.arange(1, len(profile) + 1) / r
            ax.plot(u, profile, color=DCOL[r], ls=ls, lw=1.5, label="$r{=}%d$, %s" % (r, tokens))
            ax.plot([1], [profile[r - 1]], "o", ms=4, color=DCOL[r], mec="white", mew=0.8)
    ax.axvline(1, color=MUTED, lw=0.8, ls="-", alpha=0.6)
    ax.set_xlabel("rescaled loop $k/r$")
    ax.set_title("(b) deeper with 3$\\times$ data")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.68, 0.45)
    ax.set_ylabel("typical log-stretch $\\lambda_k$")
    check_bounds(fig, "fig_brake")


def fig_wave():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.4), constrained_layout=True)
    sw = load(RUN["r8_15"], "state_distance.json")
    r, K = sw["r"], sw["K"]
    D = np.array(sw["D"])[r - 1]
    k = np.arange(1, K + 1)
    Dp = np.array(load(RUN["pois"], "state_distance.json")["D"])[7]
    kp = np.arange(1, len(Dp) + 1)
    ax = axs[0]
    La, Lpa = loss_curve(RUN["r8_15"]), loss_curve(RUN["pois"])
    ka = np.arange(len(La))
    ax.plot(ka[1:65], La[1:65] - La[r], color=DCOL[8], lw=1.5, label="fixed $r{=}8$, 1.5B")
    ax.plot(ka[1:65], Lpa[1:65] - Lpa[8], color=C["pois"], lw=1.3, ls="--", label="Poisson (mean 8)")
    ax.plot([r], [0], "o", ms=4, color=DCOL[8], mec="white", mew=0.8)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylim(-0.05, 0.6)
    ax.set_xlim(0, 64)
    ax.set_xlabel("loop $k$")
    ax.set_ylabel("$L_k - L_r$ (nats)")
    ax.set_title("(a) the loss rises and falls")
    ax.legend(loc="upper right", fontsize=6)
    ax = axs[1]
    ax.plot(k, D, color=DCOL[8], lw=1.5, label="fixed $r{=}8$, 1.5B")
    ax.plot(kp[kp <= K], Dp[kp <= K], color=C["pois"], lw=1.3, ls="--", label="Poisson (from $s_8$)")
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("loop $k$")
    ax.set_ylabel(r"$\|s_k - s_r\| \, / \, \|s_r\|$")
    ax.set_title("(b) distance from $s_r$")
    ax.legend(loc="upper right", fontsize=6)
    ax = axs[2]
    for r in DEPTHS:
        turns = [step_profiles(x)[3] for x in base_runs(r)]
        n = min(len(t) for t in turns)
        ax.plot(np.arange(n) / r, np.mean([t[:n] for t in turns], 0) / (2 * np.pi), color=DCOL[r], lw=1.3,
                label="$r{=}%d$" % r)
    turns = [step_profiles("poisson_r8_b4_s%d" % s)[3] for s in range(3)]
    n = min(len(t) for t in turns)
    ax.plot(np.arange(n) / 8, np.mean([t[:n] for t in turns], 0) / (2 * np.pi), color=C["pois"], lw=1.3, ls="--",
            label="Poisson")
    ax.axhline(1, color=MUTED, lw=0.7, ls=":")
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.05, 2.1)
    ax.set_xlabel("loops after $r$, $j/r$")
    ax.set_ylabel("turns of the step $\\Theta(j)/2\\pi$")
    ax.set_title("(c) the step keeps turning")
    ax.legend(loc="upper left", fontsize=5.5, ncol=2, columnspacing=0.6, handlelength=1.0)
    check_bounds(fig, "fig_wave")


def fig_relax():
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.2), constrained_layout=True)
    for ax, key in zip(axs, ("r8", "r16")):
        v = load(RUN[key], "plane_dynamics.json")
        r = v["r"]
        L = np.array(v["A"]["loss"])
        H = np.array(v["C"]["relax_all_rp%d" % (2 * r)]["loss"])
        k, kh = np.arange(len(L)), np.arange(len(H))
        sel, selh = (k >= r // 2) & (k <= 4 * r), (kh >= r // 2) & (kh <= 4 * r)
        ax.plot(k[sel] / r, L[sel] - L[r], color=DCOL[r], lw=1.6, label="step $d_k$ (trained)")
        ax.plot(kh[selh] / r, H[selh] - L[r], color=INK, lw=1.4, ls="--", label="step $d_k/2$")
        for x in (1, 2):
            ax.axvline(x, color=MUTED, lw=0.6, alpha=0.6)
        ax.axhline(0, color=MUTED, lw=0.6, ls=":")
        ax.set_xlabel("loop $k/r$")
        ax.set_title("(%s) fixed $r{=}%d$" % ("a" if key == "r8" else "b", r))
        ax.set_xlim(0.5, 4)
        ax.legend(loc="upper right")
    axs[0].set_ylabel("$L_k - L_r$ (nats)")
    check_bounds(fig, "fig_relax")


def random_depth_panel(ax):
    ours = [lam("poisson_r8_b4_%ss%d" % (opt, s)) for opt in ("", "muon_") for s in range(3)]
    n = min(len(x) for x in ours)
    ours = np.array([x[:n] for x in ours])
    k = np.arange(1, n + 1)
    ax.fill_between(k / 8, ours.min(0), ours.max(0), color=C["pois"], alpha=0.25, lw=0)
    ax.plot(k / 8, ours.mean(0), color=C["pois"], ls="--", lw=1.3, label="our Poisson")
    colors = {"parcae-140m": "#9ec3ea", "parcae-370m": "#6aa3df", "parcae-770m": "#2a78d6", "parcae-1.3b": "#174f96",
              "huginn-0125": INK}
    labels = {"parcae-1.3b": "Parcae", "huginn-0125": "Huginn"}
    for m, c in colors.items():
        j = load_public(m, "public_dynamics.json")
        profile = np.array(j["lyap"]).mean(1)
        k = np.arange(1, len(profile) + 1)
        ax.plot(k[k >= 2] / j["r"], profile[k >= 2], color=c, lw=1.0, label=labels.get(m))
    ax.axvline(1, color=MUTED, lw=0.8, alpha=0.6)
    ax.set_xlabel(r"loop $k/\bar r$ (training depth $\bar r$)")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.45, 0.3)


def fig_recipe():
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.75), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.12, wspace=0.06)
    for label, runs, col, ls in RECIPES:
        profile = np.mean([lam(x)[:16] for x in runs], 0)
        axs[0].plot(np.arange(1, len(profile) + 1), profile, color=col, ls=ls, lw=1.4, label=label)
        curves = [loss_curve(x) for x in runs]
        n = min(len(L) for L in curves)
        L = np.mean([x[:n] for x in curves], 0)
        kk = np.arange(n)
        m = (kk >= 4) & (kk <= 40)
        axs[1].plot(kk[m], L[m] - L[8], color=col, ls=ls, lw=1.4, label=label)
    for ax in axs[:2]:
        ax.axvline(8, color=MUTED, lw=0.8, alpha=0.6)
        ax.set_xlabel("loop $k$")
    axs[0].set_ylabel("typical log-stretch $\\lambda_k$")
    axs[0].set_title("(a) the brake, by recipe")
    axs[0].set_xlim(0, 16.5)
    axs[0].set_ylim(-0.75, 0.35)
    axs[1].axhline(0, color=MUTED, lw=0.6)
    axs[1].set_ylabel("$L_k - L_8$ (nats)")
    axs[1].set_title("(b) the loss past $r$")
    axs[1].set_xlim(4, 40)
    axs[1].set_ylim(-0.03, 0.32)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6.5, handlelength=2.2)
    ax = axs[2]
    random_depth_panel(ax)
    for name, c in (("Ouro-1.4B", "#8e5bc9"), ("Ouro-2.6B", "#5b2f99")):
        j = load_public(name, "ouro_dynamics.json")
        profile = np.array(j["lyap"]).mean(1)
        k = np.arange(1, len(profile) + 1)
        ax.plot(k[k >= 2] / j["r"], profile[k >= 2], color=c, lw=1.2, ls="-.", label=name.replace("-", " "))
    ax.set_title("(c) public models")
    ax.set_ylabel("typical log-stretch $\\lambda_k$")
    ax.set_ylim(-0.45, 0.5)
    ax.legend(loc="upper left", fontsize=5.0, ncol=2, handlelength=1.5, columnspacing=0.6)
    check_bounds(fig, "fig_recipe")


def fig_temporal():
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.3), constrained_layout=True)
    for key, ls in (("r4", "-"), ("r8", "-"), ("r16", "-"), ("pois", "--")):
        j = load(RUN[key], "loop_decoder.json")
        r = j["r"]
        kh = np.array(j["khat_mean"])
        k = np.arange(1, len(kh) + 1)
        ax.plot(k / r, kh / r, color=C[key], ls=ls, lw=1.5 if key != "pois" else 1.3, label=NAME[key])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls=":")
    ax.axvline(1, color=MUTED, lw=0.6, alpha=0.6)
    ax.set_xlabel("loop $k/r$")
    ax.set_ylabel("decoded loop $\\hat k / r$")
    ax.set_xlim(0, 4)
    ax.legend(loc="lower right")
    check_bounds(fig, "fig_temporal")


def fig_normality():
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3), constrained_layout=True)
    for key in ("r8", "r12", "r16", "pois"):
        j = load(RUN[key], "non_normality.json")
        r = j["r"]
        ks = np.array(sorted(int(k) for k in j["nu"]))
        nu = np.array([j["nu"][str(k)] for k in ks])
        m = (ks >= 2) & (ks <= 3 * r)
        ax.plot(ks[m] / r, nu[m], color=C[key], lw=1.5 if key != "pois" else 1.3,
                ls={"pois": "--", "r16": "-."}.get(key, "-"), label=NAME[key])
        if key != "pois":
            ax.plot([1], [j["nu"][str(r)]], "o", ms=5.5, color=C[key], mec="white", mew=0.8, zorder=5)
    ax.axvline(1, color=MUTED, lw=0.6, alpha=0.6)
    ax.set_xlabel("loop $k/r$")
    ax.set_ylabel("departure from normality $\\nu_k$")
    ax.set_xlim(0, 3)
    ax.legend(loc="upper right")
    check_bounds(fig, "fig_normality")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_profile()
    fig_brake()
    fig_wave()
    fig_relax()
    fig_recipe()
    fig_temporal()
    fig_normality()


if __name__ == "__main__":
    main()
