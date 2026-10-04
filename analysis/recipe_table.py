import argparse

import numpy as np

from analysis.common import paper_models, summaries

F = "fixed_r%d_b%d_s%d"
GROUPS = [
    ("Fixed $r=4$", [F % (4, 4, s) for s in range(3)], ""),
    ("Fixed $r=6$$^\\dagger$", [F % (6, 6, s) for s in range(2)], "\\rowtint "),
    ("Fixed $r=8$", [F % (8, 8, s) for s in range(3)], ""),
    ("Fixed $r=10$$^\\dagger$", [F % (10, 10, s) for s in range(2)], "\\rowtint "),
    ("Fixed $r=12$", [F % (12, 12, s) for s in range(2)], ""),
    ("Fixed $r=16$", [F % (16, 16, s) for s in range(2)], "\\rowtint "),
    ("Fixed $r=20$$^\\dagger$", [F % (20, 20, s) for s in range(2)], ""),
    ("Fixed $r=24$$^\\dagger$", [F % (24, 24, s) for s in range(2)], "\\rowtint "),
    ("Fixed $r=8$ / $16$, 1.5B tokens", ["fixed_r8_b8_D1500M_s0", "fixed_r16_b16_D1500M_s0"], ""),
    ("Fixed $r=8$, Muon", ["fixed_r8_b8_muon_s%d" % s for s in range(3)], "\\rowtint "),
    ("Fixed $r=8$, decay 0.7", ["fixed_r8_b8_dec0.7_s%d" % s for s in range(2)], ""),
    ("Fixed $r=8$, decay 0.9", ["fixed_r8_b8_dec0.9_s%d" % s for s in range(2)], "\\rowtint "),
    ("Fixed $r=8$, width 768", ["fixed_r8_b8_d768_s0"], ""),
    ("Truncated $r=8$, through 4$^\\dagger$", ["fixed_r8_b4_s%d" % s for s in range(2)], "\\rowbad "),
    ("Truncated $r=12$, through 8", ["fixed_r12_b8_s%d" % s for s in range(3)], "\\rowbad "),
    ("Truncated $r=16$, through 8", ["fixed_r16_b8_s%d" % s for s in range(3)], "\\rowbad "),
    ("Loss at every loop, $r=8$$^\\dagger$", ["fixed_r8_b8_all_s%d" % s for s in range(2)], "\\rowok "),
    ("Poisson, through 4, AdamW / Muon", ["poisson_r8_b4_s%d" % s for s in range(3)]
     + ["poisson_r8_b4_muon_s%d" % s for s in range(3)], "\\rowok "),
    ("Poisson, through 4, decay 0.9$^\\dagger$", ["poisson_r8_b4_dec0.9_s%d" % s for s in range(2)], "\\rowok "),
    ("Poisson, through every loop$^\\dagger$", ["poisson_r8_b32_s%d" % s for s in range(2)], "\\rowok "),
]


PAPER = [
    ("\\rowhi Fixed, through every loop", [F % (8, 8, s) for s in range(3)]),
    ("Fixed, through every loop, decay 0.9", ["fixed_r8_b8_dec0.9_s%d" % s for s in range(2)]),
    ("\\rowbad Fixed, truncated to 4$^\\dagger$", ["fixed_r8_b4_s%d" % s for s in range(2)]),
    ("\\rowok Loss at every loop$^\\dagger$", ["fixed_r8_b8_all_s%d" % s for s in range(2)]),
    ("\\rowok Poisson, through every loop$^\\dagger$", ["poisson_r8_b32_s%d" % s for s in range(2)]),
    ("\\rowok Poisson, truncated to 4", ["poisson_r8_b4_s%d" % s for s in range(3)]
     + ["poisson_r8_b4_muon_s%d" % s for s in range(3)]),
    ("\\rowok Poisson, truncated to 4, decay 0.9$^\\dagger$", ["poisson_r8_b4_dec0.9_s%d" % s for s in range(2)]),
]


def span(v, f):
    v = [x for x in v if x is not None]
    if not v:
        return "none"
    a, b = f % min(v), f % max(v)
    return a if a == b else "%s to %s" % (a, b)


def cells(group):
    q = span([o.get("q") for o in group], "%.2f")
    braking = [o["lam"][o["r"] - 1] for o in group if o.get("argmin_ok")]
    brake = span(braking, "$%.2f$") if braking else "none"
    if braking and len(braking) < len(group):
        brake += " (%d of %d)" % (len(braking), len(group))
    periods = [o.get("PT_sg") for o in group if o.get("PT_sg")]
    period = span(periods, "%.1f") if periods else "none"
    if periods and len(periods) < len(group):
        period += " (%d of %d)" % (len(periods), len(group))
    rises = [float(o["L"][o["r"]:2 * o["r"] + 1].max() - o["L"][o["r"]]) for o in group]
    rise = "$<%.3f$" % (np.ceil(max(rises) * 1000) / 1000) if max(rises) < 0.01 else span(rises, "%.2f")
    osc = sum(1 for o in group if o["osc"])
    return "%s & %s & %s & %s & %d of %d" % (q, brake, period, rise, osc, len(group))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every recipe and depth, not only the paper table at r = 8")
    a = ap.parse_args()
    S = summaries(paper_models())
    if not a.all:
        for name, runs in PAPER:
            print("%s & %d & %s \\\\" % (name, len(runs), cells([S[d] for d in runs])))
        return
    for name, runs, style in GROUPS:
        print("%s%s & %s \\\\" % (style, name, cells([S[d] for d in runs])))


if __name__ == "__main__":
    main()
