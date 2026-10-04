import numpy as np

from analysis.common import (CONTROL_GRID, brake_of, contraction_rate, depth, bptt, has, is_fixed, lambda_profile, load,
                             loss_curve, minima, paper_models, phase_period)

TAGS = (("_D1500M", "1.5B tok."), ("_muon", "Muon"), ("_dec0.7", "decay 0.7"), ("_dec0.9", "decay 0.9"),
        ("_d768", "width 768"))


def label(name):
    r, b, seed = depth(name), bptt(name), name.split("_s")[-1]
    tags = []
    if is_fixed(name) and b != r:
        tags.append("trunc. %d" % b)
    if not is_fixed(name) and b != 4:
        tags.append("full backprop")
    if "_all" in name:
        tags.append("loss at every loop")
    tags += [text for tag, text in TAGS if tag in name]
    text = "fixed $r{=}%d$" % r if is_fixed(name) else "Poisson, $\\bar r{=}8$"
    if tags:
        text += ", " + ", ".join(tags)
    return text + ", s" + seed + ("$^\\dagger$" if name in CONTROL_GRID else "")


def cell(x, f):
    if isinstance(x, float) and f == "%.2f" and abs(x) < 0.005:
        x = 0.0
    if isinstance(x, float) and f == "%.3f" and abs(x) < 0.0005:
        x = 0.0
    return "$\\cdot$" if x is None else f % x


def row(name):
    r = depth(name)
    L = loss_curve(name)
    lam = lambda_profile(name)
    am, rel = brake_of(lam, r)
    q = contraction_rate(name)
    fixed = is_fixed(name)
    m = minima(L, r) if fixed else []
    P = m[0] - r if (m and L[r:m[0] + 1].max() - L[m[0]] >= 0.001) else None
    PT = None
    if fixed and has(name, "step_geometry.json") and r >= 4:
        PT = phase_period(load(name, "step_geometry.json")["post"]["cumphase_med"])
    amp = L[r + 1:2 * r + 1].max() - L[r]
    trained2r = scaled2r = None
    if fixed and has(name, "plane_dynamics.json"):
        v = load(name, "plane_dynamics.json")
        La = np.array(v["A"]["loss"])
        trained2r = La[2 * r] - La[r]
        key = "relax_all_rp%d" % (2 * r)
        if key in v["C"] and "loss" in v["C"][key]:
            scaled2r = v["C"][key]["loss"][2 * r] - La[r]
    return (not fixed, r, name, label(name), q, lam[r - 1], am, rel, P, PT, amp, trained2r, scaled2r)


def main():
    rows = sorted((row(n) for n in paper_models()), key=lambda x: (x[0], x[1], x[2]))
    for x in rows:
        print("%s & %d & %s & $%.2f$ & %d & $%.3f$ & %s & %s & %s & %s & %s \\\\" % (
            x[3], x[1], cell(x[4], "%.2f"), x[5], x[6], x[7], cell(x[8], "%d"), cell(x[9], "%.1f"),
            cell(x[10], "%.2f"), cell(x[11], "%.3f"), cell(x[12], "%.3f")))
    print(len(rows), "models")


if __name__ == "__main__":
    main()
