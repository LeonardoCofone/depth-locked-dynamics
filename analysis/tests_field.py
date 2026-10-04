import json
import os

import numpy as np

from analysis.common import ROOT


def load_set(name):
    return json.load(open(os.path.join(ROOT, "field_alignment", "field_alignment_%s.json" % name)))


def depth_of(key):
    return int(key.split("s")[0])


def report(label, cells):
    n = len(cells)
    yes = sum(c >= 0.8 * C and c >= null + 0.2 for c, C, null, _, _, _ in cells)
    no = sum(c < 0.5 * C for c, C, null, _, _, _ in cells)
    ratio = sum(0.75 <= rho <= 1.33 for _, _, _, rho, _, _ in cells)
    raw = sum(abs(np.log(rho) - np.log(rb / ra)) < abs(np.log(rho)) for _, _, _, rho, ra, rb in cells)
    print("%s, direction: %d of %d (%d below half the ceiling) -> %s" % (
        label, yes, n, no, "YES" if yes / n >= 2 / 3 else ("NO" if no / n >= 2 / 3 else "grey")))
    print("%s, norm after the factor r: %d of %d -> %s" % (
        label, ratio, n, "YES" if ratio / n >= 2 / 3 else ("NO" if raw / n >= 2 / 3 else "grey")))


def base_set():
    E = load_set("base")
    same = ["8s0->8s1", "8s0->8s2", "8s1->8s2"]
    cells = []
    for key in ("8s0->12", "8s0->16", "12->16"):
        a, b = key.split("->")
        for t, v in E["cross"][key].items():
            if v["R2"] < 0.5:
                continue
            C = float(np.mean([E["same"][s][t]["cos_med"] for s in same]))
            cells.append((v["cos_med"], C, v["cos_null"], v["ratio_med"], depth_of(a), depth_of(b)))
    report("same field across depths 8, 12, 16", cells)


def grid_set():
    E = load_set("grid")
    depths = (6, 8, 10, 12, 16, 20, 24)
    cells = []
    for sd in (0, 1):
        for a, b in zip(depths[:-1], depths[1:]):
            key = "%ds%d->%ds%d" % (a, sd, b, sd)
            sa, sb = "%ds0->%ds1" % (a, a), "%ds0->%ds1" % (b, b)
            for t, v in E["cross"][key].items():
                if v["R2"] < 0.5:
                    continue
                C = 0.5 * (E["same"][sa][t]["cos_med"] + E["same"][sb][t]["cos_med"])
                cells.append((v["cos_med"], C, v["cos_null"], v["ratio_med"], a, b))
    report("same field across neighbouring depths 6..24", cells)


def main():
    base_set()
    grid_set()


if __name__ == "__main__":
    main()
