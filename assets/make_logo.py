import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INK, ACCENT, FIELD = "#33407a", "#d9822b", "#ffffff"
SIZE = 512


def geometry():
    t = np.linspace(0, 1.75 * np.pi, 400)
    radius = 170 * np.exp(-0.42 * t)
    approach = np.stack([200 + radius * np.cos(t + np.pi), 250 + radius * np.sin(t + np.pi)], 1)
    land = approach[-1]
    a = np.linspace(0, 2 * np.pi, 200)
    orbit = np.stack([land[0] + 95 - 95 * np.cos(a), land[1] + 52 * np.sin(a)], 1)
    both = np.concatenate([approach, orbit])
    shift = SIZE / 2 - (both.min(0) + both.max(0)) / 2
    return approach + shift, orbit + shift, land + shift


def svg(approach, orbit, land):
    path = " ".join("%s %.1f %.1f" % ("M" if i == 0 else "L", x, y) for i, (x, y) in enumerate(approach))
    ring = " ".join("%s %.1f %.1f" % ("M" if i == 0 else "L", x, y) for i, (x, y) in enumerate(orbit)) + " Z"
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512" role="img" '
            'aria-label="A trajectory that spirals in, lands on a point and keeps turning">\n'
            '<title>Depth-Locked Dynamics</title>\n'
            '<rect x="4" y="4" width="504" height="504" rx="28" fill="%s" stroke="%s" stroke-width="8"/>\n'
            '<path d="%s" fill="none" stroke="%s" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/>\n'
            '<path d="%s" fill="none" stroke="%s" stroke-width="10" stroke-dasharray="18 14" stroke-linecap="round"/>\n'
            '<circle cx="%.1f" cy="%.1f" r="20" fill="%s"/>\n</svg>\n'
            % (FIELD, INK, path, INK, ring, ACCENT, land[0], land[1], ACCENT))


def png(approach, orbit, land):
    fig = plt.figure(figsize=(4, 4), dpi=SIZE / 4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SIZE)
    ax.set_ylim(SIZE, 0)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((4, 4), 504, 504, facecolor=FIELD, edgecolor=INK, lw=6))
    ax.plot(approach[:, 0], approach[:, 1], color=INK, lw=13, solid_capstyle="round")
    closed = np.vstack([orbit, orbit[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=ACCENT, lw=7, ls=(0, (2, 1.5)))
    ax.add_patch(plt.Circle(land, 20, color=ACCENT))
    fig.savefig(os.path.join(HERE, "logo.png"), dpi=SIZE / 4)


def main():
    approach, orbit, land = geometry()
    open(os.path.join(HERE, "logo.svg"), "w").write(svg(approach, orbit, land))
    png(approach, orbit, land)


if __name__ == "__main__":
    main()
