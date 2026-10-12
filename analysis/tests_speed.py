import numpy as np

from analysis.common import cv, depth, load
from analysis.tests_decoder import wave_models
from analysis.tests_wave import separated_minima

BASE = ["fixed_r4_b4_s%d" % s for s in range(3)] + ["fixed_r8_b8_s%d" % s for s in range(3)] + [
    "fixed_r12_b12_s0", "fixed_r16_b16_s0"]


def step(d):
    return np.array([np.nan] + load(d, "loop_decoder.json")["step"])


def speed():
    by = {}
    for d in BASE:
        r = depth(d)
        by.setdefault(r, []).append(np.nanmean(step(d)[r + 1:2 * r + 1]))
    mean = {r: float(np.mean(v)) for r, v in by.items()}
    scaled, raw = cv([r * v for r, v in mean.items()]), cv(list(mean.values()))
    v = "YES" if scaled <= 0.2 else ("NO" if raw <= 0.2 else "grey")
    print("speed after r scales as 1/r: CV of r*speed %.2f, CV of speed %.2f -> %s  %s" % (
        scaled, raw, v, {r: round(x, 3) for r, x in mean.items()}))


def period_from_speed():
    vals = []
    for d in wave_models():
        r = depth(d)
        L = np.array([np.nan] + load(d, "state_distance.json")["loss_coda"])
        m = separated_minima(L, r)
        P = m[1] - m[0]
        vals.append(P * np.nanmean(step(d)[r + 1:r + P + 1]))
    c = cv(vals)
    print("period from the speed after r: CV of P*speed %.2f over %d models (%.2f to %.2f) -> %s" % (
        c, len(vals), min(vals), max(vals), "YES" if c <= 0.25 else "NO"))


def main():
    print("== Pure rescaling of time after r")
    speed()
    period_from_speed()


if __name__ == "__main__":
    main()
