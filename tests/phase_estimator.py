import math

import torch

from measure.common import window_geometry

STEPS, TOKENS, DIM, PERIOD, RHO, NOISE = 40, 300, 64, 13, 0.95, 0.01


def main():
    omega = 2 * math.pi / PERIOD
    g = torch.Generator().manual_seed(0)
    X = torch.zeros(STEPS, TOKENS, DIM)
    for n in range(TOKENS):
        Q, _ = torch.linalg.qr(torch.randn(DIM, 2, generator=g))
        for i in range(STEPS):
            X[i, n] = RHO ** i * (math.cos(omega * i) * Q[:, 0] + math.sin(omega * i) * Q[:, 1])
            X[i, n] += NOISE * torch.randn(DIM, generator=g)
    w = window_geometry(X)
    print("energy in two modes %.3f" % w["e2_med"])
    print("phase after one turn (%d steps) %.3f, expected %.3f" % (PERIOD, w["cumphase_med"][PERIOD], 2 * math.pi))
    print("phase per step %.3f, expected %.3f" % (w["cumphase_med"][1], omega))


if __name__ == "__main__":
    main()
