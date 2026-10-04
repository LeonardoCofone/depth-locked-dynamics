import argparse
import json
import os
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from loopedlm.checkpoint import load_model, run_config, trained_runs
from loopedlm.data import fixed_batches

STATE_SEED = 7


def base_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="folder with the trained runs")
    ap.add_argument("--data", default="data", help="folder with val.npy and tokenizer.json")
    ap.add_argument("--out", default="results/models", help="one subfolder per run is written here")
    ap.add_argument("--only", default="", help="measure only runs whose name contains this string")
    ap.add_argument("--redo", type=int, default=0, help="1 = overwrite existing outputs")
    return ap


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation(data):
    return np.load(os.path.join(data, "val.npy"), mmap_mode="r")


def sequences(data, n=64, seed=4242, T=1024):
    return fixed_batches(validation(data), T, 1, n, seed)[0]


def state_generator(dev):
    return torch.Generator(device=dev).manual_seed(STATE_SEED)


def initial_state(m, e):
    return m.init_state(e, state_generator(e.device))


def ce(logits, y):
    return float(F.cross_entropy(logits.float().flatten(0, 1), y.flatten()))


def readout_loss(m, s, y):
    return ce(m.readout(s), y)


def jvp(f, s, v, rel=1e-4):
    eps = rel * s.norm() / v.norm().clamp_min(1e-30)
    return (f(s + eps * v) - f(s - eps * v)) / (2 * eps)


def vjp(f, s, u):
    with torch.enable_grad():
        z = s.detach().requires_grad_(True)
        g, = torch.autograd.grad(f(z), z, grad_outputs=u)
    return g.detach()


@torch.no_grad()
def lyapunov(steps, s0, gen, ndir=16):
    shape = s0.shape
    Q = torch.linalg.qr(torch.randn(s0.numel(), ndir, dtype=s0.dtype, device=s0.device, generator=gen))[0]
    s, out = s0, []
    for f in steps:
        V = torch.stack([jvp(f, s, Q[:, i].view(shape)).reshape(-1) for i in range(ndir)], 1)
        Q, R = torch.linalg.qr(V)
        sign = torch.sign(torch.diagonal(R))
        Q, R = Q * sign, R * sign[:, None]
        out.append(torch.log(torch.diagonal(R).abs().clamp_min(1e-300)).cpu())
        s = f(s)
    return torch.stack(out).numpy(), s


def non_normality(f, s, n, gen, parts=None):
    num = den = 0.0
    for _ in range(n):
        v = torch.randn(s.shape, dtype=s.dtype, device=s.device, generator=gen)
        Jv, JTv = jvp(f, s, v), vjp(f, s, v)
        JJTv, JTJv = jvp(f, s, JTv), vjp(f, s, Jv)
        a, b = float(((JJTv - JTJv) ** 2).sum()), float((JTJv ** 2).sum())
        num += a
        den += b
        if parts is not None:
            parts.append([a, b])
    return (num / den) ** 0.5


def safe_eigh(G):
    G = torch.nan_to_num(G.double(), nan=0.0, posinf=0.0, neginf=0.0)
    m = G.shape[-1]
    tr = torch.diagonal(G, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-300)
    G = G + 1e-12 * tr[..., None, None] * torch.eye(m, dtype=G.dtype, device=G.device)
    try:
        return torch.linalg.eigh(G)
    except Exception:
        lam, V = torch.linalg.eigh(G.cpu())
        return lam.to(G.device), V.to(G.device)


def window_geometry(X, robust=False):
    Xt = X.permute(1, 0, 2).double()
    if robust:
        Xt = torch.nan_to_num(Xt)
        lam, V = safe_eigh(Xt @ Xt.transpose(1, 2))
    else:
        lam, V = torch.linalg.eigh(Xt @ Xt.transpose(1, 2))
    lam, V = lam.flip(-1).clamp_min(0), V.flip(-1)
    total = lam.sum(-1).clamp_min(1e-30)
    e2, e3 = lam[:, :2].sum(-1) / total, lam[:, :3].sum(-1) / total
    a = V[:, :, 0] * lam[:, 0:1].sqrt()
    b = V[:, :, 1] * lam[:, 1:2].sqrt()
    phase = torch.atan2(b, a)
    dphase = torch.diff(phase, dim=1)
    dphase = (dphase + np.pi) % (2 * np.pi) - np.pi
    cum = torch.cat([torch.zeros_like(dphase[:, :1]), dphase.cumsum(1)], 1)
    cum = cum * torch.sign(cum[:, -1:]).where(cum[:, -1:] != 0, torch.ones_like(cum[:, -1:]))
    q = lambda t, p: [float(v) for v in t.quantile(p, dim=0)] if t.dim() > 1 else float(t.quantile(p))
    return {"e2_med": q(e2, 0.5), "e2_q25": q(e2, 0.25), "e2_q75": q(e2, 0.75), "e3_med": q(e3, 0.5),
            "cumphase_med": q(cum, 0.5), "cumphase_q25": q(cum, 0.25), "cumphase_q75": q(cum, 0.75)}


def quantiles(t, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    t = t[torch.isfinite(t)].double()
    return {str(p): float(t.quantile(p)) for p in ps} if t.numel() else None


def phase_period(cum, turns=1):
    cum = np.array(cum)
    for j in range(1, len(cum)):
        if cum[j] >= 2 * np.pi * turns:
            return float(j - 1 + (2 * np.pi * turns - cum[j - 1]) / (cum[j] - cum[j - 1]))
    return None


def for_each_run(a, filename, measure, dtype=torch.float32, keep=lambda cfg, name: True):
    dev = device()
    for run in trained_runs(a.runs, a.only):
        name = os.path.basename(run)
        dst = os.path.join(a.out, name, filename)
        if os.path.exists(dst) and not a.redo:
            print(name, "already measured in", a.out, flush=True)
            continue
        if not keep(run_config(run), name):
            continue
        t0 = time.time()
        try:
            m, cfg = load_model(run, dev, dtype)
            res = measure(m, cfg, dev)
            if isinstance(res, tuple):
                res, arrays = res
            else:
                arrays = None
            res.update(run=name, secs=time.time() - t0)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(res, open(dst, "w"))
            if arrays:
                np.savez_compressed(dst.replace(".json", "_tokens.npz"), **arrays)
            print(name, "%.0fs" % res["secs"], flush=True)
        except Exception:
            print(name, "failed", traceback.format_exc()[-1500:], flush=True)
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
