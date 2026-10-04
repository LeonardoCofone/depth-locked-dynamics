import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from loopedlm.model import VALUE_GATE_CHANNELS, apply_rope
from measure.common import (base_parser, for_each_run, initial_state, jvp, phase_period, quantiles, safe_eigh,
                            sequences, window_geometry)

MODES = {"full": (), "noattn": ("attn",), "nomlp": ("mlp",), "frozenP": ("P",), "frozennorm": ("norm",),
         "decayonly": ("attn", "mlp")}


def rms(norm, x, key, mode, cache, rec):
    fac = cache[key] if "norm" in mode else torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
    if rec is not None:
        rec[key] = fac
    return x * fac * norm.weight.to(x.dtype)


def attention(at, x, cos, sin, ve, key, mode, cache, rec):
    B, T, d = x.shape
    q, k, v = at.qkv(x).view(B, T, 3, at.nh, at.hd).unbind(2)
    if at.ve_gate is not None and ve is not None:
        gate = 2 * torch.sigmoid(at.ve_gate(x[..., :VALUE_GATE_CHANNELS]))
        v = v + gate.unsqueeze(-1) * ve.view(B, T, at.nh, at.hd)
    if at.qk_style == "learned":
        q, k = at.qn(q), at.kn(k)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if at.qk_style == "parcae":
        q, k = F.rms_norm(q, (at.hd,)), F.rms_norm(k, (at.hd,))
    if "P" in mode:
        P = cache[key]
    else:
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(at.hd)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        P = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1)
    if rec is not None:
        rec[key] = P
    return at.proj((P @ v).transpose(1, 2).reshape(B, T, d))


def core_paths(m, u, mode=(), cache=None, rec=None):
    x, o = u, m.offset["core"]
    for i, b in enumerate(m.core):
        h = rms(b.n1, x, (i, "n1"), mode, cache, rec)
        a = cache[(i, "a")] if "attn" in mode else attention(b.attn, h, m.cos, m.sin, m.current_ve.get(o + i), (i, "P"),
                                                             mode, cache, rec)
        if rec is not None:
            rec[(i, "a")] = a
        x = x + a
        h2 = rms(b.n2, x, (i, "n2"), mode, cache, rec)
        mm = cache[(i, "m")] if "mlp" in mode else b.mlp(h2)
        if rec is not None:
            rec[(i, "m")] = mm
        x = x + mm
    return x


def phi(m, s, e, mode=(), cache=None, rec=None):
    return core_paths(m, m.inject(s, e), mode, cache, rec)


def tok_angle(a, b):
    c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)
    return torch.arccos(c.clamp(-1, 1))


def eig_rho_theta(T):
    ev = torch.linalg.eigvals(T.double())
    mod = ev.abs()
    rho = mod.max(-1).values
    ang = ev.angle().abs()
    cplx = ev.imag.abs() > 1e-9
    modc = torch.where(cplx, mod, torch.full_like(mod, -1.0))
    ic, im = modc.argmax(-1), mod.argmax(-1)
    th = torch.where(cplx.any(-1), ang.gather(-1, ic[:, None])[:, 0], ang.gather(-1, im[:, None])[:, 0])
    return rho, th, ev


def window_dmd(X, k=2):
    Xt = torch.nan_to_num(X.permute(1, 0, 2).double())
    lam, U = safe_eigh(Xt @ Xt.transpose(1, 2))
    lam, U = lam.flip(-1)[:, :k].clamp_min(1e-300), U.flip(-1)[:, :, :k]
    S = lam.sqrt()
    Z = U * S[:, None, :]
    Q = (Xt.transpose(1, 2) @ U) / S[:, None, :]
    Tm = torch.linalg.lstsq(Z[:, :-1], Z[:, 1:]).solution
    rho, th, ev = eig_rho_theta(Tm)
    return rho, th, ev, Q


@torch.no_grad()
def readout_loss(m, s, y):
    return float(F.cross_entropy(m.readout(s).float().flatten(0, 1), y.flatten()))


@torch.no_grad()
def run_traj(m, y, s0, K, step_fn):
    s, S, losses = s0, [s0.float().reshape(-1, s0.shape[-1])], [readout_loss(m, s0, y)]
    for k in range(K):
        s = step_fn(k, s)
        S.append(s.float().reshape(-1, s.shape[-1]))
        losses.append(readout_loss(m, s, y))
    return np.array(losses), torch.stack(S)


def traj_summary(losses, St, land, windows):
    D = St[1:] - St[:-1]
    sr = St[land]
    nr = sr.norm(dim=-1).clamp_min(1e-30)
    out = {"loss": losses.tolist(), "land": land,
           "dist_med": [float(((St[k] - sr).norm(dim=-1) / nr).median()) for k in range(St.shape[0])],
           "step_med": [float((D[k].norm(dim=-1) / nr).median()) for k in range(D.shape[0])],
           "alpha3d_med": [float(tok_angle(D[k], D[k + 1]).median()) for k in range(D.shape[0] - 1)],
           "win": {}, "finite": bool(torch.isfinite(St).all())}
    tok = {}
    for name, (a, b) in windows.items():
        b = min(b, D.shape[0])
        if b - a < 4:
            continue
        try:
            g = window_geometry(D[a:b], robust=True)
        except Exception as ex:
            out["win"][name] = {"error": repr(ex)[:300]}
            continue
        g.update(a=a, b=b, P_theta=phase_period(g["cumphase_med"]), P_theta2=phase_period(g["cumphase_med"], 2))
        try:
            rho, th, _, _ = window_dmd(D[a:b], 2)
            rho4, th4, ev4, _ = window_dmd(D[a:b], 4)
            g.update(dmd2_rho=quantiles(rho), dmd2_theta=quantiles(th), dmd4_rho=quantiles(rho4),
                     dmd4_theta=quantiles(th4))
            tok[name + "_rho2"], tok[name + "_th2"] = rho.float().cpu().numpy(), th.float().cpu().numpy()
            tok[name + "_ev4"] = torch.view_as_real(ev4).float().cpu().numpy()
        except Exception as ex:
            g["dmd_error"] = repr(ex)[:300]
        out["win"][name] = g
    return out, tok


@torch.no_grad()
def plane_jacobian(m, e, s, Q, modes):
    shape = s.shape
    cache = {}
    phi(m, s, e, (), None, cache)
    out = {}
    for name, mode in modes.items():
        f = lambda z: phi(m, z, e, mode, cache, None)
        JQ = torch.stack([jvp(f, s, Q[:, :, i].reshape(shape)).reshape(-1, shape[-1]) for i in range(2)], -1)
        Bm = Q.transpose(1, 2) @ JQ
        leak = (JQ - Q @ Bm).norm(dim=1) / JQ.norm(dim=1).clamp_min(1e-30)
        rho, th, _ = eig_rho_theta(Bm)
        out[name] = (rho, th, leak.mean(-1), Bm)
    return out


def gram_schmidt(a, b):
    q1 = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    b2 = b - (b * q1).sum(-1, keepdim=True) * q1
    return torch.stack([q1, b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-30)], -1)


@torch.no_grad()
def part_B(m, xb, r, K):
    m.double()
    e = m.embed(xb)
    s = initial_state(m, e)
    S = [s]
    for _ in range(K):
        s = m.step(s, e)
        S.append(s)
    shape = s.shape
    St = torch.stack([z.reshape(-1, shape[-1]) for z in S])
    D = St[1:] - St[:-1]
    check = float((phi(m, S[r], e) - m.step(S[r], e)).norm() / m.step(S[r], e).norm())
    planes = {name: window_dmd(D[a:min(b, K)], 2)[3]
              for name, (a, b) in {"w1": (r, 4 * r), "w2": (4 * r, 8 * r), "w3": (8 * r, K)}.items()}
    points = sorted(set([max(1, r // 2), max(1, (3 * r) // 4), r - 1, r, 2 * r, 4 * r, 8 * r, K - 1]))
    res, tok = {"core_check_f64": check, "points": points, "modes": list(MODES)}, {}
    for k in points:
        Q = gram_schmidt(D[k - 1], D[k]) if k < r else planes["w1" if k < 4 * r else ("w2" if k < 8 * r else "w3")]
        rk = {}
        for name, (rho, th, leak, Bm) in plane_jacobian(m, e, S[k], Q, MODES).items():
            rk[name] = {"rho": quantiles(rho), "theta": quantiles(th), "leak": quantiles(leak)}
            tok["B_k%d_%s" % (k, name)] = Bm.float().cpu().numpy()
        f = lambda z: phi(m, z, e)
        v = Q[:, :, 0].reshape(shape)
        j1, j2 = jvp(f, S[k], v, 1e-4), jvp(f, S[k], v, 1e-5)
        rk["fd_rel_diff"] = float((j1 - j2).norm() / j1.norm())
        dk = D[k] if k < K else D[K - 1]
        in_plane = (Q.transpose(1, 2) @ dk[:, :, None])[:, :, 0]
        rk["step_in_plane"] = quantiles(in_plane.norm(dim=-1) / dk.norm(dim=-1).clamp_min(1e-30))
        res["k%d" % k] = rk
    res["conv_last_step_rel"] = float((D[K - 1].norm(dim=-1) / St[r].norm(dim=-1)).median())
    m.float()
    return res, tok


@torch.no_grad()
def part_D(m):
    out = {"decay": quantiles(m.inject.decay().double()), "dt": quantiles(F.softplus(m.inject.dt_bias).double())}
    heads = []
    for i, b in enumerate(m.core):
        at = b.attn
        W = at.qkv.weight.double()
        d, hd = W.shape[1], at.hd
        Wv, Wo, n1 = W[2 * d:3 * d], at.proj.weight.double(), b.n1.weight.double()
        for h in range(at.nh):
            V = Wv[h * hd:(h + 1) * hd] * n1[None, :]
            O = Wo[:, h * hd:(h + 1) * hd]
            ev = torch.linalg.eigvals(V @ O)
            mod = ev.abs()
            top = mod.argsort(descending=True)[:8]
            OV = O @ V
            heads.append({"block": i, "head": h, "imag_frac": float((ev.imag ** 2).sum() / (mod ** 2).sum()),
                          "top8_abs_arg": [float(x) for x in ev[top].angle().abs()],
                          "top8_mod": [float(x) for x in mod[top]],
                          "skew_frac": float(((OV - OV.T) / 2).norm() / OV.norm().clamp_min(1e-30)),
                          "norm_OV": float(OV.norm())})
    out["heads"] = heads
    return out


def guarded(f):
    try:
        return f()
    except Exception as ex:
        return {"error": repr(ex)[:500]}


@torch.no_grad()
def measure(m, cfg, dev, a):
    r = cfg["mean_rec"]
    K = max(16 * r, 128)
    xb, yb = sequences(a.data)
    x, y = xb[: a.seqs, : a.T].to(dev), yb[: a.seqs, : a.T].to(dev)
    xj = xb[: a.jac_seqs, : a.jac_T].to(dev)
    res, tok = {"r": r, "K": K, "seqs": a.seqs, "T": a.T}, {}
    e = m.embed(x)
    s0 = initial_state(m, e)
    s1 = m.step(s0, e)
    res["core_check_f32"] = float((phi(m, s0, e) - s1).norm() / s1.norm())
    t0 = time.time()
    plain = lambda k, s: m.step(s, e)
    losses, St = run_traj(m, y, s0, K, plain)
    windows = {"pre": (0, r), "w1": (r, 4 * r), "w2": (4 * r, 8 * r), "w3": (8 * r, K), "w18": (r, 8 * r)}
    res["A"], tA = traj_summary(losses, St, r, windows)
    tok.update({"A_" + k: v for k, v in tA.items()})
    sr = St[r].reshape(s0.shape).to(s0.dtype)
    del St
    res["secs_A"] = time.time() - t0
    t0 = time.time()
    C = {}

    def after(fn):
        return lambda k, s: plain(k, s) if k < r else fn(k, s)

    for g in (0.5, 0.75, 1.5):
        fn = (lambda g: lambda k, s: s + g * (plain(k, s) - s))(g)
        Kc = min(r + int(math.ceil(6 * r / g)), 16 * r)
        C["relax_after_%g" % g] = guarded(lambda: traj_summary(*run_traj(m, y, s0, Kc, after(fn)), r,
                                                               {"w1": (r, r + int(math.ceil(3 * r / g)))})[0])
    for rp in sorted(set([max(1, r // 2), (3 * r) // 2, 2 * r])):
        g = r / rp
        fn = (lambda g: lambda k, s: s + g * (plain(k, s) - s))(g)
        C["relax_all_rp%d" % rp] = guarded(lambda: traj_summary(*run_traj(m, y, s0, rp + 3 * rp + 4, fn), rp,
                                                                {"w1": (rp, 4 * rp)})[0])
        C["relax_all_rp%d" % rp]["gamma"] = g
    for name, mode in (("freezeP_after", ("P",)), ("freezeA_after", ("attn",)), ("freezeM_after", ("mlp",)),
                       ("freezeN_after", ("norm",))):
        cache = {}
        phi(m, sr, e, (), None, cache)
        fn = (lambda mode, cache: lambda k, s: phi(m, s, e, mode, cache, None))(mode, cache)
        C[name] = guarded(lambda: traj_summary(*run_traj(m, y, s0, 8 * r, after(fn)), r,
                                               {"w1": (r, 4 * r), "w2": (4 * r, 8 * r)})[0])
        del cache
    for v in C.values():
        v.pop("alpha3d_med", None)
    res["C"] = C
    res["secs_C"] = time.time() - t0
    res["D"] = part_D(m)
    t0 = time.time()
    res["B"], tB = part_B(m, xj, r, K)
    tok.update(tB)
    res["secs_B"] = time.time() - t0
    return res, tok


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--jac_seqs", type=int, default=2)
    ap.add_argument("--jac_T", type=int, default=256)
    a = ap.parse_args()
    for_each_run(a, "plane_dynamics.json", lambda m, cfg, dev: measure(m, cfg, dev, a),
                 keep=lambda cfg, name: cfg["mean_rec"] >= 4)


if __name__ == "__main__":
    main()
