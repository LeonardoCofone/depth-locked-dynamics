import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VALUE_GATE_CHANNELS = 32


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_cache(head_dim, length, base=50000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(length).float(), inv)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]].to(x.dtype), sin[: x.shape[-2]].to(x.dtype)
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg, value_embedding=False):
        super().__init__()
        self.nh, self.hd = cfg["heads"], cfg["d"] // cfg["heads"]
        self.qkv = nn.Linear(cfg["d"], 3 * cfg["d"], bias=False)
        self.proj = nn.Linear(cfg["d"], cfg["d"], bias=False)
        self.qk_style = cfg["qk_norm_style"]
        if self.qk_style == "learned":
            self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)
        self.ve_gate = nn.Linear(VALUE_GATE_CHANNELS, self.nh, bias=False) if value_embedding else None

    def forward(self, x, cos, sin, ve=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).unbind(2)
        if self.ve_gate is not None and ve is not None:
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :VALUE_GATE_CHANNELS]))
            v = v + gate.unsqueeze(-1) * ve.view(B, T, self.nh, self.hd)
        if self.qk_style == "learned":
            q, k = self.qn(q), self.kn(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.qk_style == "parcae":
            q, k = F.rms_norm(q, (self.hd,)), F.rms_norm(k, (self.hd,))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg["d"], cfg["ffn"], bias=False)
        self.proj = nn.Linear(cfg["ffn"], cfg["d"], bias=False)

    def forward(self, x):
        return self.proj(F.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, cfg, value_embedding=False):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg["d"]), Attention(cfg, value_embedding)
        self.n2, self.mlp = RMSNorm(cfg["d"]), MLP(cfg)

    def forward(self, x, cos, sin, ve=None):
        x = x + self.attn(self.n1(x), cos, sin, ve)
        return x + self.mlp(self.n2(x))


@torch.no_grad()
def trunc_orthogonal_(t, gain):
    rows, cols = t.shape[0], t.numel() // t.shape[0]
    f = t.new_empty(rows, cols)
    nn.init.trunc_normal_(f, mean=0.0, std=1.0)
    if rows < cols:
        f.t_()
    q, r = torch.linalg.qr(f)
    q *= torch.diag(r, 0).sign()
    if rows < cols:
        q.t_()
    t.view_as(q).copy_(q)
    t.mul_(gain)


class Injection(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d"]
        dt0 = -math.log(cfg["decay_init"])
        a_log, dt_bias = torch.zeros(d), torch.full((d,), dt0 + math.log(-math.expm1(-dt0)))
        if cfg["decay_frozen"]:
            self.register_buffer("A_log", a_log)
            self.register_buffer("dt_bias", dt_bias)
        else:
            self.A_log = nn.Parameter(a_log)
            self.dt_bias = nn.Parameter(dt_bias)
        self.B = nn.Parameter(torch.eye(d))

    def decay(self):
        return torch.exp(-F.softplus(self.dt_bias) * torch.exp(self.A_log))

    def forward(self, s, e):
        dt = F.softplus(self.dt_bias)
        return s * torch.exp(-dt * torch.exp(self.A_log)) + dt * (e @ self.B.t())


class LoopedLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["d"]
        self.wte = nn.Embedding(cfg["vocab"], d)
        cos, sin = rope_cache(d // cfg["heads"], cfg["T"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        n_layers = cfg["prelude"] + cfg["core"] + cfg["coda"]
        self.ve_idx = [i for i in range(n_layers) if cfg["value_embeddings"] and i % 2 == (n_layers - 1) % 2]
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(cfg["vocab"], d) for i in self.ve_idx})
        has_ve = lambda i: i in self.ve_idx
        P, C = cfg["prelude"], cfg["core"]
        self.prelude = nn.ModuleList(Block(cfg, has_ve(i)) for i in range(P))
        self.ln_prelude = RMSNorm(d)
        self.inject = Injection(cfg)
        self.core = nn.ModuleList(Block(cfg, has_ve(P + i)) for i in range(C))
        self.C = nn.Linear(d, d, bias=False)
        self.coda = nn.ModuleList(Block(cfg, has_ve(P + C + i)) for i in range(cfg["coda"]))
        self.ln_f = RMSNorm(d)
        self.offset = {"prelude": 0, "core": P, "coda": P + C}
        parcae_init = cfg["init_style"] == "parcae"
        self.emb_std = math.sqrt(2 / (5 * d)) if parcae_init else 0.02
        self.current_ve = {}
        self.reset_parameters()

    def reset_parameters(self):
        d = self.cfg["d"]
        parcae_init = self.cfg["init_style"] == "parcae"
        std = math.sqrt(2 / (5 * d)) if parcae_init else 0.5 / math.sqrt(d)
        for n, p in self.named_parameters():
            if p.dim() != 2 or n in ("inject.B", "C.weight") or n.startswith("value_embeds.") or "ve_gate" in n:
                continue
            if n.endswith("proj.weight"):
                nn.init.zeros_(p)
            elif parcae_init:
                if n.endswith("qkv.weight"):
                    for part in p.data.chunk(3, 0):
                        trunc_orthogonal_(part, std * math.sqrt(max(part.shape)))
                else:
                    trunc_orthogonal_(p.data, std * math.sqrt(max(p.shape)))
            elif n == "wte.weight":
                nn.init.normal_(p, std=0.02)
            else:
                nn.init.uniform_(p, -std * math.sqrt(3), std * math.sqrt(3))
        nn.init.eye_(self.C.weight)

    def head(self, h):
        return F.linear(self.ln_f(h), self.wte.weight)

    def run_blocks(self, blocks, part, h):
        o = self.offset[part]
        for i, b in enumerate(blocks):
            h = b(h, self.cos, self.sin, self.current_ve.get(o + i))
        return h

    def embed(self, x):
        self.current_ve = {i: self.value_embeds[str(i)](x) for i in self.ve_idx}
        return self.ln_prelude(self.run_blocks(self.prelude, "prelude", self.wte(x)))

    def init_state(self, e, gen=None):
        s = torch.randn(e.shape, device=e.device, dtype=e.dtype, generator=gen)
        return torch.clamp(s, -3, 3) * self.emb_std

    def core_map(self, s, e):
        return self.run_blocks(self.core, "core", self.inject(s, e))

    def step(self, s, e):
        out = self.core_map(s, e)
        g = self.cfg["update_scale"]
        if g != 1.0:
            out = s + g * (out - s)
        return out

    def readout(self, s):
        return self.head(self.run_blocks(self.coda, "coda", self.C(s)))

    def forward(self, x, n_nograd, n_grad, active_nograd=None, active_grad=None, gen=None):
        e = self.embed(x)
        s = self.init_state(e, gen)
        with torch.no_grad():
            for t in range(n_nograd):
                new = self.step(s, e)
                s = new if active_nograd is None else torch.where(active_nograd[t][:, None, None], new, s)
        for t in range(n_grad):
            new = self.step(s, e)
            s = new if active_grad is None else torch.where(active_grad[t][:, None, None], new, s)
        return self.readout(s)

    def forward_every_loop(self, x, gen=None):
        r = self.cfg["mean_rec"]
        kb = min(self.cfg["mean_bptt"], r)
        e = self.embed(x)
        s = self.init_state(e, gen)
        out = []
        for t in range(r):
            if t < r - kb:
                with torch.no_grad():
                    s = self.step(s, e)
                    out.append(self.readout(s))
            else:
                s = self.step(s, e)
                out.append(self.readout(s))
        return out

    @torch.no_grad()
    def trajectory(self, x, depth, gen=None):
        e = self.embed(x)
        s = self.init_state(e, gen)
        out = [s]
        for t in range(depth):
            s = self.step(s, e)
            out.append(s)
        return e, out
