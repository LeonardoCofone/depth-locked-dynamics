import torch

POLAR_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, groups):
        super().__init__(groups, defaults={})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            if g["kind"] == "adamw":
                self.adamw_step(g)
            else:
                self.muon_step(g)

    def adamw_step(self, g):
        b1, b2 = g["betas"]
        for p in g["params"]:
            if p.grad is None:
                continue
            s = self.state[p]
            if not s:
                s["step"], s["m"], s["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
            s["step"] += 1
            p.mul_(1 - g["lr"] * g["weight_decay"])
            s["m"].lerp_(p.grad, 1 - b1)
            s["v"].lerp_(p.grad.square(), 1 - b2)
            bc1, bc2 = 1 - b1 ** s["step"], 1 - b2 ** s["step"]
            p.add_(s["m"] / bc1 / ((s["v"] / bc2).sqrt() + g["eps"]), alpha=-g["lr"])

    def muon_step(self, g):
        ps = g["params"]
        p0 = ps[0]
        shape = p0.shape
        rd = -1 if shape[-2] >= shape[-1] else -2
        s = self.state[p0]
        if "m" not in s:
            s["m"] = torch.zeros(len(ps), *shape, dtype=p0.dtype, device=p0.device)
            vs = (len(ps), shape[-2], 1) if shape[-2] >= shape[-1] else (len(ps), 1, shape[-1])
            s["v"] = torch.zeros(vs, dtype=p0.dtype, device=p0.device)
        grad = torch.stack([p.grad if p.grad is not None else torch.zeros_like(p) for p in ps])
        stacked = torch.stack(ps)
        mom, wd = g["momentum"], g["weight_decay"]
        lr = g["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5
        s["m"].lerp_(grad, 1 - mom)
        grad = grad.lerp_(s["m"], mom)
        X = grad.bfloat16()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
        if grad.size(-2) > grad.size(-1):
            for a, b, c in POLAR_COEFFS:
                A = X.mT @ X
                X = a * X + X @ (b * A + c * (A @ A))
        else:
            for a, b, c in POLAR_COEFFS:
                A = X @ X.mT
                X = a * X + (b * A + c * (A @ A)) @ X
        u = X.float()
        v_mean = u.square().mean(dim=rd, keepdim=True)
        v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * u.size(rd)).sqrt()
        s["v"].lerp_(v_mean.to(s["v"].dtype), 1 - g["beta2"])
        scale = s["v"].clamp_min(1e-10).rsqrt()
        v_new = ((v_mean * u.size(rd)) * scale.float().square()).sum(dim=(-2, -1), keepdim=True).sqrt()
        u = u * (scale * (v_norm / v_new.clamp_min(1e-10)))
        mask = (u * stacked) >= 0
        stacked.sub_(lr * u + lr * wd * stacked * mask)
        torch._foreach_copy_(ps, list(stacked.unbind(0)))


def make_optimizer(model, cfg, dev):
    if cfg["optimizer"] == "adamw":
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            (no_decay if p.dim() < 2 or n.startswith("inject.") else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"], "kind": "adamw"},
                                  {"params": no_decay, "weight_decay": 0.0, "kind": "adamw"}],
                                 lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=cfg["eps"], fused=dev == "cuda")
    adam, muon = [], {}
    for n, p in model.named_parameters():
        if (p.dim() < 2 or n.startswith("inject.") or n.startswith("value_embeds.")
                or n in ("wte.weight", "C.weight")):
            adam.append(p)
        else:
            muon.setdefault(tuple(p.shape), []).append(p)
    groups = [{"params": adam, "kind": "adamw", "lr": cfg["lr"], "betas": tuple(cfg["betas"]), "eps": cfg["eps"],
               "weight_decay": cfg["wd"]}]
    for ps in muon.values():
        groups.append({"params": ps, "kind": "muon", "lr": cfg["lr"], "momentum": cfg["muon_mom"],
                       "weight_decay": cfg["muon_wd"], "beta2": 0.95})
    return MuonAdamW(groups)
