import math

DEFAULTS = dict(
    d=640, heads=10, ffn=2560, prelude=2, core=2, coda=2,
    mean_rec=8, mean_bptt=8, sampling="fixed", loss_every_loop=0,
    decay_init=math.sqrt(1 / 5), decay_frozen=0, update_scale=1.0,
    value_embeddings=1, qk_norm_style="parcae", init_style="parcae",
    vocab=32768, T=1024, batch=64, micro=32, tokens=5e8, seed=0,
    optimizer="adamw", lr=1e-3, wd=0.1, betas=(0.9, 0.95), eps=1e-8, warmup=0.01, cooldown=0.5, clip=1.0,
    muon_wd=0.2, muon_mom=0.95,
    eval_every=500, eval_batches=20, log_every=20,
    eval_depths=(1, 2, 4, 6, 8, 12, 16, 24, 32), name="",
)

PRESETS = {
    "base": {},
    "muon": dict(optimizer="muon", lr=8e-3, betas=(0.8, 0.95), eps=1e-10, wd=0.0, warmup=0.0, batch=512),
    "width768": dict(d=768, heads=6, ffn=3072, value_embeddings=0, qk_norm_style="learned", init_style="simple",
                     tokens=1e9),
}

DEFAULT_TOKENS = {"base": 5e8, "muon": 5e8, "width768": 1e9}


def run_name(cfg):
    name = "%s_r%d_b%d" % ("fixed" if cfg["sampling"] == "fixed" else "poisson", cfg["mean_rec"], cfg["mean_bptt"])
    if cfg["d"] != 640:
        name += "_d%d" % cfg["d"]
    if cfg["optimizer"] != "adamw":
        name += "_" + cfg["optimizer"]
    if abs(cfg["decay_init"] - math.sqrt(1 / 5)) > 1e-9:
        name += "_dec%g" % cfg["decay_init"]
    if cfg["loss_every_loop"]:
        name += "_all"
    if cfg["update_scale"] != 1.0:
        name += "_us%g" % cfg["update_scale"]
    if cfg["decay_frozen"]:
        name += "_dfz"
    if cfg["tokens"] != DEFAULT_TOKENS[cfg["preset"]]:
        name += "_D%dM" % round(cfg["tokens"] / 1e6)
    return name + "_s%d" % cfg["seed"]


def resolve(preset, overrides):
    cfg = dict(DEFAULTS)
    cfg.update(PRESETS[preset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    cfg["preset"] = preset
    cfg["eval_depths"] = tuple(cfg["eval_depths"])
    cfg["betas"] = tuple(cfg["betas"])
    if not cfg["name"]:
        cfg["name"] = run_name(cfg)
    return cfg


def from_original(c):
    cfg = dict(DEFAULTS)
    for k in ("d", "heads", "ffn", "prelude", "core", "coda", "mean_rec", "mean_bptt", "decay_init", "vocab", "T",
              "batch", "micro", "tokens", "seed", "optimizer", "lr", "wd", "eps", "warmup", "cooldown", "clip",
              "muon_wd", "muon_mom"):
        if c.get(k) is not None:
            cfg[k] = c[k]
    cfg["sampling"] = "fixed" if c["sampling"] == "fixed" else "poisson"
    cfg["loss_every_loop"] = int(c.get("loss_all") or 0)
    cfg["decay_frozen"] = int(c.get("decay_frozen") or 0)
    cfg["update_scale"] = float(c.get("update_scale") or 1.0)
    cfg["value_embeddings"] = int(c.get("ve") or 0)
    cfg["qk_norm_style"] = c.get("qk_norm_style") or "learned"
    cfg["init_style"] = c.get("init_style") or "simple"
    cfg["betas"] = tuple(c.get("betas") or DEFAULTS["betas"])
    cfg["preset"] = "width768" if cfg["d"] == 768 else ("muon" if cfg["optimizer"] == "muon" else "base")
    cfg["name"] = run_name(cfg)
    return cfg
