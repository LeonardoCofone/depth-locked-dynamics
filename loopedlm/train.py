import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from loopedlm.config import DEFAULTS, PRESETS, resolve
from loopedlm.data import Batches, fixed_batches
from loopedlm.model import LoopedLM
from loopedlm.optim import make_optimizer


def sample_depths(cfg, B, gen):
    if cfg["sampling"] == "fixed":
        k = min(cfg["mean_bptt"], cfg["mean_rec"])
        return cfg["mean_rec"] - k, k, None, None, [cfg["mean_rec"]] * B
    total = torch.clamp(torch.poisson(torch.full((B,), float(cfg["mean_rec"])), generator=gen), min=1).long()
    k = torch.clamp(total, max=cfg["mean_bptt"])
    n = total - k
    N, K = int(n.max()), int(k.max())
    return N, K, [(t < n) for t in range(N)], [(t < k) for t in range(K)], total.tolist()


@torch.no_grad()
def eval_depths(model, batches, depths, dev, amp):
    model.eval()
    res = {}
    for depth in depths:
        total = 0.0
        g = torch.Generator(device=dev).manual_seed(7)
        for x, y in batches:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
                logits = model(x, 0, depth, gen=g)
            total += float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)))
        res[str(depth)] = total / len(batches)
    model.train()
    return res


def lr_at(cfg, it, steps):
    w, cd = int(cfg["warmup"] * steps), int(cfg["cooldown"] * steps)
    if it < w:
        return cfg["lr"] * (it + 1) / max(1, w)
    if it >= steps - cd:
        return cfg["lr"] * (steps - it) / max(1, cd)
    return cfg["lr"]


def train(cfg, data, runs):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cfg["vocab"] = json.load(open(os.path.join(data, "manifest.json")))["vocab"]
    run = os.path.join(runs, cfg["name"])
    info_path = os.path.join(run, "run_info.json")
    if os.path.exists(info_path) and json.load(open(info_path)).get("done"):
        print("already trained:", run, flush=True)
        return
    os.makedirs(run, exist_ok=True)
    train_tokens = np.load(os.path.join(data, "train.npy"), mmap_mode="r")
    val_tokens = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    torch.manual_seed(cfg["seed"])
    model = LoopedLM(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, cfg, dev)
    steps = int(cfg["tokens"] // (cfg["batch"] * cfg["T"]))
    accum = max(1, cfg["batch"] // cfg["micro"])
    batches = Batches(train_tokens, cfg["T"], 1000 + cfg["seed"])
    depth_gen = torch.Generator().manual_seed(2000 + cfg["seed"])
    state_gen = torch.Generator(device=dev).manual_seed(3000 + cfg["seed"])
    eval_set = fixed_batches(val_tokens, cfg["T"], cfg["eval_batches"], cfg["micro"], 123)
    start = 0
    latest = os.path.join(run, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        batches.g.set_state(ck["data_rng"])
        depth_gen.set_state(ck["depth_rng"])
        state_gen.set_state(ck["state_rng"])
        torch.set_rng_state(ck["torch_rng"])
        start = ck["step"]
    else:
        json.dump({"config": cfg, "steps": steps, "accum": accum, "n_params": n_params,
                   "n_params_nonembed": n_params - model.wte.weight.numel()},
                  open(info_path, "w"), indent=1)
    train_log = open(os.path.join(run, "train_log.jsonl"), "a")
    eval_log = open(os.path.join(run, "eval_log.jsonl"), "a")
    t0, seen = time.time(), 0
    model.train()
    for it in range(start, steps):
        lr = lr_at(cfg, it, steps)
        for group in opt.param_groups:
            group["lr"] = lr
            if group.get("kind") == "muon":
                group["weight_decay"] = max(0.0, cfg["muon_wd"] * (1 - it / steps))
        step_loss, depths_used = 0.0, []
        for _ in range(accum):
            x, y = batches.next(cfg["micro"])
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
                every = None
                if cfg["loss_every_loop"]:
                    depths_used += [cfg["mean_rec"]] * x.shape[0]
                    every = model.forward_every_loop(x, state_gen)
                    logits = every[-1]
                else:
                    N, K, an, ak, total = sample_depths(cfg, x.shape[0], depth_gen)
                    depths_used += total
                    if an is not None:
                        an, ak = [m.to(dev) for m in an], [m.to(dev) for m in ak]
                    logits = model(x, N, K, an, ak, state_gen)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                if every is not None:
                    objective = sum(F.cross_entropy(l.float().reshape(-1, l.shape[-1]), y.reshape(-1))
                                    for l in every) / len(every)
                else:
                    objective = loss
            (objective / accum).backward()
            step_loss += loss.item() / accum
            seen += x.numel()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not math.isfinite(step_loss):
            print("non-finite loss at step %d" % it, flush=True)
            sys.exit(3)
        if it % cfg["log_every"] == 0 or it == steps - 1:
            elapsed = time.time() - t0
            decay = model.inject.decay().detach().float()
            row = {"step": it, "loss": step_loss, "lr": lr, "grad_norm": grad_norm, "time": elapsed,
                   "tok_per_s": seen / max(elapsed, 1e-9), "depth_mean": float(np.mean(depths_used)),
                   "decay_mean": float(decay.mean()), "decay_max": float(decay.max()), "decay_min": float(decay.min())}
            train_log.write(json.dumps(row) + "\n")
            train_log.flush()
            print("step %d/%d loss %.4f" % (it, steps, step_loss), flush=True)
        if (it + 1) % cfg["eval_every"] == 0 or it == steps - 1:
            res = eval_depths(model, eval_set, cfg["eval_depths"], dev, amp)
            row = {"step": it + 1, "tokens": (it + 1) * cfg["batch"] * cfg["T"], "val": res}
            eval_log.write(json.dumps(row) + "\n")
            eval_log.flush()
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": it + 1,
                        "data_rng": batches.g.get_state(), "depth_rng": depth_gen.get_state(),
                        "state_rng": state_gen.get_state(), "torch_rng": torch.get_rng_state()}, latest + ".tmp")
            os.replace(latest + ".tmp", latest)
    torch.save({"model": model.state_dict(), "config": cfg}, os.path.join(run, "model.pt"))
    info = json.load(open(info_path))
    info["done"] = True
    json.dump(info, open(info_path, "w"), indent=1)
    if os.path.exists(latest):
        os.remove(latest)
    print("trained:", run, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=list(PRESETS))
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", default="runs")
    for k, v in DEFAULTS.items():
        if isinstance(v, tuple):
            ap.add_argument("--" + k, type=type(v[0]), nargs="+", default=None)
        elif k != "vocab":
            ap.add_argument("--" + k, type=type(v), default=None)
    a = ap.parse_args()
    overrides = {k: getattr(a, k) for k in DEFAULTS if k != "vocab"}
    train(resolve(a.preset, overrides), a.data, a.runs)


if __name__ == "__main__":
    main()
