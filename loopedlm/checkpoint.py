import glob
import json
import os
import re

import torch

from loopedlm.config import from_original
from loopedlm.model import LoopedLM


def weights_file(run):
    final = os.path.join(run, "model.pt")
    if os.path.exists(final):
        return final
    steps = glob.glob(os.path.join(run, "ckpt_step*.pt"))
    return max(steps, key=lambda p: int(re.search(r"ckpt_step(\d+)\.pt", p).group(1)))


def run_config(run):
    cfg = json.load(open(os.path.join(run, "run_info.json")))["config"]
    return cfg if "loss_every_loop" in cfg else from_original(cfg)


def load_model(run, dev, dtype=torch.float32):
    cfg = run_config(run)
    model = LoopedLM(cfg)
    state = torch.load(weights_file(run), map_location="cpu", weights_only=False)["model"]
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    return model.to(device=dev, dtype=dtype).eval().requires_grad_(False), cfg


def trained_runs(runs, only=""):
    out = []
    for run in sorted(glob.glob(os.path.join(runs, "*"))):
        p = os.path.join(run, "run_info.json")
        if os.path.exists(p) and json.load(open(p)).get("done") and only in os.path.basename(run):
            out.append(run)
    return out
