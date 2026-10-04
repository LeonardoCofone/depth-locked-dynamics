import numpy as np
import torch
import torch.nn.functional as F

from measure.common import base_parser, for_each_run, sequences, state_generator

FIELDS = ("loss", "acc", "entropy", "state_norm", "step_norm", "rel_step", "cos_prev_step", "step_ratio", "cos_to_r",
          "reldist_to_r")


class Means:
    def __init__(self, n):
        self.sums = [dict() for _ in range(n)]

    def add(self, k, name, t):
        d = self.sums[k]
        d[name] = d.get(name, 0.0) + float(t.double().sum())
        d["n_" + name] = d.get("n_" + name, 0) + t.numel()

    def result(self):
        return [dict({"k": k}, **{f: d[f] / max(1, d["n_" + f]) for f in FIELDS if f in d})
                for k, d in enumerate(self.sums)]


@torch.no_grad()
def measure_chunk(m, x, y, K, amp, means, keep):
    dev = x.device
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
        S = m.trajectory(x, K, gen=state_generator(dev))[1]
    r = min(m.cfg["mean_rec"], len(S) - 1)
    sr = S[r].float()
    nr = sr.norm(dim=-1)
    for k in range(len(S)):
        s = S[k].float()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            logits = m.readout(S[k]).float()
        logp = F.log_softmax(logits, -1)
        loss = -logp.gather(-1, y[..., None])[..., 0]
        p = logp.exp()
        nk = s.norm(dim=-1)
        means.add(k, "loss", loss)
        means.add(k, "acc", (p.argmax(-1) == y).float())
        means.add(k, "entropy", -(p * logp).sum(-1))
        means.add(k, "state_norm", nk)
        means.add(k, "cos_to_r", F.cosine_similarity(s, sr, dim=-1))
        means.add(k, "reldist_to_r", (s - sr).norm(dim=-1) / nr)
        if keep is not None:
            keep.setdefault("loss", []).append(loss.cpu())
        if k >= 1:
            prev = S[k - 1].float()
            step = (s - prev).norm(dim=-1)
            means.add(k, "step_norm", step)
            means.add(k, "rel_step", step / prev.norm(dim=-1).clamp_min(1e-12))
            if keep is not None:
                keep.setdefault("step_norm", []).append(step.cpu())
            if k >= 2:
                d_prev = prev - S[k - 2].float()
                means.add(k, "cos_prev_step", F.cosine_similarity(s - prev, d_prev, dim=-1))
                means.add(k, "step_ratio", step / d_prev.norm(dim=-1).clamp_min(1e-12))


def measure(m, cfg, dev, a):
    amp = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    x, y = sequences(a.data, n=a.seqs)
    K = max(a.maxk, 8 * cfg["mean_rec"])
    means, keep = Means(K + 1), {}
    for i in range(0, x.shape[0], a.chunk):
        measure_chunk(m, x[i:i + a.chunk].to(dev), y[i:i + a.chunk].to(dev), K, amp, means,
                      keep if i < a.keep_seqs else None)
    loss_tok = torch.stack(keep["loss"]).numpy().astype(np.float16)
    step_tok = torch.stack(keep["step_norm"]).numpy().astype(np.float16)
    step_median = np.median(step_tok.astype(np.float64).reshape(K, -1), 1)
    res = {"r": cfg["mean_rec"], "maxk": K, "config": cfg, "per_k": means.result(),
           "step_norm_median": [None] + step_median.tolist()}
    return res, {"token_loss": loss_tok, "token_step_norm": step_tok}


def main():
    ap = base_parser()
    ap.add_argument("--seqs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--keep_seqs", type=int, default=8)
    ap.add_argument("--maxk", type=int, default=64)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for_each_run(a, "curves.json", lambda m, cfg, dev: measure(m, cfg, dev, a))


if __name__ == "__main__":
    main()
