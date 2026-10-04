import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from measure.common import device, lyapunov, non_normality, sequences, window_geometry


def load_ouro(path, dev):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if int(transformers.__version__.split(".")[0]) >= 5:
        raise SystemExit("the Ouro code needs transformers 4.5x (tested with 4.56.2)")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.float32,
                                             attn_implementation="sdpa").to(dev).eval()
    return m, tok


def make_step(m, T, dev):
    core = m.model
    pos = torch.arange(T, device=dev)[None]
    mask = torch.full((T, T), float("-inf"), device=dev).triu(1)[None, None]

    def step(s):
        pe = core.rotary_emb(s, pos)
        h = s
        for layer in core.layers[: core.config.num_hidden_layers]:
            h = layer(h, attention_mask=mask.to(h.dtype), position_ids=pos, past_key_value=None, use_cache=False,
                      cache_position=None, position_embeddings=pe)
            if isinstance(h, tuple):
                h = h[0]
        return core.norm(h)
    return step


def retokenized(data, tok, nseq, T):
    from tokenizers import Tokenizer
    ours = Tokenizer.from_file(os.path.join(data, "tokenizer.json"))
    xb = sequences(data)[0]
    ids = []
    for i in range(nseq):
        enc = tok(ours.decode([int(t) for t in xb[i]]), add_special_tokens=False)["input_ids"]
        if len(enc) < T + 1:
            raise SystemExit("sequence %d too short after retokenisation (%d)" % (i, len(enc)))
        ids.append(enc[: T + 1])
    return torch.tensor(ids)


def load_sequences(model, data, seqs, T, dev):
    m, tok = load_ouro(model, dev)
    xy = retokenized(data, tok, seqs, T).to(dev)
    return m, xy[:, :-1], xy[:, 1:]


@torch.no_grad()
def run(m, x, y, K, r, n_probes, gen):
    dev = x.device
    T = x.shape[1]
    step = make_step(m, T, dev)
    e = m.model.embed_tokens(x).float()
    hidden = m.model(input_ids=x, use_cache=False)[1]
    s, S, losses = e, [e], []
    for _ in range(K):
        s = step(s)
        S.append(s)
        losses.append(float(F.cross_entropy(m.lm_head(s).float().flatten(0, 1), y.flatten())))
    check = [float((S[k + 1] - hidden[k].float()).norm() / hidden[k].float().norm()) for k in range(len(hidden))]
    Tl = min(256, T)
    lyap, _ = lyapunov([make_step(m, Tl, dev)] * min(K, 12), e[:, :Tl].contiguous(), gen)
    xs = x[:2, :256]
    short = make_step(m, xs.shape[1], dev)
    s2 = m.model.embed_tokens(xs).float()
    nu = {}
    for k in range(1, 13):
        s2 = short(s2)
        nu[str(k)] = non_normality(short, s2, n_probes, gen)
    St = torch.stack(S).reshape(K + 1, -1, S[0].shape[-1])
    D = St[1:] - St[:-1]
    dist = [float((St[k] - St[r]).norm(dim=-1).mean() / St[r].norm(dim=-1).mean()) for k in range(K + 1)]
    Dd = D.double()
    G = torch.einsum("and,bnd->ab", Dd, Dd) / Dd.shape[1] / float((St[r].double() ** 2).sum(-1).mean())
    return {"r": r, "K": K, "loss": losses, "lyap": lyap.tolist(), "nu": nu, "dist_from_sr": dist,
            "check_vs_forward": check, "G": G.cpu().numpy().round(8).tolist(), "post": window_geometry(D[r:]),
            "pre": window_geometry(D[:r])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local folder or hub id of an Ouro model")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/public")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--n", type=int, default=8, help="Hutchinson probes for nu")
    a = ap.parse_args()
    dev = device()
    t0 = time.time()
    m, x, y = load_sequences(a.model, a.data, a.seqs, a.T, dev)
    r = int(getattr(m.config, "total_ut_steps", 4))
    res = run(m, x, y, a.K, r, a.n, torch.Generator(device=dev).manual_seed(0))
    name = os.path.basename(a.model.rstrip("/"))
    res.update(model=name, secs=time.time() - t0)
    dst = os.path.join(a.out, name, "ouro_dynamics.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, "w"))


if __name__ == "__main__":
    main()
