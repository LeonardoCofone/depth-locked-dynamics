import argparse
import json
import os
import sys
import time
import types

import torch
import torch.nn.functional as F

from measure.common import device, lyapunov, non_normality, sequences, window_geometry


class Parcae:
    def __init__(self, path, src, tokenizer, dev):
        sys.path.insert(0, src)
        import typing
        if not hasattr(typing, "Self"):
            import typing_extensions
            typing.Self = typing_extensions.Self
        ops = types.ModuleType("parcae_lm.ops")
        ops.LinearCrossEntropyLoss = None
        ops.linear_cross_entropy = None
        sys.modules["parcae_lm.ops"] = ops
        import parcae_lm.attention_backends.flash_attention as fa
        fa._override_impl = "sdpa"
        from parcae_lm.models.parcae.config import ParcaeConfig
        cd = json.load(open(os.path.join(path, "config.json")))
        if cd.get("use_fused_head", "pytorch") not in ("pytorch", None):
            raise SystemExit("unsupported head %s" % cd.get("use_fused_head"))
        cd.pop("_class_name", None)
        cd.pop("rope_settings", None)
        cfg = ParcaeConfig(**cd)
        m = cfg.construct_model()
        sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        res = m.load_state_dict(sd, strict=False)
        self.load_report = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys)}
        self.m = m.float().to(dev).eval().requires_grad_(False)
        self.r = int(cfg.mean_recurrence)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(tokenizer, "tokenizer.json"))

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def prepare(self, x):
        m = self.m
        f = m.freqs_cis[:, : x.shape[1]]
        e = m.transformer.wte(x)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        for i, b in enumerate(m.transformer.prelude):
            ve = m.value_embeds[str(i)](x) if str(i) in m.value_embeds else None
            e = b(e, f, None, ve=ve)
        if m.config.prelude_norm:
            e = m.transformer.ln_prelude(e)
        return {"x": x, "e": e, "f": f}

    def init_state(self, ctx, seed):
        torch.manual_seed(seed)
        return self.m.initialize_state(ctx["e"])

    def step(self, ctx, s, k):
        self.m._current_input_ids = ctx["x"]
        return self.m.core_block_forward(s, ctx["e"], ctx["f"], None, torch.tensor(k), torch.tensor(0))

    def loss(self, ctx, s, y):
        m, f, x = self.m, ctx["f"], ctx["x"]
        h = m.transformer.C(s)
        off = m.config.n_layers_in_prelude + m.config.n_layers_in_recurrent_block
        for i, b in enumerate(m.transformer.coda):
            ve = m.value_embeds[str(off + i)](x) if str(off + i) in m.value_embeds else None
            h = b(h, f, None, ve=ve)
        logits = m.lm_head(m.transformer.ln_f(h)).float() * m.config.init.logit_scale
        if m.config.logit_softcap is not None:
            logits = m.config.logit_softcap * torch.tanh(logits / m.config.logit_softcap)
        return float(F.cross_entropy(logits.flatten(0, 1), y.flatten()))

    def official(self, x, y, r, seed):
        torch.manual_seed(seed)
        return float(self.m(x, labels=y, num_steps_pair=torch.tensor([r, 0]))["loss"])


class Huginn:
    def __init__(self, path, dev):
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.float32).to(dev).eval()
        self.m.requires_grad_(False)
        self.load_report = {}
        self.r = int(self.m.config.mean_recurrence)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def prepare(self, x):
        m = self.m
        f = m.freqs_cis[:, : x.shape[1]]
        e = m.transformer.wte(x)
        if m.emb_scale != 1:
            e = e * m.emb_scale
        bi = torch.tensor(-1, dtype=torch.long)
        for b in m.transformer.prelude:
            bi += 1
            e = b(e, f, bi, None, None)
        return {"x": x, "e": e, "f": f, "bi": bi}

    def init_state(self, ctx, seed):
        torch.manual_seed(seed)
        return self.m.initialize_state(ctx["e"])

    def step(self, ctx, s, k):
        return self.m.core_block_forward(s, ctx["e"], ctx["f"], None, None, ctx["bi"], k)[0]

    def loss(self, ctx, s, y):
        logits = self.m.predict_from_latents(s).logits
        return float(F.cross_entropy(logits.flatten(0, 1).float(), y.flatten()))

    def official(self, x, y, r, seed):
        torch.manual_seed(seed)
        return float(self.m(x, labels=y, num_steps=torch.tensor([r, 0])).loss)


def retokenized(data, encode, nseq, T):
    from tokenizers import Tokenizer
    ours = Tokenizer.from_file(os.path.join(data, "tokenizer.json"))
    xb = sequences(data)[0]
    ids = []
    for i in range(nseq):
        enc = encode(ours.decode([int(t) for t in xb[i]]))
        if len(enc) < T + 1:
            raise SystemExit("sequence %d too short after retokenisation (%d)" % (i, len(enc)))
        ids.append(enc[: T + 1])
    return torch.tensor(ids)


@torch.no_grad()
def run(A, x, y, K, n_probes, gen, seed=7):
    r = A.r
    ctx = A.prepare(x)
    s = A.init_state(ctx, seed)
    S, losses = [s], []
    for k in range(K):
        s = A.step(ctx, s, k)
        S.append(s)
        losses.append(A.loss(ctx, s, y))
    official = A.official(x, y, r, seed)
    cs = A.prepare(x[:, : min(256, x.shape[1])])
    lyap, _ = lyapunov([(lambda k: (lambda z: A.step(cs, z, k)))(k) for k in range(min(K, 2 * r))],
                       A.init_state(cs, seed), gen)
    c2 = A.prepare(x[:2, : min(256, x.shape[1])])
    s2 = A.init_state(c2, seed)
    nu = {}
    for k in range(3 * r // 2):
        s2 = A.step(c2, s2, k)
        nu[str(k + 1)] = non_normality((lambda kk: (lambda z: A.step(c2, z, kk)))(k + 1), s2, n_probes, gen)
    St = torch.stack(S).reshape(K + 1, -1, S[0].shape[-1])
    D = St[1:] - St[:-1]
    dist = [float((St[k] - St[r]).norm(dim=-1).mean() / St[r].norm(dim=-1).mean()) for k in range(K + 1)]
    Dd = D.double()
    G = torch.einsum("and,bnd->ab", Dd, Dd) / Dd.shape[1] / float((St[r].double() ** 2).sum(-1).mean())
    return {"r": r, "K": K, "loss": losses, "official_loss_at_r": official, "check_abs": abs(official - losses[r - 1]),
            "lyap": lyap.tolist(), "nu": nu, "dist_from_sr": dist, "G": G.cpu().numpy().round(8).tolist(),
            "post": window_geometry(D[r:]), "pre": window_geometry(D[:r])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["parcae", "huginn"])
    ap.add_argument("--model", required=True, help="local folder with the released weights")
    ap.add_argument("--src", default="", help="parcae: clone of the Parcae source repository")
    ap.add_argument("--tokenizer", default="", help="parcae: folder with the Parcae tokenizer.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/public")
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--K", type=int, default=0, help="0 = 4r for parcae, 2r for huginn")
    ap.add_argument("--n", type=int, default=8, help="Hutchinson probes for nu")
    a = ap.parse_args()
    dev = device()
    t0 = time.time()
    A = Parcae(a.model, a.src, a.tokenizer, dev) if a.family == "parcae" else Huginn(a.model, dev)
    K = a.K or (4 * A.r if a.family == "parcae" else 2 * A.r)
    xy = retokenized(a.data, A.encode, a.seqs, a.T).to(dev)
    x, y = xy[:, :-1].contiguous(), xy[:, 1:].contiguous()
    res = run(A, x, y, K, a.n, torch.Generator(device=dev).manual_seed(0))
    name = os.path.basename(a.model.rstrip("/"))
    res.update(family=a.family, model=name, load_report=A.load_report, secs=time.time() - t0)
    dst = os.path.join(a.out, name, "public_dynamics.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    json.dump(res, open(dst, "w"))
    print(name, "loss at r: ours %.5f, official %.5f" % (res["loss"][A.r - 1], res["official_loss_at_r"]), flush=True)


if __name__ == "__main__":
    main()
