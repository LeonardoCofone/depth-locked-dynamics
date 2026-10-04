import argparse
import hashlib
import json
import os

import numpy as np
import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(out, subset, tokenizer_docs, vocab, train_tokens, val_tokens):
    from datasets import load_dataset
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    os.makedirs(out, exist_ok=True)
    stream = iter(load_dataset("HuggingFaceFW/fineweb-edu", name=subset, split="train", streaming=True))
    docs = [next(stream)["text"] for _ in range(int(tokenizer_docs))]
    tok_path = os.path.join(out, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=int(vocab), special_tokens=["<|eot|>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(docs, trainer)
        tok.save(tok_path)
    tok = Tokenizer.from_file(tok_path)
    eot = tok.token_to_id("<|eot|>")

    def fill(path, n, source):
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(n,))
        state = {"count": 0, "docs": 0}

        def flush(buf):
            for enc in tok.encode_batch(buf):
                ids = enc.ids + [eot]
                k = min(len(ids), n - state["count"])
                mm[state["count"]:state["count"] + k] = ids[:k]
                state["count"] += k
                state["docs"] += 1
                if state["count"] >= n:
                    return True
            return False

        buf = []
        for text in source:
            buf.append(text)
            if len(buf) < 512:
                continue
            if flush(buf):
                break
            buf = []
        else:
            if buf:
                flush(buf)
        mm.flush()
        if state["count"] < n:
            raise RuntimeError("%s: %d of %d tokens, more documents needed" % (path, state["count"], n))
        return state["docs"]

    val_docs = fill(os.path.join(out, "val.npy"), int(val_tokens), iter(docs))
    train_docs = fill(os.path.join(out, "train.npy"), int(train_tokens), (d["text"] for d in stream))
    manifest = {"dataset": "HuggingFaceFW/fineweb-edu", "subset": subset, "tokenizer_docs": int(tokenizer_docs),
                "val_docs": val_docs, "train_docs": train_docs, "vocab": tok.get_vocab_size(), "eot_id": eot,
                "train_tokens": int(train_tokens), "val_tokens": int(val_tokens),
                "sha256": {f: sha256_file(os.path.join(out, f)) for f in ("tokenizer.json", "val.npy", "train.npy")}}
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)


class Batches:
    def __init__(self, arr, T, seed):
        self.arr, self.T = arr, T
        self.g = torch.Generator().manual_seed(seed)

    def next(self, b):
        i = torch.randint(0, len(self.arr) - self.T - 1, (b,), generator=self.g).tolist()
        x = torch.from_numpy(np.stack([self.arr[j:j + self.T + 1] for j in i]).astype(np.int64))
        return x[:, :-1], x[:, 1:]


def fixed_batches(arr, T, n, b, seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        i = torch.randint(0, len(arr) - T - 1, (b,), generator=g).tolist()
        x = torch.from_numpy(np.stack([arr[j:j + T + 1] for j in i]).astype(np.int64))
        out.append((x[:, :-1], x[:, 1:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--tokenizer_docs", type=float, default=50000)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--train_tokens", type=float, default=1.5e9)
    ap.add_argument("--val_tokens", type=float, default=2e7)
    a = ap.parse_args()
    prepare(a.out, a.subset, a.tokenizer_docs, a.vocab, a.train_tokens, a.val_tokens)


if __name__ == "__main__":
    main()
