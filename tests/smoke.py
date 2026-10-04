import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SMALL = ["--d", "64", "--heads", "2", "--ffn", "128", "--T", "1024", "--batch", "4", "--micro", "4",
         "--tokens", "16384", "--eval_every", "100", "--eval_batches", "1", "--log_every", "1"]


def run(args):
    subprocess.run([sys.executable] + args, cwd=ROOT, check=True)


def main():
    work = tempfile.mkdtemp()
    data, runs, out = (os.path.join(work, x) for x in ("data", "runs", "results"))
    os.makedirs(data)
    rng = np.random.default_rng(0)
    np.save(os.path.join(data, "train.npy"), rng.integers(0, 256, 200000).astype(np.uint16))
    np.save(os.path.join(data, "val.npy"), rng.integers(0, 256, 120000).astype(np.uint16))
    json.dump({"vocab": 256}, open(os.path.join(data, "manifest.json"), "w"))
    for sampling, b in (("fixed", "4"), ("poisson", "8")):
        run(["-m", "loopedlm.train", "--sampling", sampling, "--mean_rec", "4", "--mean_bptt", b, "--data", data,
             "--runs", runs] + SMALL)
    for script in ("curves", "lyapunov", "step_geometry", "state_distance", "scaled_update"):
        run(["-m", "measure." + script, "--runs", runs, "--data", data, "--out", out])
    for name in sorted(os.listdir(out)):
        files = sorted(os.listdir(os.path.join(out, name)))
        print(name, files)
        assert len(files) == 6, files
    print("smoke test passed")


if __name__ == "__main__":
    main()
