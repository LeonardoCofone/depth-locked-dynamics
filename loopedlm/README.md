# loopedlm/

The looped language model, its data pipeline and its training loop. Everything else in the repository imports from
here.

| file | content |
|---|---|
| `model.py` | `LoopedLM`: a prelude of two blocks, a core of two blocks applied `k` times, and a coda of two blocks. At every loop the core receives the prelude output `e` through an injection with a diagonal decay, `s <- decay * s + dt * B e`. Blocks are pre-norm with RMSNorm, attention with RoPE and query-key norm, and a squared-ReLU MLP. `core_map(s, e)` is one loop; `step(s, e)` applies the update scale `g` of the configuration, `s + g (core_map(s, e) - s)`, which equals `core_map` when `g = 1`. |
| `config.py` | Default hyperparameters, the four presets (`base`, `muon`, `width768`, `large`), the run name built from the configuration, and `from_original`, which converts the configuration format of the original training code so that its checkpoints load with this model. |
| `data.py` | `python -m loopedlm.data`: streams FineWeb-Edu (`sample-10BT`), trains a byte-level BPE tokenizer of 32,768 tokens on the first 50,000 documents, writes these documents as validation tokens (`val.npy`, 20M) and the following documents as training tokens (`train.npy`, 3B), then `tokenizer.json` and `manifest.json`. Also `Batches`, which samples training sequences, and `fixed_batches`, which draws the fixed validation sequences of the evaluation during training and of every measurement. |
| `optim.py` | AdamW parameter groups and the Muon optimizer used by the `muon` preset. |
| `train.py` | `python -m loopedlm.train`: one run. Fixed depth (`--sampling fixed`) runs `mean_rec` loops and backpropagates through the last `mean_bptt`; random depth (`--sampling poisson`) draws the depth of each sequence from a Poisson of mean `mean_rec` and backpropagates through at most `mean_bptt` loops. `--loss_every_loop 1` averages the loss over loops 1 to `r`. Writes `run_info.json`, `train_log.jsonl`, `eval_log.jsonl` and `model.pt`, and resumes from `latest.pt`. |
| `checkpoint.py` | Loads a trained run for measurement. |

## Training a model

```
python -m loopedlm.train --preset base --sampling fixed --mean_rec 8 --mean_bptt 8 --seed 0
```

The run is written to `runs/fixed_r8_b8_s0/`. The name lists what differs from the base recipe: depth `r`, gradient
depth `b`, then `_d<width>`, `_muon`, `_dec<decay>`, `_all` (loss at every loop), `_us<scale>` (damped update),
`_dfz` (frozen decay), `_D<tokens>M` (a number of training tokens other than the preset's) and the seed.

On one GPU with bfloat16 a model of 114M parameters at 0.5B tokens takes between one and six hours; the time grows with
`r`. The `large` preset (472M parameters, 3B tokens) takes 30 to 45 hours on one H100.
