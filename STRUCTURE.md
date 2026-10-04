# Structure

```
depth-locked-dynamics/
├── loopedlm/      the looped model, data preparation, optimizer and training loop
├── measure/       one script per measurement; each reads trained runs and writes JSON into results/
├── results/       the measurements of every model in the paper (no checkpoints)
│   ├── models/          one folder per trained model
│   ├── public/          the public looped models (Parcae, Huginn, Ouro)
│   └── field_alignment/ the alignment of the loop field across models
├── analysis/      the tables and the registered tests, recomputed from results/
├── figures/       the figures of the paper, drawn from results/
├── registry/      the 117 registered tests: threshold date, outcome, verdict, script
├── configs/       the training command of every model
├── assets/        the logo
└── tests/         an end-to-end smoke test on the CPU and the check of the phase estimator
```

## How the parts connect

```
loopedlm.data ──> data/
loopedlm.train ──> runs/<run>/model.pt
measure.* ──> results/models/<run>/*.json
analysis.*, figures/make_figures.py <── results/
registry/tests.csv ──> names the analysis script and the printed line of each test
```

Preparing the data needs internet access; training and measuring need a GPU. Everything from `results/` on runs on a
CPU in about a minute.

## Run names

A run is named `<sampling>_r<depth>_b<backprop>[_<variant>]_s<seed>`, for example `fixed_r8_b8_s0` (fixed depth 8,
gradient through all 8 loops, seed 0) or `poisson_r8_b4_s1` (random depth with mean 8, gradient through the last 4
loops, seed 1). The variants are `muon` (optimizer), `d768` (width), `dec0.7`, `dec0.9` and `dec0.95` (initial decay),
`all` (loss at every loop), `us0.1` and `us0.25` (update damped in training), `dfz` (decay frozen) and `D100M`, `D1500M`
(training tokens).
