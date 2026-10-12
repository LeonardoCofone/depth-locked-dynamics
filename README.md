<h1 align="center">Depth-Locked Dynamics in Looped Language Models</h1>

<p align="center">
  <img src="assets/logo.svg" alt="Depth-Locked Dynamics" width="200">
</p>

Code, measurements and the test registry for the paper *Depth-Locked Dynamics in Looped Language Models*.

A looped language model applies one block of layers many times. When the loss sits at one fixed depth `r` and the
gradient reaches every loop, the loop locks onto `r`:

- **before `r`** it follows one step profile at every depth (the step shrinks as `r` grows);
- **at `r`** it brakes (the Jacobian contracts most within one loop of `r`);
- **past `r`** its step keeps turning in a plane, and the loss rises and falls with a period of about two training
  depths.

Scaling every update by `r/r'` moves the landing to loop `r' = 1.5r` or `2r` without retraining. In the controls, none
of these signatures appears when the depth is random or when the loss sits at every loop. These recipes also differ in
contraction, which may mediate the effect.

## Reproduce the paper from the included results

No GPU and no checkpoints are needed. Every number, table, figure and registered test of the paper is computed from
`results/`:

```
pip install -r requirements.txt
python -m analysis.model_table            # the table of the 51 looped models
python -m analysis.recipe_table           # the table of recipes at r = 8
python -m analysis.tests_controls         # registered tests on the controls
python figures/make_figures.py            # the figures, in figures/out/
```

`registry/tests.csv` lists the 123 registered tests with the date their threshold was written, the outcome, the
verdict and the script that recomputes it. Run any script in `analysis/` to see its tests.

## Retrain and remeasure

Training and measurement need a GPU (0.5B tokens per model of 114M parameters, 3B for the 472M models).

```
python -m loopedlm.data --out data                          # FineWeb-Edu, BPE tokenizer of 32,768 tokens
python -m loopedlm.train --preset base --sampling fixed --mean_rec 8 --mean_bptt 8 --seed 0
python -m measure.curves --runs runs --data data            # then any other script in measure/
```

`configs/` lists the command of every model in the paper. Each measurement script writes one JSON file per model into
`results/models/<run>/`, the files that `analysis/` reads. A run already measured there is skipped: pass `--redo 1` to
replace the included measurements, or `--out` to write elsewhere.

## What is not included

Checkpoints (about 0.5 GB per model of 114M parameters) and the per-token arrays of most models are not in the
repository. Every checkpoint can be retrained with its command in `configs/`.

## Layout

See [STRUCTURE.md](STRUCTURE.md). Every folder has a `README.md` that describes each of its files.

## License

CC BY-NC-SA 4.0, see [LICENSE](LICENSE).

## Citation

```
@article{depthlocked2026,
  title  = {Depth-Locked Dynamics in Looped Language Models},
  author = {Anonymous},
  year   = {2026}
}
```
