# figures/

| file | content |
|---|---|
| `make_figures.py` | Draws every figure of the paper from `results/` into `out/` and prints `OK` for each one whose panels, labels and legends fit inside the figure without overlapping. |
| `out/fig_profile.pdf` | Step before `r` for `r = 6..24`, raw and multiplied by `r` (Section 3). |
| `out/fig_brake.pdf` | Typical log-stretch `lambda_k` in `k/r` for `r = 6..24`, and with three times the data (Section 4). |
| `out/fig_wave.pdf` | Loss and distance from `s_r` past `r`, and the accumulated phase of the step (Section 5). |
| `out/fig_relax.pdf` | Loss with the update scaled by 1/2 against the trained loop (Section 6). |
| `out/fig_recipe.pdf` | Brake and loss past `r` by recipe at `r = 8`, and the public models (Section 7). |
| `out/fig_temporal.pdf` | Decoded loop index (appendix). |
| `out/fig_normality.pdf` | Departure from normality along the loop (appendix). |

```
python figures/make_figures.py
```
