# analysis/

Scripts that read `results/` and print the numbers of the paper: its two tables and every registered test, with the
threshold applied as it was written. They run on a CPU in seconds.

```
python -m analysis.tests_wave
```

Each test is applied to the models that existed when its threshold was written: the first grid (33 models), the
control grid trained afterwards (18 models, marked with a dagger in the paper), the contraction experiments or the
three models of 472M parameters. The sets are defined in `common.py`.

| file | prints | paper |
|---|---|---|
| `common.py` | | Loaders, model sets (`first_grid`, `CONTROL_GRID`, `CONTRACTION`, `SCALE`, `WAVE15`), and the shared definitions: contraction rate `q`, brake (lowest `lambda_k` within one loop of `r`, relative brake), loss minima past `r`, rise from `r` to `2r`, period from the phase `P_Theta`, rotation angle of a 2x2 Jacobian. |
| `model_table.py` | LaTeX rows of the table of the 51 looped models | Table 3 (Appendix A) |
| `recipe_table.py` | LaTeX rows of the table of recipes at `r = 8` (`--all`: every recipe and depth) | Table 2 (Section 7) |
| `tests_wave.py` | Oscillation, amplitude, period and contraction, collapse in `(k - r)/r`, distances between states | Table 6 |
| `tests_decoder.py` | Loop decoder, its restarts, edits of the decoded loop index | Table 6 |
| `tests_landing.py` | Brake on held-out models, path length, slow phase and final arc, step growth, non-normality, truncated models, log-stretch after `r`, angular velocity at landing | Tables 6 and 7 |
| `tests_geometry.py` | Planar steps, turns per period, damping of the state, quadratic landing, tests per token | Table 6 |
| `tests_public.py` | Parcae, Huginn and Ouro | Table 6 |
| `tests_scaling.py` | Collapse of step, angle and log-stretch before `r`; sum of the log-stretch; period from the phase | Tables 6 and 7 |
| `tests_plane_dynamics.py` | Period from 8 sequences, rotation at the fixed point and at landing, paths that make the rotation, rescaled update | Table 7 |
| `tests_euler.py` | Decay scale, exact phase of the rescaled update, stability of the doubled step, continuous limit, Richardson, cancellation | Table 7 |
| `tests_field.py` | Same field across depths (8, 12, 16 and the grid 6 to 24) | Tables 7 and 8 |
| `tests_scaled_update.py` | Update doubled after the first loops, downstream accuracy loop by loop, states of the scaled loop | Table 7 |
| `tests_controls.py` | Recipes at `r = 8`, new fixed models, exponents and collapses at `r = 6..24`, rotation at landing, Ouro with the scaled update | Table 8 |
| `tests_robustness_contraction.py` | New sequences and 32 directions, the contraction pilot, the damped recipes | Table 8 |
| `tests_scale.py` | Two fixed models and one Poisson model of 472M parameters at `r = 8`: loss at `r`, brake, oscillation, rise from `r` to `2r`, planar steps | Table 8 (Section 10) |

`registry/tests.csv` maps each registered test to the script that prints it.
