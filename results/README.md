# results/

The measurements behind the paper, as written by `measure/`. Nothing here is edited by hand.

## models/

One folder per trained model, named as in `loopedlm/config.py` (for example `fixed_r8_b8_s0`, `poisson_r8_b4_muon_s1`,
`fixed_r16_b16_D1500M_s0`). 51 models make up the model table of the paper; the others are the contraction
experiments: the pilot at 0.1B tokens (`_D100M`) and the recipes with the update damped in training (`_us`).

| file | written by | content |
|---|---|---|
| `run_info.json` | `loopedlm.train` | Resolved configuration, number of steps and parameters. |
| `train_log.jsonl`, `eval_log.jsonl` | `loopedlm.train` | Every 20 steps: training loss, learning rate, gradient norm, time, mean depth and the range of the decay. Every 500 steps: validation loss at depths 1 to 32. |
| `curves.json` | `measure.curves` | `per_k[k]`: loss and step statistics after `k` loops; `step_norm_median[k]`: median step over the tokens of the first 8 sequences. |
| `curves_tokens.npz` | `measure.curves` | Loss and step of every token (8 sequences); kept only for the 15 models of the per-token tests. |
| `lyapunov.json` | `measure.lyapunov` | `lyap[k][i]`: log-stretch of direction `i` at loop `k + 1`. |
| `step_geometry.json` | `measure.step_geometry` | `G`: Gram matrix of the steps; `pre`, `post`: mode energy and accumulated phase before and after `r`; `loss_coda`. |
| `state_distance.json` | `measure.state_distance` | `D[i][j]`: mean distance between the states after `i + 1` and `j + 1` loops, divided by the norm at `r`. |
| `loop_decoder.json`, `loop_decoder_k4.json` | `measure.loop_decoder` | `r2`, decoded loop index `khat_mean` per loop, relative step per loop. |
| `temporal_edit.json` | `measure.temporal_edit` | Loss after edits of the state at loop `r/2`. |
| `non_normality.json`, `non_normality_coarse.json` | `measure.non_normality` | `nu[k]` and the per-probe terms used for bootstrap intervals. |
| `plane_dynamics.json`, `plane_dynamics_tokens.npz` | `measure.plane_dynamics` | `A`: trajectory to `16r`; `B`: Jacobian in the plane at selected loops (per-token 2x2 matrices in the npz); `C`: interventions; `D`: decay and attention heads. |
| `scaled_update.json` | `measure.scaled_update` | Loss of `r' = r, 2r, 4r, 8r` loops with the update scaled by `r/r'`, distances between their states, Richardson extrapolation, split of the step into injection and core. |
| `doubled_update.json` | `measure.doubled_update` | Loss when the update is doubled after `k` trained loops. |
| `downstream.json` | `measure.downstream` | Accuracy per task and loop, trained and with the update scaled by 1/2. |
| `robustness.json` | `measure.robustness` | Log-stretch with 16 and 32 directions and step geometry on new sequences. |

Indices: in `curves.json`, `per_k[k]` is the state after `k` loops (`k = 0` is the initial state). In the other files
lists usually start at loop 1, so entry `k - 1` is loop `k`.

## public/

`Ouro-1.4B`, `Ouro-2.6B` (`ouro_dynamics.json`, `ouro_scaled.json`), `parcae-140m`, `parcae-370m`, `parcae-770m`,
`parcae-1.3b` and `huginn-0125` (`public_dynamics.json`), measured on the same validation text retokenised for each
model.

## field_alignment/

`field_alignment_base.json` (depths 8, 12, 16) and `field_alignment_grid.json` (depths 6 to 24, two seeds): the
alignment of the rescaled steps of two models, from `measure.field_alignment`.
