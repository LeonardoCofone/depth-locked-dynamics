# configs/

The training command of every model in `results/models/`, one per line. Each command recreates the run name of its
results folder.

| file | models |
|---|---|
| `grid.txt` | The first grid (33 models): fixed depth `r = 2..16` with full or truncated backpropagation, random depth, Muon, initial decay 0.7 and 0.9, 1.5B tokens, width 768. |
| `controls.txt` | The controls trained after their thresholds were written (18 models): fixed depth at `r = 6, 10, 20, 24` and second seeds, random depth through every loop, truncation to 4, loss at every loop, random depth with decay 0.9. |
| `contraction.txt` | The contraction experiments: the 0.1B-token pilot and the recipes with the update damped by 0.1 and 0.25. |

To run one file:

```
while read -r cmd; do $cmd --data data --runs runs; done < configs/controls.txt
```
