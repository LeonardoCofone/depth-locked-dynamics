# tests/

| file | content |
|---|---|
| `phase_estimator.py` | Runs the phase estimator of `measure/common.py` on a synthetic damped spiral (300 tokens, 64 dimensions, one turn in 13 steps) and prints the phase after one turn against `2 pi`, the check quoted in Section 5. Seconds. |
| `smoke.py` | Trains two tiny models (fixed and random depth) on random tokens on the CPU, runs five measurement scripts on them and checks that every output file is written. About six minutes. |

```
python tests/smoke.py
python -m tests.phase_estimator
```

The smoke test checks that the pipeline runs end to end; it says nothing about the numbers. The numbers of the paper
are checked by running the scripts in `analysis/` on `results/`.
