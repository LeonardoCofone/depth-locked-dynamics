# registry/

`tests.csv` lists the 127 registered tests of the paper. A test is registered when its threshold was written in a
dated log before the data it concerns were examined; thresholds were never changed afterwards. A verdict is `grey` when
the result falls between the thresholds for yes and no.

| column | content |
|---|---|
| `id` | Order in which the tests appear in the appendix tables. |
| `table` | Appendix table of the paper (6, 7 or 8). |
| `registered` | Date the threshold was written. |
| `test` | The test and its threshold. |
| `outcome` | What the data gave. |
| `verdict` | `YES`, `NO` or `grey`. |
| `script` | The script in `analysis/` that recomputes the outcome from `results/`. |
| `output` | The beginning of the line (or lines, separated by `\|`) that the script prints for this test. |

Totals: 67 YES, 42 NO, 18 grey.

For 98 tests the script prints the verdict next to the outcome; for the other 29 it prints the outcome, and the
verdict is the one in the `verdict` column.

Two thresholds are applied with a detail fixed when they were first evaluated: the restart of the decoded loop index
compares the drop rounded to two decimals with 0.25 (`analysis/tests_decoder.py`), and the minimum of non-normality in
Ouro must be interior to loops 2 to 8 (`analysis/tests_public.py`).
