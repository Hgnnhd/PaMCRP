# Data format

## Directory structure

```text
prepared_tensors/
  train.npz
  validation.npz
  test.npz
```

Each NumPy NPZ file contains the six numeric arrays below, aligned by row.
`N` is the split size, `L` is the sequence length, and `T=5` is the number of
prediction horizons. Use numeric arrays rather than object arrays.

| Array | Shape | Convention |
| --- | --- | --- |
| `features` | `[N, 4, L]` | Integer token indices from 0 to 9998 across four aligned disease hierarchy channels. |
| `times` | `[N, 3, L]` | Temporal inputs in days: event/reference differences, relative time to the final event, and age at the event, in that order. |
| `labels` | `[N, T]` | Binary cumulative event targets for the row's outcome category; reference rows have zero targets. |
| `masks` | `[N, T]` | Binary observation flags: 1 includes a target in the risk loss and AUC; 0 excludes it. |
| `categories` | `[N]` | Integer category: 0 reference, 1 lung, 2 breast, 3 colorectal, 4 prostate, 5 stomach. |
| `sex` | `[N]` | Integer code: 0 male, 1 female. Breast and prostate risk heads use codes 1 and 0, respectively. |

## Tensor conventions

- Order the target columns by 90, 180, 360, 1080, and 1800 days, matching
  `evaluation_times` in `src/config.py`.
- Use a consistent token vocabulary and fixed sequence length across splits,
  with `1 <= L <= 241`. The model attends to every sequence position, including
  zero-valued tokens; it does not apply an attention padding mask.
- Supply finite numeric values in every array. Labels and masks are binary,
  including entries at masked positions.
- Keep participants disjoint across training, validation, and test splits.

Category and proxy objectives use all training rows. Risk loss and AUC use
observed horizon targets from the corresponding cancer category and reference
group, with the sex eligibility shown above. AUC is reported for pairs containing
both label classes; the mean uses the available pairs.
