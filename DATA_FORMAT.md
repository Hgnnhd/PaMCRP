# Prepared tensor interface

Supply tensors prepared independently in an authorized private environment.
This document describes the training interface only. It does not specify UKB
source fields, participant selection, clinical outcome derivation, cleaning,
vocabulary construction, or a data processing recipe. No records or example
datasets are included.

## Private directory

```text
/private/prepared_tensors/
  train.npz
  validation.npz
  test.npz
```

Each file must be a NumPy NPZ archive containing exactly the six numeric arrays
below. Object arrays and pickle content are not accepted. Arrays in each file
must use the same row order. `N` is the number of rows in that split, `L` is the
sequence length, and `T=5` is the number of prediction horizons.

| Array | Shape | Values and meaning |
| --- | --- | --- |
| `features` | `[N, 4, L]` | Integer token indices, 0 through 9998, for four aligned hierarchy channels. All channels address the same event positions. |
| `times` | `[N, 3, L]` | Finite numeric temporal inputs in days: event/reference differences, relative time to the final event, and age at the event, in that channel order. Values are supplied as model-ready inputs. |
| `labels` | `[N, T]` | Binary cumulative event targets at each prediction horizon for the row's outcome category; reference rows have zero targets. |
| `masks` | `[N, T]` | Binary observation flags: 1 includes this horizon target in loss/AUC, 0 excludes it. Excluded target entries must still be finite binary values. |
| `categories` | `[N]` | Integer category: 0 reference, 1 lung, 2 breast, 3 colorectal, 4 prostate, 5 stomach. Used by auxiliary classification and proxy learning. |
| `sex` | `[N]` | Integer model eligibility code: 0 male, 1 female. Breast risk-head loss/AUC uses code 1; prostate uses code 0. Other heads use both codes. |

The five target columns are ordered by `evaluation_times` in `src/config.py`:
90, 180, 360, 1080, and 1800 days. Use one consistent token-index convention and
sequence length across all splits. Category and sex coding must follow this
interface even if the source data uses different encodings.

The existing network accepts `1 <= L <= 241` and has no attention padding mask.
There is no reserved token ID that the model automatically ignores. This loader
does not pad, truncate, sample, normalize, or map events. Supply fixed-length,
aligned model-ready arrays under your private protocol; do not assume zero
padding is ignored. The batch loader only stacks rows and converts numeric types.

## Split and target responsibilities

Train, validation, and test participants must be disjoint. Keep split identifiers
and linkage keys private; they are not model inputs. Establish splits and any
fitted transformations privately without using validation/test outcomes to fit
training inputs. The public loader cannot verify participant disjointness
because participant identifiers are intentionally absent.

The caller is responsible for clinically valid labels, temporal alignment,
censoring masks, and category eligibility. The loader checks shapes, numeric
finiteness, token ranges, and binary/category ranges; it cannot establish clinical
validity or reproduce the study's outcome definitions. Validation AUC uses the
supplied labels wherever masks equal 1, with the same category/sex eligibility
as the risk loss. Auxiliary category and proxy objectives use all training rows,
independently of horizon masks.

Use a private output directory for trained weights and metrics. Do not place
raw records, processed arrays, participant identifiers, data dictionaries derived
from restricted data, or study artifacts into the public source distribution.
