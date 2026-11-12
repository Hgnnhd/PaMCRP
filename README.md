# PaMCRP

A Passive Multi-Cancer Risk Prediction Framework Using Longitudinal Disease Trajectories.

This public source release contains the core PaMCRP network and training code.
It accepts privately prepared tensors through a documented interface. In keeping
with the authors' requested disclosure scope for their UK Biobank (UKB) work,
participant data, example records, cohort construction, source-field mappings,
data cleaning, preprocessing scripts, derived datasets, and study outputs are
not distributed. This repository does not implement a raw-data-to-tensor pipeline.

## Contents

```text
src/
  config.py                 Model dimensions and prediction horizons
  train_multicancer.py      Training, early stopping, checkpoint selection
  module/
    net.py                  Hierarchical embeddings and PaMCRP network
    model.py                Transformer, cumulative risk heads, MGDA
    utils.py                Orthogonality regularizer used by the network
    data.py                 Prepared-tensor loading and interface validation
    eval.py                 Aggregate AUC needed for validation and testing
DATA_FORMAT.md              Private directory and tensor field conventions
requirements.txt
```

The model retains hierarchical token embeddings, temporal/age conditioning,
Transformer encoding, proxy learning, five cumulative risk heads, and a six-class
auxiliary classifier. Training combines masked binary cross entropy, category
cross entropy, and the proxy objective using MGDA.

## Run training

Use Python 3.10 or newer and install dependencies in a separate environment:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your shell.
python -m pip install -r requirements.txt
```

Prepare the three private tensor files described in [DATA_FORMAT.md](DATA_FORMAT.md).
Run from the repository root:

```bash
python -m src.train_multicancer --help
python -m src.train_multicancer --data-dir /private/prepared_tensors --output-dir /private/pamcrp_runs --device cpu
```

For a CUDA environment, use an appropriate PyTorch installation and set
`--device cuda:0`. Defaults: 50 epochs, batch size 1024, learning rate 0.0005,
patience 5, seed 2023. Smaller batches may be needed for the available memory.
The loader holds each split in memory.

The best checkpoint is selected using the unweighted mean of available validation
AUCs across cancer categories and prediction horizons. A category/horizon with
only one label class has an undefined AUC and is omitted from the mean; if all
validation AUCs are undefined, training stops with an explicit error. The test
split is evaluated once after restoring the best checkpoint.

Local outputs are `best_model.pt` and aggregate `metrics.json`. No participant
identifiers or individual predictions are exported. Keep all outputs in the
authorized environment; creating a checkpoint or aggregate result does not
establish that it is approved for public release. The included `.gitignore`
excludes common local data, checkpoints, and run directories.

## Public-release changes

- Removed preprocessing, example CSV files, data-derived figures, interpretation
  utilities, unused analysis helpers, and compiled Python caches.
- Replaced preprocessing-specific pickle filenames and patient-ID inputs with
  the prepared-tensor interface. This is an interface change; old datasets are
  not read directly.
- Preserved the core network and three training objectives. Cancer-head sample
  eligibility now uses six-class category labels; the old training entry point
  compared a binary event indicator against cancer category IDs.
- Restricted validation to the provided horizon labels and observation masks.
  The former evaluation inferred horizon labels from mask positions. The public
  version also omits C-index, bootstrap analysis, risk-group tables, and plots.
  It is not a reproduction of the original study's complete analysis pipeline.
- Added a zero-gradient normalization guard for MGDA and made positional
  embeddings follow the input device.

No UKB data or pretrained weights are provided, and no study performance is
claimed by this source release. Users must obtain their own authorized data and
prepare compatible tensors within their approved environment. The documentation
describes only the model input contract, not the original cohort or processing
procedures.

## License

The project's existing Apache License 2.0 is retained in [LICENSE](LICENSE).
