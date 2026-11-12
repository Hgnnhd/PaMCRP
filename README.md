# PaMCRP

**A Passive Multi-Cancer Risk Prediction Framework Using Longitudinal Disease Trajectories**

PaMCRP predicts multi-cancer risk from longitudinal disease trajectories. Our
framework combines hierarchical disease representations, temporal and age
information, and proxy-based learning to capture shared and cancer-specific
patterns for lung, breast, colorectal, prostate, and stomach cancer.

## Model

- **Hierarchical disease embeddings** integrate four levels of disease tokens.
- **Temporal and age conditioning** represents the timing of disease events.
- **Transformer encoding** captures relationships across the disease trajectory.
- **Proxy learning** guides category-aware patient representations.
- **Cumulative risk heads** predict risk for five cancers at 90, 180, 360, 1080,
  and 1800 days. A six-class auxiliary classifier distinguishes the reference
  group and the five cancer categories.

We jointly optimize masked binary cross entropy, category cross entropy, and
the proxy objective using the Multiple Gradient Descent Algorithm (MGDA).

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -r requirements.txt
```

## Training

Organize the input tensors as described in [DATA_FORMAT.md](DATA_FORMAT.md),
then run from the repository root:

```bash
python -m src.train_multicancer --data-dir /path/to/prepared_tensors --output-dir runs/pamcrp --device cpu
```

Use `--device cuda:0` for GPU training. The default configuration uses 50 epochs,
a batch size of 1024, a learning rate of 0.0005, and an early-stopping patience
of 5 epochs. Run `python -m src.train_multicancer --help` for all options.

We select the best checkpoint by the mean validation AUC across available
cancer/horizon pairs and evaluate it on the test split. Training produces
`best_model.pt` and `metrics.json` in the output directory.

## Code structure

```text
src/
  config.py                 Model configuration
  train_multicancer.py      Training entry point
  module/
    net.py                  PaMCRP network
    model.py                Transformer, cumulative risk heads, and MGDA
    utils.py                Orthogonality regularization
    data.py                 Tensor loading
    eval.py                 AUC evaluation
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
