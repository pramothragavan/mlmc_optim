# MLMC Optim

**Multi-Level Monte Carlo Optimization for Neural Operators**

`mlmc_optim` is a PyTorch-based codebase for training neural operators on multi-resolution PDE data using Multi-Level Monte Carlo (MLMC) techniques. It contains the production implementation used in the paper experiments.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Installation

Install mlmc_optim in editable mode and its dependencies from our pyproject.toml:

```bash
git clone https://github.com/JRowbottomGit/mlmc_optim.git
cd mlmc_optim
pip install -e .
```

## Quick Start

```python
import torch
from mlmc_optim import MLMCTrainer
from mlmc_optim.dataset import MLMCDataset as MultiResolutionDataset  # base class; see examples for concrete datasets

# Load multi-resolution data
datasets = {
    res: MultiResolutionDataset(f"data/train_r{res}.pt", resolution=res)
    for res in [30, 60, 120, 241]
}

# Initialize your model
model = YourNeuralOperator()

# Create MLMC trainer
trainer = MLMCTrainer(
    model=model,
    datasets=datasets,
    resolutions=[30, 60, 120, 241],
    device="cuda"
)

# Train (see examples for full MLMC training/evaluation loops)
trainer.fit(epochs=500)
```

## Key Features

- **MLMC Optimizers**: Variance-reduced gradient estimation across resolution levels
- **Efficient Data Loading**: Automatic batching and caching for multi-resolution data
- **PyTorch Integration**: Drop-in replacement for standard training loops
- **Flexible Architecture**: Works with any neural operator (FNO, DeepONet, etc.)

## Usage

### Basic Training Loop

```python
from mlmc_optim import MLMCTrainer
from mlmc_optim.dataset import MLMCDataset as MultiResolutionDataset

# Setup datasets
train_data = {
    30: MultiResolutionDataset("data/train_r30.pt", resolution=30),
    60: MultiResolutionDataset("data/train_r60.pt", resolution=60),
    120: MultiResolutionDataset("data/train_r120.pt", resolution=120),
}

# Initialize trainer
trainer = MLMCTrainer(
    model=model,
    datasets=train_data,
    resolutions=[30, 60, 120],
    lr=1e-3,
    device="cuda"
)

# Train
trainer.fit(epochs=100)
```

### Multi-Resolution Data Loading

```python
from mlmc_optim.dataset import MLMCDataset as MultiResolutionDataset
from mlmc_optim.batcher import MLMCBatcher

# Load dataset
dataset = MultiResolutionDataset(
    data_path="data/darcy_train.pt",
    resolution=120,
    normalize=True
)

# Create MLMC batcher
batcher = MLMCBatcher(
    datasets={30: ds30, 60: ds60, 120: ds120},
    sample_sizes=[32, 16, 8],
    batch_sizes=[160, 80, 40]
)

# Get batches
for batch in batcher:
    # batch contains paired coarse/fine resolution data
    fine_data, coarse_data = batch
```

## Repository layout

- `mlmc_optim/` – core training logic (`MLMCTrainer`, dataset bases, MLMC batching).
- `examples/pdes/` – paper experiments (Darcy, ADR, Navier–Stokes, FlowPastCylinder, JEB).
- `examples/examples_src/` – shared models, dataset wrappers and the generic MLMC training script.
- `tests/` – sanity tests for the paper experiments.

## Running experiments

From the repo root:

```bash
# Darcy FNO MLMC sweep
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/darcy_flow/configs/darcy_fno_mlmc_sweep.yaml

# FlowPastCylinder MP-PDE baseline
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/flow_past_cylinder/configs/fpc_mp_pde_baseline.yaml
```

To run the short paper-experiment harness:

```bash
python tests/test_paper_experiments.py
```

## Data

This repo assumes preprocessed `.pt` datasets are available under `examples/pdes/*/data`, but you obtain them by downloading **raw** data and running the provided preprocessing utilities. Briefly:

- **Darcy (darcy2d)**: download the raw Darcy data (e.g. `Darcy_241.zip`) from the Google Drive folder:

  https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-

  Unpack it locally, then run your preprocessing to generate multi-resolution `.pt` files for the resolutions you need. The training code expects the final files under:

  ```text
  examples/pdes/darcy_flow/data/darcy2d/
  ```

- **Navier–Stokes (ns2d_time)**: download the raw Navier–Stokes data (e.g. `NavierStokes_V1e-3_N5000_T50.zip`) from the same Google Drive folder:

  https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-

  Then follow the instructions in `examples/pdes/navier_stokes/README.md` to run the preprocessing script, which creates the required multi-resolution `.pt` files under:

  ```text
  examples/pdes/navier_stokes/data/ns2d_time/
  ```

- **ADR**: download `adr.zip` from Zenodo `https://doi.org/10.5281/zenodo.17694930` and unpack from the repo root:

  ```bash
  cd /path/to/mlmc_optim
  unzip adr.zip -d examples/pdes
  ```

  You should then have `.pt` files under `examples/pdes/adr/data/adr/`.

- **FlowPastCylinder (FPC)**: download `FlowPastCylinder.zip` from the same Zenodo record `https://doi.org/10.5281/zenodo.17694930` and unpack:

  ```bash
  cd /path/to/mlmc_optim
  unzip FlowPastCylinder.zip -d examples/pdes
  ```

  You should then have the graph datasets under `examples/pdes/flow_past_cylinder/data/FlowPastCylinder/`.

- **JEB (Jet Engine Bracket)**: see `examples/pdes/jeb/README.md` for instructions and expected data structure for point-cloud experiments.

## API Reference

### MLMCTrainer

Main training interface compatible with PyTorch Lightning style.

```python
trainer = MLMCTrainer(
    model,                    # Neural operator model
    datasets,                 # Dict[int, Dataset] - resolution -> dataset
    resolutions,              # List[int] - resolutions to use
    lr=1e-3,                  # Learning rate
    device="cuda",            # Device
    epochs=500,               # Number of epochs
    eval_every=10,            # Evaluation frequency
)
```


### MultiResolutionDataset

PyTorch Dataset for multi-resolution PDE data.

```python
dataset = MultiResolutionDataset(
    data_path,               # Path to .pt file
    resolution,              # Grid resolution
    train=True,              # Train or test split
    normalize=True,          # Normalize data
)
```

### MLMCBatcher

Handles batching logic for MLMC training.

```python
batcher = MLMCBatcher(
    datasets,                # Dict of datasets per resolution
    sample_sizes,            # Samples per resolution level
    batch_sizes,             # Batch size per level
    pairing="hierarchy",     # Pairing strategy
)
```

## Citation

```bibtex
@article{rowbottom2025mlmc,
  title   = {Multi-Level Monte Carlo Training of Neural Operators},
  author  = {Rowbottom, James and Fresca, Stefania and Lio, Pietro and Sch\"{o}nlieb, Carola-Bibiane and Boull\'{e}, Nicolas},
  journal = {arXiv preprint arXiv:2505.12940},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.12940}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
