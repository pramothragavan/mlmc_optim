# ADR (Advection-Diffusion-Reaction) Example

Advection-diffusion-reaction equation solving using ParabolicCNN.

## Dataset

This example uses 2D ADR simulations at multiple resolutions.

**Data structure:**
- Resolution levels: 75×75, 150×150, 300×300
- Input: Initial concentration field
- Output: Final concentration field
- Includes advection, diffusion, and reaction terms

## Model

**ParabolicCNN** - Resolution-invariant parabolic convolution
- Operator type: Diagonal (anisotropic diffusion)
- Advection: Enabled (learnable advection field)
- Hidden dim: 32
- Layers: 2 parabolic convolution layers
- Parameters: ~2,400 (resolution-invariant!)

## Training

```bash
python examples/adr/train.py
```

## Data

Download `adr.zip` from Zenodo:

- DOI: `https://doi.org/10.5281/zenodo.17694930`

From the repo root:

```bash
cd /path/to/mlmc_optim
unzip adr.zip -d examples/pdes
```

You should then have the ADR `.pt` files under:

```text
examples/pdes/adr/data/adr/
```

## Expected Results

- Test Loss: ~0.01-0.03 (MSE)
- Training time: ~1-2 hours on GPU
- MLMC speedup: 2-3x over baseline
- Resolution-invariant performance
