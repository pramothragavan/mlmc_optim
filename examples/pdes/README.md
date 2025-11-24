# MLMC Optim Examples

This directory contains example configurations and training scripts for all datasets from the paper.

## Available Examples

### 1. Darcy Flow (`darcy_flow/`)
**Model:** FNO2d  
**Task:** Permeability to pressure field mapping  
**Resolutions:** 30, 60, 120, 241

### 2. Navier-Stokes (`navier_stokes/`)
**Model:** FNO3d  
**Task:** 2D+time turbulent flow prediction  
**Resolutions:** 16, 32, 64

### 3. ADR - Advection-Diffusion-Reaction (`adr/`)
**Model:** ParabolicCNN  
**Task:** Concentration field evolution  
**Resolutions:** 75, 150, 300

### 4. FlowPastCylinder (`flow_past_cylinder/`)
**Model:** MP_PDE (GNN)  
**Task:** Flow field prediction around cylinder  
**Resolutions:** Mesh levels 1-4

### 5. JEB - Jet Engine Bracket (`jeb/`)
**Model:** GINOT (Transformer)  
**Task:** Structural stress prediction on point clouds  
**Resolutions:** Point cloud densities 2-6

## Quick Start

Each dataset directory under `examples/pdes/` contains:

- `configs/` – YAML configs for baselines and MLMC sweeps
- `README.md` – dataset and model details
- `data/` – expected location of `.pt` files (see dataset README)

From the repo root, run an experiment via the generic MLMC training script:

```bash
# Darcy FNO baseline
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/darcy_flow/configs/darcy_fno_baseline_sweep.yaml

# FlowPastCylinder MP-PDE baseline
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/flow_past_cylinder/configs/fpc_mp_pde_baseline.yaml
```

## Model Summary

| Dataset | Model | Parameters | Resolution-Invariant |
|---------|-------|------------|---------------------|
| Darcy | FNO2d | 1.2M | ❌ |
| Darcy | ParabolicCNN | 1.8K | ✅ |
| ADR | ParabolicCNN | 2.4K | ✅ |
| Navier-Stokes | FNO3d | 3.3M | ❌ |
| FlowPastCylinder | MP_PDE | 695K | ✅ (mesh) |
| JEB | GINOT | 579K | ✅ (point cloud) |

## MLMC Benefits

All examples demonstrate MLMC's advantages:
- **2-4x speedup** over single-resolution training
- **Better sample efficiency** at low data budgets
- **Variance reduction** through hierarchical pairing
- **Automatic level balancing** via geometric progression

See `sweep_configs/` for Pareto curve generation configs.
