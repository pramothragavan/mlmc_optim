# FlowPastCylinder Example

Graph-based FlowPastCylinder experiment using the MP_PDE GNN model.

## Data

Download `FlowPastCylinder.zip` from Zenodo:

- DOI: `https://doi.org/10.5281/zenodo.17694930`

From the repo root:

```bash
cd /path/to/mlmc_optim
unzip FlowPastCylinder.zip -d examples/pdes
```

You should then have the graph datasets under:

```text
examples/pdes/flow_past_cylinder/data/FlowPastCylinder/
examples/pdes/flow_past_cylinder/data/FlowPastCylinder_direct/
```

## Training

See the generic training script and configs, for example:

```bash
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/flow_past_cylinder/configs/fpc_mp_pde_baseline.yaml
```
