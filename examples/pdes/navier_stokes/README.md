# Navier-Stokes Example

3D Navier-Stokes turbulence problem solved with FNO3d using MLMC optimization.

## Data

1. Download the raw Navier–Stokes dataset (e.g. `ns_V1e-3_N5000_T50.mat`) from the
   official FNO / NeuralOperator Google Drive.

2. From the repo root, convert it into finest-resolution `.pt` files using the
   helper in `data_fno.py`:

   ```bash
   python -c "from examples.examples_src.data_classes.data_fno import convert_ns_mat_to_pt; \
   convert_ns_mat_to_pt( \
       ns_mat_path='/absolute/path/to/ns_V1e-3_N5000_T50.mat', \
       output_dir='examples/pdes/navier_stokes/data/ns2d_time' \
   )"
   ```

3. Then run the MLMC preprocessing utility to build the multi-resolution `.pt` files
   used by this example:

   ```bash
   python -m mlmc_optim.preprocess_data \
     --input_dir examples/pdes/navier_stokes/data/ns2d_time \
     --dataset_name ns \
     --finest_res 64 \
     --c2f_resolutions 16 32 64 \
     --dimension 2d \
     --method avgpool \
     --max_samples 3000
   ```

The expected training and test files will live under:

```text
examples/pdes/navier_stokes/data/ns2d_time/
```

## Training

From the repository root:

```bash
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/navier_stokes/configs/ns_fno3d_baseline_sweep.yaml
```

## Configuration

See the YAML configs under `examples/pdes/navier_stokes/configs/` for exact hyperparameters.
