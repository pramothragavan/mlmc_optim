# Darcy Flow Example

2D Darcy flow problem solved with FNO using MLMC optimization.

## Data

1. Download the official Darcy `.mat` files from the original FNO/NeuralOperator source
   (e.g. `piececonst_r241_N1024_smooth1.mat` and `piececonst_r241_N1024_smooth2.mat`).

2. From the repo root, convert them to finest-resolution `.pt` files using the helper
   in `data_fno.py`:

   ```bash
   python -c "from examples.examples_src.data_classes.data_fno import convert_darcy_mat_to_pt; \
   convert_darcy_mat_to_pt( \
       train_mat_path='/absolute/path/to/piececonst_r241_N1024_smooth1.mat', \
       test_mat_path='/absolute/path/to/piececonst_r241_N1024_smooth2.mat', \
       output_dir='examples/pdes/darcy_flow/data' \
   )"
   ```

3. This will create finest-resolution files at 241×241. To generate additional
   coarser resolutions, you can optionally run:

   ```bash
   python -m mlmc_optim.preprocess_data \
     --input_dir examples/pdes/darcy_flow/data/darcy2d \
     --dataset_name darcy2d \
     --finest_res 241 \
     --c2f_resolutions 30 60 120 241 \
     --dimension 2d \
     --method avgpool
   ```

4. The data is then expected under `examples/pdes/darcy_flow/data/darcy2d/`:
   - Training data: `train_r{30,60,120,241}.pt`
   - Test data: `test_r{30,60,120,241}.pt`

## Training

**Important**: Run from the repository root, not from this directory.

Basic MLMC training for Darcy FNO:

```bash
cd /path/to/mlmc_optim

python examples/examples_src/train_mlmc.py \
  --config examples/pdes/darcy_flow/configs/darcy_fno_mlmc_sweep.yaml
```

Override config options on the command line, for example:

```bash
python examples/examples_src/train_mlmc.py \
  --config examples/pdes/darcy_flow/configs/darcy_fno_mlmc_sweep.yaml \
  --epochs 100 --lr 0.001 --device mps
```

## Configuration

See the YAML files under `examples/pdes/darcy_flow/configs/` for all options (model, MLMC schedule, training hyperparameters).
