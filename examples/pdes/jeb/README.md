# JEB (Jet Engine Bracket) Example

Structural stress prediction on jet engine bracket using GINOT transformer on point clouds.

## Dataset

This example uses point cloud data at multiple resolutions representing a jet engine bracket under load.

**Data structure:**
- Point clouds with varying densities (resolutions 0(fine) to 6(coarse))
- Input: 3D coordinates + boundary conditions
- Output: Stress field at each point
- Irregular geometry with complex loading

## Model

**GINOT** - Geometry-Informed Neural Operator Transformer
- Branch encoder: Point cloud perceiver (4 layers, 4 heads)
- Trunk decoder: Cross-attention transformer
- Embedding dim: 64
- Handles variable-size point clouds
- Coordinate embedding: NeRF-style positional encoding

## Training

```bash
python examples/jeb/train.py
```

## Data Download

The raw JEB data comes from the public datasets used in the original GiN-OT paper.

1. Download the datasets from Zenodo (see GiN-OT repo README):
   - Zenodo record containing `GEJetEngineBracket.zip` and related files
   - URL: see https://github.com/QibangLiu/GINOT (section *Dataset*)

2. From the Zenodo page, download at least:
   - `GEJetEngineBracket.zip` (jet engine bracket volume/boundary/stress data)

3. Unzip into the JEB data directory for this repo, so you get a structure like:

   ```text
   examples/pdes/jeb/data/
       GEJetEngineBracket/
           VolumeMesh/
           FieldMesh/
           BRep/
           ...
   ```

4. Run the preprocessing script in this repo to generate the processed
   `GE-JEB.pkl` / multi-resolution point-cloud data used by the examples. A
   simple entry point is:

   ```
   python examples/examples_src/data_classes/data_jeb.py
   ```

   which will create multi-resolution JEB datasets under the configured
   `data_dir` using `create_multi_resolution_jeb_data`.

> Note: The original GE Jet Engine Bracket meshes and nodal fields are
> provided under the GE-JEB license used by the GiN-OT project
> (see https://github.com/QibangLiu/GINOT and the license text at
> https://drive.google.com/file/d/1S4_DphSGNcIzGvFMOjVRJvyhcC_5-nJG/view).
> The 4-node tetrahedral meshes and point-cloud datasets used here are a
> *derivative database* of that dataset and remain governed by that license.

## Expected Results

- Test Loss: ~0.05-0.10 (GINOT loss)
- Training time: ~4-6 hours on GPU
- MLMC speedup: 2-4x over baseline
- Handles irregular geometries well
