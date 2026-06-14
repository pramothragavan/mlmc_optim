import os
import scipy.io
import h5py
import numpy as np
from typing import Tuple
from tqdm import tqdm
import torch

from mlmc_optim.preprocess_data import upscale_2d
from examples.examples_src.data_classes.data_fno import MultiResolutionDataset, MultiResolutionDataset3D
from examples.examples_src.data_classes.data_pyg import NavierStokesDataset
from examples.examples_src.data_classes.data_jeb import JEBDataset


def encode(x, m, s):
    return (x - m) / (s + 0.00001)

def decode(x, m, s):
    return x * (s + 0.00001) + m


def compute_cfl_osc(data: np.ndarray, dt: float, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generic CFL/oscillation metrics.

    Args:
        data: array with shape [T, H, W] (scalar) or [T, C, H, W] (vector)
        dt: time step
        dx: spatial grid spacing (assumes square grid)
    """
    if data.ndim == 3:
        # [T, H, W] scalar
        T, H, W = data.shape
        C = 1
        data_c = data.reshape(T, 1, H, W)
    elif data.ndim == 4:
        # [T, C, H, W]
        T, C, H, W = data.shape
        data_c = data
    else:
        raise ValueError(f"Unsupported data shape for CFL/osc: {data.shape}")

    cfl_max_history = []
    osc_history = []
    for t in range(T):
        fields = data_c[t]  # [C, H, W]
        if C == 1:
            vel_mag = np.abs(fields[0])
        else:
            vel_mag = np.sqrt(np.sum(fields ** 2, axis=0))
        cfl_field = dt * vel_mag / dx
        cfl_max_history.append(cfl_field.max())

        if H > 2 and W > 2:
            osc_terms = []
            for c in range(C):
                u = fields[c]
                d2x = np.diff(u, n=2, axis=1)
                d2y = np.diff(u, n=2, axis=0)
                osc_terms.append(np.std(d2x))
                osc_terms.append(np.std(d2y))
            osc_metric = float(np.mean(osc_terms))
        else:
            osc_metric = 0.0
        osc_history.append(osc_metric)

    return np.array(cfl_max_history), np.array(osc_history)


def compute_grad_regularity(data: np.ndarray, dx: float) -> Tuple[float, float]:
    """Compute basic spatial regularity via gradient norms.

    Args:
        data: array with shape [..., H, W]; leading dims are batch/time.
        dx: spatial grid spacing (assumes square grid).

    Returns:
        mean_grad_l2: sqrt of mean squared gradient magnitude over all dims
        max_grad: L_inf of gradient magnitude over all dims
    """
    if data.ndim < 2:
        raise ValueError(f"Data must have at least 2 dims for spatial grid, got {data.shape}")

    # First-order differences in x (last axis) and y (second-to-last axis)
    grad_x = np.diff(data, axis=-1) / dx              # [..., H, W-1]
    grad_y = np.diff(data, axis=-2) / dx              # [..., H-1, W]

    # Align shapes to common interior region [..., H-1, W-1]
    # Drop last row from grad_x and last column from grad_y
    gx_int = grad_x[..., :-1, :]                      # [..., H-1, W-1]
    gy_int = grad_y[..., :, :-1]                      # [..., H-1, W-1]
    grad_mag = np.sqrt(gx_int ** 2 + gy_int ** 2)

    mean_grad_l2 = float(np.sqrt((grad_mag ** 2).mean()))
    max_grad = float(grad_mag.max())
    return mean_grad_l2, max_grad


def norm_darcy_dataset(dep_dataset, ctrl_dataset, use_grads=True):
    dep_dataset.input_data = encode(dep_dataset.input_data, ctrl_dataset.input_mean, ctrl_dataset.input_std)
    if use_grads:
        dep_dataset.input_smooth = encode(dep_dataset.input_smooth, ctrl_dataset.smooth_mean, ctrl_dataset.smooth_std)
        dep_dataset.input_gradx  = encode(dep_dataset.input_gradx,  ctrl_dataset.gradx_mean,  ctrl_dataset.gradx_std)
        dep_dataset.input_grady  = encode(dep_dataset.input_grady,  ctrl_dataset.grady_mean,  ctrl_dataset.grady_std)
    dep_dataset.output_data = encode(dep_dataset.output_data, ctrl_dataset.output_mean, ctrl_dataset.output_std)

def norm_dataset(dep_dataset, ctrl_dataset):
    dep_dataset.input_data  = encode(dep_dataset.input_data,  ctrl_dataset.input_mean,  ctrl_dataset.input_std)
    dep_dataset.output_data = encode(dep_dataset.output_data, ctrl_dataset.output_mean, ctrl_dataset.output_std)

class MatReader:
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            print(f"Scipy loading {self.file_path}")
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            print(f"Failing scipy loading {self.file_path}")
            print(f"H5PY loading {self.file_path}")
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

class GaussianNormalizer:
    """Normalize a Tensor to have zero mean and unit variance"""
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x, sample_idx=None):
        return x * (self.std + self.eps) + self.mean


def print_dataset_info(train_dataset, test_dataset, config):
    """Print detailed information about datasets and resolutions."""

    print("\nDataset Statistics:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Check if data is loaded in memory
    load_in_memory = config.get('load_in_memory', True)
    
    if config['dataset'] not in ['FlowPastCylinder', 'jeb']:
        if load_in_memory:
            # Only print shapes and ranges if data is in memory
            input_data = train_dataset.input_data if config['dataset'] != 'navier_stokes' else train_dataset.a
            output_data = train_dataset.output_data if config['dataset'] != 'navier_stokes' else train_dataset.u
            test_input = test_dataset.input_data if config['dataset'] != 'navier_stokes' else test_dataset.a
            test_output = test_dataset.output_data if config['dataset'] != 'navier_stokes' else test_dataset.u
            
            if input_data is not None:
                print(f"Input data shape: {input_data.shape}")
                print(f"Output data shape: {output_data.shape}")
                print("\nTest Dataset Details:")
                print(f"Input data shape: {test_input.shape}")
                print(f"Output data shape: {test_output.shape}")
                print(f"Input data range: [{test_input.min()}, {test_input.max()}]")
                print(f"Output data range: [{test_output.min()}, {test_output.max()}]")
        else:
            print("Data loaded on-demand (not in memory) - shape info not available")
        
        if hasattr(test_dataset, 'offset'):
            print(f"Test dataset offset: {test_dataset.offset}")

    elif config['dataset'] == 'FlowPastCylinder':
        print(f"FPC Train Data {train_dataset.data}")
        print(f"FPC Test Data {test_dataset.data}")

    elif config['dataset'] == 'jeb':
        print(f"JEB Train Data {train_dataset}")
        print(f"JEB Test Data {test_dataset}")

    if hasattr(train_dataset, 'has_gradients') and train_dataset.has_gradients:
        print("Dataset includes gradient information")


def get_datasets(config, c2f_resolutions, device):
    train_datasets = {}
    test_datasets = {}
    dataset_specific = {}

    # Load train datasets for each resolution
    temp_c2f_resolutions = c2f_resolutions + [config['base_res']]

    for res in tqdm(temp_c2f_resolutions, desc="Loading datasets"):
        add_coords = config.get('add_coords', False)
        use_grads = config.get('use_grads', False)

        if config['dataset'] == 'navier_stokes':
            train_datasets[res] = MultiResolutionDataset3D(
                config,
                data_dir=config['data_dir'],
                resolution=res,
                train=True,
                load_in_memory=config['load_in_memory'],
                normalize=config['normalize'],
                add_coords=add_coords,
            )
        elif config['dataset'] in ['darcy', 'adr']:
            train_datasets[res] = MultiResolutionDataset(
                config,
                config['data_dir'],
                res,
                train=True,
                load_in_memory=config['load_in_memory'],
                normalize=config['normalize'],
                add_coords=add_coords,
                use_grads=use_grads
            )
        elif config['dataset'] == 'jeb':
            level_dir = os.path.join(config['data_dir'], "GEJetEngineBracket", f"level_{res}")
            # Create datasets for this resolution level
            train_datasets[res] = JEBDataset(
                config=config,
                root=os.path.join(level_dir, "train"),
                data_dir=config['data_dir'],
                resolution=res,
                train=True,
                load_in_memory=config.get('load_in_memory', True),
                normalize=config.get('normalize', True)
            )

            # Special handling for JEB dataset - we don't need to normalize per level
            # since normalization is done during dataset preparation
            if res == min(c2f_resolutions) and 'inverse_transforms' not in dataset_specific:
                dataset_specific['inverse_transforms'] = {
                    's_inverse': train_datasets[res].s_inverse,
                    'pc_inverse': train_datasets[res].pc_inverse,
                    'vert_inverse': train_datasets[res].vert_inverse
                }

        elif config['dataset'] == 'FlowPastCylinder':
            source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "navier_stokes_data")

            # Original slices for FlowPastCylinder
            # train_slice = slice(0, 4)
            train_slice = slice(0, config['total_samples'])
            N_train = train_slice.stop - train_slice.start
            data_split_train = train_slice

            # Add the _direct suffix to the dataset name when direct loading is enabled
            direct_suffix = "_direct" if config.get('FPC_direct', False) else ""
            dataset_name_with_suffix = f"{config['dataset']}{direct_suffix}"
            train_data_res = os.path.join(config['data_dir'], dataset_name_with_suffix,
                                          f"level_{res}_N{N_train}_train")
            print(f"Loading train data {train_data_res}")

            train_datasets[res] = NavierStokesDataset(
                config=config,
                root=train_data_res,
                data_dir=source_dir,  # Load data from mesh-specific directory if direct loading
                mesh_level=res,
                mesh_prefix="mesh",
                input_window=config.get('mp_pde_input_window', 30),
                output_window=config.get('mp_pde_output_window', 20),
                Delta_t=config.get('dt', 0.25),
                n_timesteps=config.get('tmax', 50),
                train=True,
                data_split=data_split_train
            )

        if config['normalize']:
            # Normalize test dataset with respect to train dataset of the same resolution
            if config['dataset'] == 'darcy':
                norm_darcy_dataset(dep_dataset=train_datasets[res], ctrl_dataset=train_datasets[res],
                                   use_grads=config['use_grads'])
            elif config['dataset'] == 'adr':
                norm_dataset(dep_dataset=train_datasets[res], ctrl_dataset=train_datasets[res])
            elif config['dataset'] == 'navier_stokes':
                norm_dataset(dep_dataset=train_datasets[res], ctrl_dataset=train_datasets[res])

    # Load test datasets for the config[base_res]
    if config['dataset'] == 'navier_stokes':
        test_datasets[config['base_res']] = MultiResolutionDataset3D(
            config,
            data_dir=config['data_dir'],
            resolution=config['base_res'],
            train=False,
            load_in_memory=config['load_in_memory'],
            normalize=config['normalize'],
            add_coords=add_coords,
        )
    elif config['dataset'] in ['darcy', 'adr']:
        test_datasets[config['base_res']] = MultiResolutionDataset(
            config,
            config['data_dir'],
            config['base_res'],
            train=False,
            load_in_memory=config['load_in_memory'],
            normalize=config['normalize'],
            add_coords=add_coords,
            use_grads=use_grads
        )
    elif config['dataset'] == 'jeb':
        level_dir = os.path.join(config['data_dir'], "GEJetEngineBracket", f"level_{config['base_res']}")
        test_datasets[config['base_res']] = JEBDataset(
            config=config,
            root=os.path.join(level_dir, "test"),
            data_dir=config['data_dir'],
            resolution=config['base_res'],
            train=False,
            load_in_memory=config.get('load_in_memory', True),
            normalize=config.get('normalize', True)
        )
    elif config['dataset'] == 'FlowPastCylinder':
        source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "navier_stokes_data")

        # Use filtered indices for FlowPastCylinder
        test_indices = [i for i in range(150, 200)]
        N_test = len(test_indices)
        data_split_test = slice(150, 200)

        # Add the _direct suffix to the dataset name when direct loading is enabled
        direct_suffix = "_direct" if config.get('FPC_direct', False) else ""
        dataset_name_with_suffix = f"{config['dataset']}{direct_suffix}"
        test_data_res = os.path.join(config['data_dir'], dataset_name_with_suffix,
                                     f"level_{config['base_res']}_N{N_test}_test")
        print(f"Loading test data {test_data_res}")

        test_datasets[config['base_res']] = NavierStokesDataset(
            config=config,
            root=test_data_res,
            data_dir=source_dir,  # mesh_data_dir,
            mesh_level=config['base_res'],
            mesh_prefix="mesh",
            input_window=config.get('mp_pde_input_window', 30),
            output_window=config.get('mp_pde_output_window', 20),
            Delta_t=config.get('dt', 0.25),
            n_timesteps=config.get('tmax', 50),
            train=False,
            data_split=data_split_test
        )

    if config['normalize']:
        # Normalize test dataset with respect to train dataset of the same resolution
        if config['dataset'] == 'darcy':
            norm_darcy_dataset(dep_dataset=test_datasets[config['base_res']],
                               ctrl_dataset=train_datasets[config['base_res']],
                               use_grads=config['use_grads'])
        elif config['dataset'] == 'adr':
            norm_dataset(dep_dataset=test_datasets[config['base_res']],
                         ctrl_dataset=train_datasets[config['base_res']])
        elif config['dataset'] == 'navier_stokes':
            norm_dataset(dep_dataset=test_datasets[config['base_res']],
                         ctrl_dataset=train_datasets[config['base_res']])

    if config['base_res'] in train_datasets:
        base_train = train_datasets[config['base_res']]
        if hasattr(base_train, 'output_mean') and hasattr(base_train, 'output_std'):
            dataset_specific['eval_output_mean'] = base_train.output_mean
            dataset_specific['eval_output_std'] = base_train.output_std
            dataset_specific['eval_resolution'] = config['base_res']

    # If the base-resolution train dataset was loaded only for test
    # normalization, drop it after retaining its eval normalization stats.
    if config['base_res'] not in c2f_resolutions:
        del train_datasets[config['base_res']]

    if config['load_gpu'] and not config['load_gpu_epoch']:
        for res in tqdm(c2f_resolutions, desc="Loading datasets to gpu"):
            train_datasets[res].to_device(device)
        test_datasets[config['base_res']].to_device(device)

    # Print dataset information using the finest resolution dataset
    print_dataset_info(train_datasets[c2f_resolutions[-1]], test_datasets[config['base_res']], config)

    if config['dataset'] == 'darcy':
        input_channels = 1
        if config['use_grads']:
            input_channels += 3
        if config['model'] == 'fno' and config.get('add_coords', False):
            input_channels += 2
    elif config['dataset'] == 'adr':
        input_channels = 1
    elif config['dataset'] == 'jeb':
        input_channels = 3
    elif config['dataset'] == 'navier_stokes':
        # T_in timesteps + 3 coords (if add_coords=True)
        input_channels = config['T_in']
        if config.get('add_coords', False):
            input_channels += 3
    elif config['dataset'] == 'FlowPastCylinder':
        input_channels = None

    # Get input size and channels from finest resolution dataset
    if config['override_res']:
        print(f"Using override_res: {config['override_res']}")
        input_size = config['override_res']
    else:
        input_size = config['base_res']

    config['input_channels'] = input_channels
    config['input_size'] = input_size

    return train_datasets, test_datasets, input_channels, input_size, dataset_specific


def data_MLMC_analysis(data_dir, dataset, resolutions, T_in, T, dt=None, max_samples=None):
    """Simple MLMC data diagnostics: level differences and correlations."""

    datasets = {}
    full_time_data = {}
    if dataset == 'navier_stokes':
        for res in resolutions:
            path = os.path.join(data_dir, f"ns_train_r{res}.pt")
            data = torch.load(path, weights_only=True)
            # ns2d_time files already contain full time window [0, T_in+T)
            datasets[res] = data['u']            # [N, H, W, T_total]
            full_time_data[res] = data['u']      # same tensor for regularity
    elif dataset == 'darcy':
        for res in resolutions:
            path = os.path.join(data_dir, f"train_r{res}.pt")
            data = torch.load(path, weights_only=True)
            datasets[res] = data['sol']        # [N, H, W]
            full_time_data[res] = data['sol']  # no time dimension
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    n = min(v.shape[0] for v in datasets.values())
    if max_samples is not None:
        n = min(n, max_samples)
    idx = slice(0, n)

    print(f"Using {n} samples for analysis ({dataset})")

    def l2_norm(x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x.reshape(x.shape[0], -1), p=2, dim=1)

    for coarse_res, fine_res in zip(resolutions[:-1], resolutions[1:]):
        u_c = datasets[coarse_res][idx]
        u_f = datasets[fine_res][idx]

        if dataset == 'darcy':
            # Upscale coarse Darcy solution to fine grid
            if u_c.shape[1] != u_f.shape[1]:
                u_c = upscale_2d(u_c, target_size=u_f.shape[1], method='bilinear')
        else:
            # Spatiotemporal: [N, H, W, T], upscale coarse spatial grid to fine
            if u_c.shape[1] != u_f.shape[1]:
                N, Hc, Wc, Td = u_c.shape
                _, Hf, Wf, _ = u_f.shape
                u_c_flat = u_c.permute(0, 3, 1, 2).reshape(-1, Hc, Wc)
                u_c_up = upscale_2d(u_c_flat, target_size=Hf, method='bilinear')
                u_c = u_c_up.reshape(N, Td, Hf, Wf).permute(0, 2, 3, 1)

        if dataset == 'darcy':
            u_c = u_c.unsqueeze(-1)
            u_f = u_f.unsqueeze(-1)

        diff = u_f - u_c
        diff_norm = l2_norm(diff)
        fine_norm = l2_norm(u_f)
        coarse_norm = l2_norm(u_c)

        diff_mean = diff_norm.mean().item()
        diff_std = diff_norm.std().item()

        # Relative difference ||u_f - u_c||_2 / ||u_f||_2
        rel_diff = (diff_norm / (fine_norm + 1e-8))
        rel_mean = rel_diff.mean().item()
        rel_std = rel_diff.std().item()

        fn = fine_norm - fine_norm.mean()
        cn = coarse_norm - coarse_norm.mean()
        cov = (fn * cn).mean()
        corr = (cov / (fine_norm.std() * coarse_norm.std() + 1e-8)).item()

        print(f"Pair {fine_res}/{coarse_res}:")
        print(f"  E[||u_f - u_c||_2]         = {diff_mean:.6e}, std = {diff_std:.6e}")
        print(f"  E[||u_f - u_c||_2/||u_f||_2] = {rel_mean:.6e}, std = {rel_std:.6e}")
        print(f"  corr(||u_f||_2, ||u_c||_2) = {corr:.4f}")

    # Regularity diagnostics on finest resolution only.
    finest_res = resolutions[-1]
    u_full = full_time_data[finest_res][idx]

    print("Regularity diagnostics (finest level):")

    if dataset == 'darcy':
        # Stationary problem: report spatial regularity only (no time-based stats).
        u_all = u_full.cpu().numpy()          # [N, H, W]
        abs_u = np.abs(u_all)
        linf_all = float(abs_u.max())
        H = u_all.shape[-2]
        dx = 1.0 / H
        mean_grad_l2, max_grad = compute_grad_regularity(u_all, dx=dx)
        print(f"  L_inf(|u|)       = {linf_all:.4e} (max over N, x)")
        print(f"  mean(||∇u||_2)   = {mean_grad_l2:.4e} (spatial, over N,x)")
        print(f"  L_inf(||∇u||)    = {max_grad:.4e} (max over N, x)")
    else:  # navier_stokes
        # Time-dependent problem (Navier-Stokes): use full time window [0, T_in+T].
        u_sample = u_full[0].cpu().numpy()    # [H, W, T_total]
        u_np = np.transpose(u_sample, (2, 0, 1))  # [T_total, H, W]
        eff_dt = dt if dt is not None else 1.0 / u_np.shape[0]
        H = u_np.shape[1]
        dx = 1.0 / H                           # assume unit spatial domain

        cfl_hist, osc_hist = compute_cfl_osc(u_np, dt=eff_dt, dx=dx)
        print(f"  mean(CFL_max) = {float(cfl_hist.mean()):.4f}, max(CFL_max) = {float(cfl_hist.max()):.4f}")
        print(f"  mean(osc)     = {float(osc_hist.mean()):.4e}, max(osc)     = {float(osc_hist.max()):.4e}")

        # Gradient regularity over all samples / space / time
        u_all = u_full.cpu().numpy()          # [N, H, W, T_total]
        abs_u = np.abs(u_all)
        linf_all = float(abs_u.max())
        u_all_bt = np.moveaxis(u_all, -1, -3)  # [N, T_total, H, W]
        mean_grad_l2, max_grad = compute_grad_regularity(u_all_bt, dx=dx)

        print(f"  L_inf(|u|)       = {linf_all:.4e} (max over N, x, t in [0, T_in+T])")
        print(f"  mean(||∇u||_2)   = {mean_grad_l2:.4e} (over N,x,t)")
        print(f"  L_inf(||∇u||)    = {max_grad:.4e} (max over N,x,t)")


if __name__ == "__main__":
    cases = [
        {
            'name': 'Darcy',
            'data_dir': "examples/pdes/darcy_flow/data/darcy2d",
            'dataset': 'darcy',
            'resolutions': [30, 60, 120, 241],
            'max_samples': None,
            'T_in': 0,
            'T': 1,
            'dt': None

        },
        {
            'name': 'Navier-Stokes',
            'data_dir': "examples/pdes/navier_stokes/data/ns2d_time",
            'dataset': 'navier_stokes',
            'resolutions': [8, 16, 32, 64],
            'T_in': 10,
            'T': 40,
            'dt': 0.0001,
            'max_samples': None,
        },
    ]

    for cfg in cases:
        print("\n" + "=" * 80)
        print(f"Dataset: {cfg['name']}")
        # Provide sensible defaults for static Darcy dataset
        if cfg['dataset'] == 'darcy':
            T_in = 0
            T = 1
            dt = None
        else:
            T_in = cfg.get('T_in', 0)
            T = cfg.get('T', 1)
            dt = cfg.get('dt', None)
        data_MLMC_analysis(
            data_dir=cfg['data_dir'],
            dataset=cfg['dataset'],
            resolutions=cfg['resolutions'],
            T_in=T_in,
            T=T,
            dt=dt,
            max_samples=cfg['max_samples'],
        )
