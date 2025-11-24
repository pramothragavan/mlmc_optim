"""Model utilities.

Unified model creation from configuration.
"""

from collections import namedtuple
from examples.examples_src.models import FNO2d, FNO3d
from examples.examples_src.models import ParabolicCNN
from examples.examples_src.models import MP_PDE_Solver
from examples.examples_src.models import Trunk, PointCloudPerceiverChannelsEncoder


def get_model(config, input_channels=None, input_size=None, out_size=None):
    """Create model from configuration.
    
    Supported models:
    - fno/fno2d: Fourier Neural Operator for 2D problems (Darcy, ADR)
    - fno3d: Fourier Neural Operator for 3D problems (Navier-Stokes)
    - parabolic: Parabolic CNN for advection-diffusion (requires mesh_free_conv)
    - mp_pde: Message Passing GNN for flow problems (requires torch_geometric)
    - ginot: Transformer for point clouds (requires custom implementation)
    
    Args:
        config: Configuration dictionary
        input_channels: Number of input channels (for parabolic/cnn)
        input_size: Input spatial size (for parabolic/cnn)
        out_size: Output size (for parabolic/cnn)
        
    Returns:
        PyTorch model
    """
    model_type = config['model']
    
    # FNO models
    if model_type in ['fno', 'fno2d']:
        model = FNO2d(
            modes1=config['fno_modes'],
            modes2=config['fno_modes'],
            width=config['fno_width'],
            in_channels=input_channels,
            out_channels=config['out_channels'],
        )
    
    elif model_type == 'fno3d':
        model = FNO3d(
            modes1=config['fno_modes'],
            modes2=config['fno_modes'],
            modes3=config['fno_modes'],
            width=config['fno_width'],
            in_channels=input_channels,
            out_channels=config['out_channels'],
        )
    
    # Parabolic CNN
    elif model_type == 'parabolic':
        model = ParabolicCNN(
            config,
            in_channels=input_channels or config['in_channels'],
            hidden_dim=config['hidden_dim'],
            out_size=out_size or config['out_channels'],
            # Use override_res as a fallback for input_size when not explicitly provided
            input_size=input_size or config['override_res'],
            loss_type=config['loss_type'],
            max_pool=config['max_pool']
        )
    
    # GNN model
    elif model_type == 'mp_pde':
        PDE = namedtuple('PDE', ['L', 'tmax', 'dt'])
        pde = PDE(
            L=config['L'],
            tmax=config['tmax'],
            dt=config['dt']
        )
        
        model = MP_PDE_Solver(
            pde=pde,
            input_window=config['mp_pde_input_window'],
            output_window=config['mp_pde_output_window'],
            hidden_features=config['mp_pde_hidden_dim'],
            hidden_layer=config['mp_pde_layers'],
            eq_variables=config['mp_pde_eq_vars']
        )
    
    # GINOT transformer
    elif model_type == 'ginot':
        branch_args = {
            'in_dim': config['ginot_in_dim'],
            'global_pooling': config['ginot_global_pooling'],
            'latent_dim': config['ginot_latent_dim'],
            'embed_dim': config['ginot_embed_dim'],
            'num_layers': config['ginot_num_layers'],
            'num_heads': config['ginot_num_heads'],
            'padding_value': config['ginot_padding_value'],
            'dropout': config['ginot_dropout'],
        }
        
        trunk_args = {
            'embed_dim': config['ginot_embed_dim'],
            'cross_attn_layers': config['ginot_cross_attn_layers'],
            'num_heads': config['ginot_num_heads'],
            'in_channels': input_channels or config['ginot_in_channels'],
            'out_channels': out_size or config['ginot_out_channels'],
            'dropout': config['ginot_dropout'],
            'emd_version': config['ginot_emd_version'],
            'padding_value': config['ginot_padding_value'],
        }
        
        branch = PointCloudPerceiverChannelsEncoder(config, **branch_args)
        model = Trunk(config, branch, **trunk_args)
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}\n"
            f"Supported: fno/fno2d, fno3d, parabolic, mp_pde, ginot"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: {config.get('model', 'N/A')}")
    print(f"  Parameters: {n_params:,}")

    return model