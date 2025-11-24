"""GNN models for mesh-based PDE solving.

Message Passing PDE Solver for flow problems like FlowPastCylinder.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, InstanceNorm
from collections import namedtuple


class Swish(nn.Module):
    """Swish activation function."""
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """Message passing layer for PDE solving."""
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 n_variables: int):
        """
        Initialize message passing layers.
        
        Args:
            in_features: Number of node input features
            out_features: Number of node output features
            hidden_features: Number of hidden features
            time_window: Number of input/output timesteps (temporal bundling)
            n_variables: Number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=0, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        
        self.message_net_1 = nn.Sequential(
            nn.Linear(2 * in_features + time_window*2 + 2 + n_variables, hidden_features),
            Swish()
        )
        self.message_net_2 = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            Swish()
        )
        self.update_net_1 = nn.Sequential(
            nn.Linear(in_features + hidden_features + n_variables, hidden_features),
            Swish()
        )
        self.update_net_2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            Swish()
        )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch):
        """Propagate messages along edges."""
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """Message update."""
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """Node update."""
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MP_PDE_Solver(torch.nn.Module):
    """Message Passing PDE Solver for flow problems."""
    
    def __init__(self,
                 pde: namedtuple,
                 input_window: int = 25,
                 output_window: int = 20,
                 block_size: int = 5,
                 hidden_features: int = 128,
                 hidden_layer: int = 6,
                 eq_variables: dict = {},
                 autoregressive: bool = False):
        """
        Initialize MP-PDE solver.
        
        Args:
            pde: PDE namedtuple with fields L, tmax, dt
            input_window: Number of input timesteps
            output_window: Number of output timesteps
            block_size: Number of timesteps to predict at once in autoregressive mode
            hidden_features: Number of hidden features
            hidden_layer: Number of hidden layers
            eq_variables: Dictionary of equation specific parameters
            autoregressive: Whether to use autoregressive mode
        """
        super(MP_PDE_Solver, self).__init__()
        self.pde = pde
        self.input_window = input_window
        self.output_window = output_window
        self.block_size = block_size
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.eq_variables = eq_variables
        self.autoregressive = autoregressive

        # GNN layers
        time_window = self.block_size if autoregressive else self.input_window
        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=time_window,  
            n_variables=len(self.eq_variables) + 1
        ) for _ in range(self.hidden_layer - 1)))

        self.gnn_layers.append(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=time_window,
            n_variables=len(self.eq_variables) + 1
        ))

        # MLP for embedding
        self.embedding_mlp = nn.Sequential(
            nn.Linear(time_window*2 + 3 + len(self.eq_variables), self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )

        # Output MLP
        output_size = self.block_size if autoregressive else self.output_window
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish(),
            nn.Linear(self.hidden_features, output_size*2)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass."""
        u = data.x
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch

        # Time variable
        t = torch.zeros_like(pos[:, 0])[:, None] / self.pde.tmax
        variables = t

        # Encode equation specific parameters
        if "inlet" in self.eq_variables.keys():
            variables = torch.cat((variables, data.inlet_mask.float()[:, None]), -1)
        if "sides" in self.eq_variables.keys():
            variables = torch.cat((variables, data.sides_mask.float()[:, None]), -1)
        if "obstacle" in self.eq_variables.keys():
            variables = torch.cat((variables, data.obstacle_mask.float()[:, None]), -1)
        if "outlet" in self.eq_variables.keys():
            variables = torch.cat((variables, data.outlet_mask.float()[:, None]), -1)

        # Encoder
        u_flat = u.reshape(u.shape[0], -1)
        node_input = torch.cat((u_flat, pos, variables), -1)
        h = self.embedding_mlp(node_input)

        # Message passing
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u_flat, pos, variables, edge_index, batch)

        # Decoder
        out = self.output_mlp(h).reshape(h.shape[0], -1, 2)

        return out
