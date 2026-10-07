import torch
import torch.nn as nn

from torch_geometric import nn as gnn
from torch_geometric.data import HeteroData


class MPNNMapEncoder(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 edge_dim: int = 6):
        super().__init__()

        self.num_classes = edge_dim

        self.map_encoder = gnn.Sequential('x, edge_index, edge_attr', [
            (gnn.GraphConv(num_inputs, num_hidden), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (gnn.GraphConv(num_hidden, num_hidden // 2), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (gnn.GraphConv(num_hidden // 2, num_hidden), 'x, edge_index, edge_attr -> x'),
        ])

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, num_hidden),
            nn.GELU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        num_map_nodes = data['map_point']["num_nodes"]
        pos = data['map_point']["position"]
        edge_index = data['map_point', 'to', 'map_point']['edge_index']
        edge_type = data['map_point', 'to', 'map_point']['type'].squeeze()
        edge_type = nn.functional.one_hot(edge_type, num_classes=self.num_classes).float()

        edge_attr = self.edge_encoder(edge_type)
        x = self.map_encoder(pos, edge_index, edge_attr)

        return x[num_map_nodes:]


class GATMapEncoder(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 edge_dim: int = 6):
        super().__init__()

        self.num_classes = edge_dim

        self.map_encoder = gnn.Sequential('x, edge_index, edge_attr', [
            (gnn.GATv2Conv(num_inputs, num_hidden, edge_dim=edge_dim, concat=False), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (gnn.GATv2Conv(num_hidden, num_hidden, edge_dim=edge_dim, concat=False), 'x, edge_index, edge_attr -> x'),
            nn.GELU(),
            (gnn.GATv2Conv(num_hidden, num_hidden, edge_dim=edge_dim, concat=False), 'x, edge_index, edge_attr -> x'),
        ])

    def forward(self, data: HeteroData) -> torch.Tensor:
        num_map_nodes = data['map_point']["num_nodes"]
        pos = data['map_point']["position"]
        edge_index = data['map_point', 'to', 'map_point']['edge_index']
        edge_type = data['map_point', 'to', 'map_point']['type'].squeeze()
        edge_attr = nn.functional.one_hot(edge_type, num_classes=self.num_classes).float()

        x = self.map_encoder(pos, edge_index, edge_attr)

        return x[num_map_nodes:]

