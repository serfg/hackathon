import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
# from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, width=128, depth=2):
        super().__init__()
        self.conv = nn.ModuleList([torch_geometric.nn.GCNConv(num_node_features if not index else width, width)
                                   for index in range(depth)])
        self.mlp_first = nn.Sequential(nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
                                       nn.Linear(width, width))
        self.mlp = nn.Sequential(nn.Linear(width, width//2), nn.BatchNorm1d(width//2), nn.ReLU(),
                                 nn.Linear(width//2, 1))
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr.float()
        
        for conv in self.conv:
            x = conv(x, edge_index)
            x = F.relu(x)

        x = self.mlp_first(x)
        x = F.relu(x)
        x = torch_geometric.nn.global_max_pool(x, batch=data.batch)
        x = self.mlp(x)
        return x[:, 0]