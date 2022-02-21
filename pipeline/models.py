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
    
    
from timm.models.resnet import resnet18
class ModelResnet18(nn.Module):
    def __init__(self, num_features):
        super(self.__class__, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(num_features, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        
    def forward(self, x):
        return self.model.forward(x)[:, 0]