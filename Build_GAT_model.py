import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
import torch.nn as nn


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=3, heads=6, dropout=0.2, residual=True):
        """GAT模型（128维+3层+6头）"""
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 输入层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * heads))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * heads))

        # 输出层
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)