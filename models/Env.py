import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        # 简单的线性变换 + 激活
        return self.activation(self.linear(x))


class PedEnvGNN(nn.Module):
    def __init__(self, ped_dim, env_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.num_layers = num_layers

        # 第一层输入是 ped_dim + env_dim
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(ped_dim + env_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
        self.layers.append(GNNLayer(hidden_dim, out_dim))

    def forward(self, ped_feats, env_feats):
        # 拼接行人与环境特征
        ped_feats = torch.nan_to_num(ped_feats, nan=0.0)
        x = torch.cat([ped_feats, env_feats], dim=-1)  # [B, N, ped_dim + env_dim]

        # 多层前馈GNN（无邻居聚合，类似MLP）
        for layer in self.layers:
            x = layer(x)  # [B, N, hidden/out_dim]

        return x




class GNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))

class PedEnvGNN_v2(nn.Module):
    def __init__(self, ped_dim, env_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.num_layers = num_layers

        # 环境门控模块：让环境来影响行人特征
        self.env_gate = nn.Sequential(
            nn.Linear(env_dim, ped_dim),
            nn.Sigmoid()
        )

        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(ped_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GNNLayer(hidden_dim, hidden_dim))
        self.layers.append(GNNLayer(hidden_dim, out_dim))

    def forward(self, ped_feats, env_feats):
        ped_feats = torch.nan_to_num(ped_feats, nan=0.0)
        env_feats = torch.nan_to_num(env_feats, nan=0.0)

        # 门控：用环境控制行人的特征更新
        gate = self.env_gate(env_feats)               # [B, N, ped_dim]
        gated_ped_feats = ped_feats * gate            # 每个行人的特征被自己的环境调制

        x = gated_ped_feats                           # 只使用 gated ped_feats，环境作用已注入

        for layer in self.layers:
            x = layer(x)

        return x
