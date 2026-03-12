# import torch
# import torch.nn as nn
#
# class BrightnessEncoder(nn.Module):
#     def __init__(self, in_channels=18, out_dim=64):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_channels * 37, 32, kernel_size=3, padding=1)  # (B*T, 1, 18) → (B*T, 32, 18)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)  # → (B*T, 32, 18)
#         self.relu2 = nn.ReLU()
#         self.pool = nn.AdaptiveAvgPool1d(1)  # → (B*T, 32, 1)
#         self.fc = nn.Linear(32, out_dim)     # → (B*T, 64)
#
#         self.skip_proj = nn.Conv1d(1, 32, kernel_size=1)  # (B*T, 1, 18) → (B*T, 32, 18)
#
#     def forward(self, bright):
#         B, T, N, D = bright.shape  # (B, T,N, 5)
#         x = bright.reshape(B * T , 1, N*D)         # → (B*T, 1, 18)
#         x = x.float()
#         skip = self.skip_proj(x)             # → (B*T, 32, 18)
#
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#
#         x = x + skip                         # 跳跃连接
#         x = self.pool(x).squeeze(-1)         # → (B*T, 32)
#         x = self.fc(x)                       # → (B*T, 64)
#         return x.view(B, T,  -1)              # → (B, T, N,,64)
#             # → (B, T, 64)
#


import torch
import torch.nn as nn


class BrightnessEncoder(nn.Module):
    # def __init__(self, in_dim=5, out_dim=64):
    #     super().__init__()
    #     self.conv1 = nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=3, padding=1)  # (B*T, 5, 37) → (B*T, 32, 37)
    #     self.relu1 = nn.ReLU()
    #     self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)  # → (B*T, 32, 37)
    #     self.relu2 = nn.ReLU()
    #     self.pool = nn.AdaptiveAvgPool1d(1)  # → (B*T, 32, 1)
    #     self.fc = nn.Linear(32, out_dim)     # → (B*T, 64)
    #
    #     self.skip_proj = nn.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=1)  # skip connection
    #
    # def forward(self, bright):
    #     bright = torch.nan_to_num(bright, nan=0.0)
    #     B, T, M, D = bright.shape  # (32, 144, 48, 5)
    #     x = bright.reshape(B * T, M, D).permute(0, 2, 1).float()
    #
    #
    #     skip = self.skip_proj(x)  # → (B*T, 32, 37)
    #
    #     x = self.conv1(x)
    #     x = self.relu1(x)
    #     x = self.conv2(x)
    #     x = self.relu2(x)
    #
    #     # x = x + skip  # skip connection
    #     x = self.pool(x).squeeze(-1)  # → (B*T, 32)
    #     x = self.fc(x)  # → (B*T, 64)
    #     return x.view(B, T, -1)  # → (B, T, 64)

    def __init__(self, in_dim=5, hidden_dim=32, out_dim=64):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.project = nn.Linear(hidden_dim, out_dim)

    def forward(self, bright):
        bright = torch.nan_to_num(bright, nan=0.0)  # [B, T, M, D]
        B, T, M, D = bright.shape

        x = bright.view(B * T, M, D)  # → (B*T, M, D)
        x = self.point_mlp(x[...,2:].float())  # → (B*T, M, hidden_dim)
        x = x.mean(dim=1)  # 平均池化 → (B*T, hidden_dim)
        x = self.project(x)  # → (B*T, out_dim)
        return x.view(B, T, -1)  # → (B, T, out_dim)
