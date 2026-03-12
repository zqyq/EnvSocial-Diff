import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
# 计算机视觉
from torchvision.transforms import functional as TF
class SceneBrightnessAwareContext(nn.Module):
    def __init__(self, scene_feat_dim=2048, hidden_dim=128, device='cuda', config=None, out_dim = 128,temperature=10000.0):
        super().__init__()
        self.device = device
        self.config = config
        self.image = config.bg_image
        self.temperature = temperature
        # 亮度处理
        self.brightness_fc = nn.Linear(1, scene_feat_dim+hidden_dim)

        # 注意力层
        self.scene_k = nn.Linear(scene_feat_dim, hidden_dim)
        self.scene_v = nn.Linear(scene_feat_dim, hidden_dim)
        self.person_query = nn.Linear(6, hidden_dim)

        self.out = nn.Linear(scene_feat_dim+hidden_dim, hidden_dim)
        # 初始化
        self._init_weights()
        grid_size = 7
        # 生成网格坐标
        # dim = scene_feat_dim
        dim = hidden_dim
        x = torch.arange(grid_size, dtype=torch.float32)
        y = torch.arange(grid_size, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')

        # 计算位置编码
        dim_t = torch.arange(dim // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (dim // 2))

        pos_x = grid_x.unsqueeze(-1) / dim_t
        pos_y = grid_y.unsqueeze(-1) / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)

        self.register_buffer('pos_embed', torch.cat([pos_x, pos_y], dim=-1).view(-1, dim))

    def _init_weights(self):
        nn.init.xavier_uniform_(self.brightness_fc.weight)
        nn.init.zeros_(self.brightness_fc.bias)

    def get_brightness(self, bg_image, num_patches=7):
        """优化后的亮度提取"""
        if isinstance(bg_image, str):
            img = Image.open(bg_image).convert('RGB')
        else:
            img = bg_image

        hsv = TF.to_tensor(img.convert("HSV")).to(self.device)
        v = hsv[2:3]  # [1,H,W]
        patches = F.adaptive_avg_pool2d(v, (num_patches, num_patches))
        return patches.view(-1, 1)  # [L,1]

    # def forward(self, person_feats, image):
    #     B, N, _ = person_feats.shape
    #     L = image .size(0)
    #     image = image + self.pos_embed.to(self.device)
    #
    #     # 亮度增强（带残差）
    #     brightness = self.get_brightness(self.image, int(L ** 0.5))  # [L,1]
    #     gate = torch.sigmoid(self.brightness_fc(brightness))  # [L,D]
    #     # scene_feats = scene_feats * gate + scene_feats * (1 - gate)
    #     scene_feats = image  + gate * image
    #     person_feats = torch.nan_to_num(person_feats, nan=0.0)  # bool
    #     # 注意力计算
    #     Q = self.person_query(person_feats)  # [B,N,D]
    #     K = self.scene_k(scene_feats)  # [L,D]
    #     V = self.scene_v(scene_feats)  # [L,D]
    #     if K.dim() == 2:
    #         K = K.unsqueeze(0).expand(Q.size(0), -1, -1)  # [B,L,D]
    #         V = V.unsqueeze(0).expand(Q.size(0), -1, -1)  # 同理处理V
    #     attn = torch.einsum('bnd,bld->bnl', Q, K) / (K.size(-1) ** 0.5)
    #     attn = F.softmax(attn, dim=-1)
    #     context = torch.einsum('bnl,bld->bnd', attn, V)
    #
    #     return context

    def forward(self, person_feats, image):
        B, N, _ = person_feats.shape
        L = image .size(0)
        # image = image + self.pos_embed.to(self.device)
        image  = torch.cat((image, self.pos_embed), dim=-1) # 2048  + 128
         # 亮度增强（带残差）
        brightness = self.get_brightness(self.image, int(L ** 0.5))  # [L,1]
        gate = torch.sigmoid(self.brightness_fc(brightness))  # [L,D]
        # scene_feats = scene_feats * gate + scene_feats * (1 - gate)
        scene_feats = image  + gate * image
        scene_feats = self.out(scene_feats)


        # person_feats = torch.nan_to_num(person_feats, nan=0.0)  # bool
        # # 注意力计算
        # Q = self.person_query(person_feats)  # [B,N,D]
        # K = self.scene_k(scene_feats)  # [L,D]
        # V = self.scene_v(scene_feats)  # [L,D]
        # if K.dim() == 2:
        #     K = K.unsqueeze(0).expand(Q.size(0), -1, -1)  # [B,L,D]
        #     V = V.unsqueeze(0).expand(Q.size(0), -1, -1)  # 同理处理V
        # attn = torch.einsum('bnd,bld->bnl', Q, K) / (K.size(-1) ** 0.5)
        # attn = F.softmax(attn, dim=-1)
        # context = torch.einsum('bnl,bld->bnd', attn, V)

        return scene_feats