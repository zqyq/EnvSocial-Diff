from typing import Any
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import  numpy as np
import torchvision.transforms as T

class PositionAwareEnvAttention(nn.Module):
    def __init__(self, person_dim, env_dim, hidden_dim, out_dim=64, pos_embed_dim=32,device=None,config=None):
        super().__init__()
        self.query_fc = nn.Linear(person_dim, hidden_dim)
        self.key_fc = nn.Linear(env_dim, hidden_dim)
        self.value_fc = nn.Linear(env_dim, out_dim)
        self.pos_embed_fc = nn.Sequential(
            nn.Linear(4, pos_embed_dim),  # diffx diffy w h
            # nn.Linear(3, pos_embed_dim),  # diffx diffy w h
            nn.ReLU(),
            nn.Linear(pos_embed_dim, 1)   # scalar score adjustment
        )
        self.pos_fc = nn.Sequential(
            nn.Linear(5, pos_embed_dim),  #5维度，x,y,w,h,argle# 编码障碍物box位置
            nn.ReLU(),
            nn.Linear(pos_embed_dim, env_dim)
        )
        self.obs_query = nn.Linear(517, env_dim,device=device)
        #self.scene_value = nn.Linear(2816, out_dim)
        self.scene_kv = nn.Linear(2816, env_dim,device=device)
        self.final_fc = nn.Linear(env_dim, out_dim)
        self.person_query = nn.Linear(person_dim, hidden_dim)
        self.person_k = nn.Linear(person_dim, hidden_dim)
        self.device = device
        self.config = config

        # if config.use_bright2:
        #     self.brightness_fc = nn.Linear(1, 128, device=self.device)
        #     self.transform = T.Compose([
        #         T.Lambda(lambda x: x.convert("HSV") if isinstance(x, Image.Image) else x),
        #         T.Resize(256),
        #         T.CenterCrop(224),
        #         T.ToTensor()
        #     ])
        #     self.perceptual_weights = nn.Parameter(torch.tensor([0.0722, 0.7152, 0.2126])).to(device=self.device)
        #     # 动态范围处理
        #     self.tone_mapping = nn.Sequential(
        #         nn.Linear(1, 16),
        #         nn.ReLU(),
        #         nn.Linear(16, 1),
        #         nn.Sigmoid()
        #     ).to(device=self.device)
        #     self.adaptive_pool = nn.AdaptiveAvgPool2d(7).to(device=self.device)

    def pixel_2_world(self, px, py, H,label):
        H = torch.from_numpy(H).float().to(px.device)
        ones = torch.ones_like(px)
        pix = torch.stack([px, py, ones], dim=-1)
        pix = pix.transpose(0, 1)  # [3, N]
        if pix.dim() == 2:
            # 单个batch
            pix = pix.T  # [3, N]
            world = torch.matmul(H, pix)  # [3, N]
            world = world[:2, :] / (world[2:, :] + 1e-6)  # 防除0
            world = world.T  # [N, 2]
            if label == 'city':
                world = world/1000
            return world
        elif pix.dim() == 3:
            # 多个batch
            B, N, _ = pix.shape
            pix = pix.transpose(1, 2)  # [B, 3, N]
            H = H.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
            world = torch.bmm(H, pix)  # [B, 3, N]
            world = world[:, :2, :] / (world[:, 2:, :] + 1e-6)
            world = world.transpose(1, 2)  # [B, N, 2]
            if label == 'city':
                world = world/1000
            return world[0]




    # def forward(self, person_feats,mask, env_feats, env_boxes, H):
    #     """
    #     person_feats: [B, N, 6]
    #
    #     env_feats:    [M=6, env_dim]
    #     env_boxes:    [ M=6, 5]  # (cx, cy, w, h, angle)
    #     """
    #     person_feats = person_feats * mask
    #     B, N, _ = person_feats.shape
    #     M = env_feats.size(0)
    #     person_valid_mask = mask.any(dim=-1)
    #     person_feats = torch.nan_to_num(person_feats, nan=0.0)# bool
    #     person_feats = person_feats * mask
    #     Q = self.query_fc(person_feats)       # [B, N, 64]
    #     K = self.key_fc(env_feats)            # [6, 64]
    #     V = self.value_fc(env_feats)          # [6 ,64]
    #     person_pos = person_feats[...,:2] # 32 144 2
    #     # 计算空间相对关系 (x_diff, y_diff, w, h)
    #     cx, cy, w, h, _ = torch.chunk(env_boxes, 5, dim=-1)  # [B, M, 1]
    #     person_x, person_y = torch.chunk(person_pos, 2, dim=-1)  # [B, N, 1]
    #     pix = self.pixel_2_world(cx, cy, H)
    #
    #     # 扩展维度以匹配 N 与 M
    #     cx = pix[..., 0].unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)  # [B, N, M, 1]
    #     cy = pix[..., 1].unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
    #     w = w.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
    #     h = h.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
    #     px = person_x.unsqueeze(2).expand(-1, -1, M, -1)
    #     py = person_y.unsqueeze(2).expand(-1, -1, M, -1)
    #
    #     pos_diff = torch.cat([px - cx, py - cy, w, h], dim=-1)  # [B, N, M, 4]
    #     pos_score = self.pos_embed_fc(pos_diff).squeeze(-1)    # [B, N, M]
    #
    #     K = K.unsqueeze(0).expand(B,-1,-1)
    #     V = V.unsqueeze(0).expand(B,-1,-1)
    #     # 原始注意力分数
    #     attn_scores = torch.matmul(Q, K.transpose(1, 2)) / (K.size(-1) ** 0.5)  # [B, N, M]
    #     attn_scores += pos_score  # 加入空间引导
    #     attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, M]
    #     env_effect = torch.matmul(attn_weights, V)     # [B, N, out_dim]
    #     env_effect = env_effect * person_valid_mask .unsqueeze(-1)
    #     return env_effect

    def world2pixel(self, x,y, H,label):
        if label == "city":
            x = x * 1000
            y = y * 1000
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H).float().to(self.device)
        B, N, _ = x.shape

        x = x.squeeze(-1)
        y = y.squeeze(-1)
        H_inv = torch.inverse(H)
        ones = torch.ones_like(x)
        wrold_coords = torch.stack([x,y,ones], dim=-1)
        img_homo = wrold_coords @ H_inv.T
        img = img_homo[:,:,:2] / img_homo[:,:,2:3]
        if label == "ucy" or label == "zara":
            return img[..., 0] + 360, img[..., 1] * -1 + 288
        else:

            return img[..., 0], img[..., 1]
            # return img[..., 0]*-1 + 360, img[..., 1] * -1 + 288

    # def get_brightness(self, image_path, patch_size=7):
    #     """
    #     从图像中提取亮度 返回【L=49，1】
    #     """
    #     # 读取并转换
    #     img = Image.open(image_path)
    #     rgb = self.transform(img).to(self.device)  # [3,224,224]
    #
    #     # 计算感知亮度
    #     luminance = (rgb * self.perceptual_weights.view(3, 1, 1)).sum(0, keepdim=True)
    #
    #     # 动态范围调整
    #     luminance = self.tone_mapping(luminance.unsqueeze(-1)).squeeze()
    #
    #     # 自适应池化
    #     patches = self.adaptive_pool(luminance.unsqueeze(0))
    #     return patches.view(-1, 1)  # [49,1]





    def forward(self, person_feats, env_feats, env_boxes, H, image,text, label,bg,config):
        """
        person_feats: [B, N, 6]
        mask:
        env_feats:    [M=6, env_dim=512]
        env_boxes:    [ M=6, 5]  # (cx, cy, w, h, angle)
        image:    [49, 2078]
        text: 768
        """

        person_feats = torch.nan_to_num(person_feats, nan=0.0)
        B, N, _ = person_feats.shape
        M = env_feats.size(0)
        text = text.expand(image.size(0), -1)
        if self.config.use_obs_sae:
            scene_feats = torch.cat([image, text], dim=-1) #49, 2816 整张场景的语义
            obs_pos_enc = self.pos_fc(env_boxes)

            KV_scene = self.scene_kv(scene_feats)
            Q_input = torch.cat([env_feats, env_boxes], dim=-1)  # M,517=512+5 障碍物的位置感知
            Q_obs = self.obs_query(Q_input)
            attn_weights_obs = torch.matmul(Q_obs, KV_scene.T) / (Q_obs.size(-1) ** 0.5)  # [M, L]
            attn_weights_obs = F.softmax(attn_weights_obs, dim=-1)
            obs_context = torch.matmul(attn_weights_obs, KV_scene) #位置感知的障碍物
            obs_enhanced = obs_context + obs_pos_enc + env_feats  # looklook
        else:
            obs_enhanced = env_feats

        # person_feats = torch.nan_to_num(person_feats, nan=0.0)# bool
        # if label == "ucy":
        # person_feats = person_feats
        Q_person  = self.person_query(person_feats)       # [B, N, hid]
        K_obs  = self.key_fc(obs_enhanced).unsqueeze(0).expand(B, -1, -1)            # [B,6, hid]
        # V = self.value_fc(env_feats)          # [6 ,hid]

        person_pos = person_feats[...,:2] # 32 144 2
        # 计算空间相对关系 (x_diff, y_diff, w, h)
        cx, cy, w, h, r = torch.chunk(env_boxes, 5, dim=-1)  #  M, 1]
        person_x, person_y = torch.chunk(person_pos, 2, dim=-1)  # [B, N, 1]
        # pix = self.pixel_2_world(cx, cy, H)
        #
        # # 扩展维度以匹配 N 与 M
        # cx = pix[..., 0].unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)  # [B, N, M, 1]
        # cy = pix[..., 1].unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
        person_x,person_y = self.world2pixel(person_x,person_y,H,label)


        cx = cx.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)  # [B, N, M, 1]
        cy = cy.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
        w = w.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
        h = h.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
        r = r.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        # px = person_x.unsqueeze(2).expand(-1, -1, M, -1)
        # py = person_y.unsqueeze(2).expand(-1, -1, M, -1)
        px = person_x.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,M,1)
        py = person_y.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,M,1)

        #mask
        # pm_bn11 = mask.unsqueeze(-1).unsqueeze(-1)


        pos_diff = torch.cat([px - cx, py - cy, w, h], dim=-1)

        # pos_diff = pos_diff * pm_bn11

        # pos_dist = torch.sqrt((px-cx)**2 + (py-cy)**2 + 1e-6)
        # pos_diff = torch.cat([pos_dist, w, h], dim=-1)
        pos_score = self.pos_embed_fc(pos_diff).squeeze(-1)    # [B, N, M]


        attn_scores = torch.matmul(Q_person, K_obs.transpose(1, 2)) / (K_obs.size(-1) ** 0.5)  # [B, N, M]
        attn_scores += pos_score
        attn_weights = F.softmax(attn_scores, dim=-1)

        V_obs = obs_enhanced.unsqueeze(0).expand(B, -1, -1)  # [B, M, hidden]
        person_output = torch.matmul(attn_weights, V_obs)  # [B, N, hidden]



        person_output = self.final_fc(person_output)

        # person_output = person_output * mask.unsqueeze(-1).float()

        return person_output  # [B, N, out_dim]