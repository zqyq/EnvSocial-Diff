import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Obs_Aware(nn.Module):
    def __init__(self, img_feat_dim=512, text_feat_dim=768, hidden_dim=256,out_dim=64, n_heads=4):
        super().__init__()

        self.poi_context_encoder = nn.Sequential(
            nn.Linear(img_feat_dim + text_feat_dim + 4 + hidden_dim+hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.sc_encoder = nn.Sequential(
            nn.Linear(49*2816, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.kv_proj = nn.Sequential(
            nn.Linear(img_feat_dim + text_feat_dim + 5 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.pos_fc = nn.Sequential(
            nn.Linear(5, hidden_dim),  # 5维度，x,y,w,h,argle# 编码障碍物box位置
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.query_proj = nn.Sequential(
            nn.Linear(6 + 5, hidden_dim),  # ped_state(6) + relative(delta 2 + dist 1 + dir_cos 2)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def pixel_2_world(self, px, py, H):
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
            return world
        elif pix.dim() == 3:
            # 多个batch
            B, N, _ = pix.shape
            pix = pix.transpose(1, 2)  # [B, 3, N]
            H = H.unsqueeze(0).expand(B, -1, -1)  # [B, 3, 3]
            world = torch.bmm(H, pix)  # [B, 3, N]
            world = world[:, :2, :] / (world[:, 2:, :] + 1e-6)
            world = world.transpose(1, 2)  # [B, N, 2]
            return world[0]


    def forward(self, poi_img_feats, poi_text_feats, poi_boxes,
                person_feats,
                image, text,
                H):

        """
        poi_img_feats: (M, D_img)
        poi_text_feats: (M, D_txt)
        poi_boxes: (M, 4)
        person_feats (B,N,6)
        scene_img_feat, scene_text_feat: (D_img), (D_txt)
        """
        M = poi_img_feats.shape[0]
        B, N, _ = person_feats.shape
        x,y,w,h,_ = torch.chunk(poi_boxes, 5, dim=-1)
        #整个场景
        text = text.unsqueeze(0).expand(image.size(0), -1)
        scene_feats = torch.cat([image, text], dim=-1) # #49, 2816 整张场景的语义
        scene_feats = self.sc_encoder(scene_feats.reshape(-1)) # dim
        scene_feats = scene_feats.unsqueeze(0).expand(M,-1) # M d

        #障碍物
        person_feats = torch.nan_to_num(person_feats, nan=0.0)
        poi_feats = torch.cat([poi_img_feats,poi_text_feats,poi_boxes],dim=-1) # M d1+d2+5
        kv_input = torch.cat([poi_feats, scene_feats], dim=-1)  # M hid+d1+d2+5
        kv_input = kv_input.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
        key_value  = self.kv_proj(kv_input)



        world = self.pixel_2_world(x, y, H) # M 2
        world = world.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1) # B N M 2
        ped_pos = person_feats[...,:2].unsqueeze(-2).expand(-1,-1,M,-1) # B N M 2
        delta = ped_pos - world
        dist = torch.norm(delta, p=2, dim=-1).unsqueeze(-1)
        box = poi_boxes.unsqueeze(0).unsqueeze(0).expand(B,N,-1,-1)
        person = person_feats.unsqueeze(-2).expand(-1,-1,M,-1)
        query_input = torch.cat([person, delta, dist,  box[...,2:4]],dim=-1) # B N M 11
        query = self.query_proj(query_input)

        Q = query.view(B * N, M, -1)
        K = key_value.view(B * N, M, -1)
        V = key_value.view(B * N, M, -1)

        attn_output, _ = self.attn(Q, K, V)  # [B*N, M, hidden_dim]
        attn_output = attn_output.mean(dim=1)  # [B*N, hidden_dim]
        attn_output = attn_output.view(B, N, -1)  # [B, N, hidden_dim]
        nfluence_feat = self.out_proj(attn_output)  # [B, M, output_dim]

        return nfluence_feat
