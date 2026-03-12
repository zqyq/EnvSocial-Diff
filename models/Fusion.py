import torch
import torch.nn.functional as F
import torch.nn as nn


class LocalFusion(nn.Module):
    def __init__(self, device, text_dim, image_dim, out_dim, dropout=0.1):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.device = device

    def forward(self, text_emb, image_emb):
        text = self.text_encoder(text_emb) # M d
        image = self.image_encoder(image_emb)# M d
        combined = torch.cat((text, image), dim=-1)
        fusion = self.fusion(combined)
        return fusion



class GlobalFusion(nn.Module):

    def __init__(self, image_dim=2048, text_dim=768, embed_dim=512, nhead=8, num_layers=4, dropout=0.1,device="cpu"):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, image_feats, text_feats):
        """
        image_feats: [49, 2048]  单张图像patch特征
        text_feats:  [768]       单条文本特征
        """
        # 线性映射
        img_embeds = self.image_proj(image_feats)  # [49, embed_dim]
        txt_embed = self.text_proj(text_feats).unsqueeze(0)  # [1, embed_dim]

        # 拼接文本token到序列首位
        combined = torch.cat([txt_embed, img_embeds], dim=0)  # [50, embed_dim]

        # Transformer输入格式 [seq_len, batch, embed_dim]，这里batch=1
        combined = combined.unsqueeze(1)  # [50, 1, embed_dim]

        fused = self.transformer(combined)  # [50, 1, embed_dim]

        fused = fused.squeeze(1)  # [50, embed_dim]

        # 第一个token作为全局融合特征
        # global_feat = fused[0]  # [embed_dim]

        return  fused  # 返回全局特征和序列特征