import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SceneLightAttention(nn.Module):
    def __init__(self, scene_dim=2816, obs_dim=512, hidden_dim=512):
        """
        scene_dim: 整张场景语义特征（如图像+文本拼接后）维度
        obs_dim: 障碍物原始特征维度
        hidden_dim: 中间投影维度
        """
        super(SceneLightAttention, self).__init__()
        self.scene_kv = nn.Linear(scene_dim, hidden_dim)
        self.obs_query = nn.Linear(obs_dim, hidden_dim)
        self.brightness_fc = nn.Linear(1, 1)  # learnable bias weight
        self.output_fc = nn.Linear(hidden_dim, obs_dim)

        # 图像预处理
        self.image_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])

    def extract_brightness(self, image_path, patch_size=7):
        """
        从图像中提取亮度 patch：返回 [L=49, 1]
        """
        img = Image.open(image_path).convert("HSV")
        img = self.image_transform(img)
        v = np.array(img.split()[2]).astype(np.float32) / 255.0  # [H, W] ∈ [0, 1]
        brightness_tensor = torch.tensor(v).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        patch_brightness = torch.nn.functional.adaptive_avg_pool2d(
            brightness_tensor, (patch_size, patch_size)
        )
        return patch_brightness.view(-1, 1)  # [49, 1]
