import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pyexpat import features
from torch import nn
import numpy as np
import cv2
import math
class ImageBase(nn.Module):
    def __init__(self,device,weights_path):
        super(ImageBase, self).__init__()
        self.device = device
        self.model = ResNet(weights_path, device)


    def get_image_patch(self, image_path:str, boxes):
        image = Image.open(image_path).convert("RGB")
        img_cv = np.array(image)
        h_img, w_img = img_cv.shape[:2]

        patches = []
        for i in range(boxes.shape[0]):
            cx, cy, w, h, angle_rad = boxes[i]

            # 如果是 tensor，转成 float
            if isinstance(cx, torch.Tensor):
                cx, cy, w, h, angle_rad = (
                    cx.item(), cy.item(), w.item(), h.item(), angle_rad.item()
                )

            angle_deg = math.degrees(angle_rad)

            M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
            rotated = cv2.warpAffine(img_cv, M, (w_img, h_img), flags=cv2.INTER_LINEAR)

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # 越界处理
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w_img), min(y2, h_img)

            roi = rotated[y1:y2, x1:x2] #目标区域
            roi_pil = Image.fromarray(roi)
            features = self.model(roi_pil)
            patches.append(features)

        return torch.stack(patches, dim=0)


class ResNet(nn.Module):
    def __init__(self,weights_path=None, device=None):
        super(ResNet, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.device = device
        state_dict = torch.load(weights_path)
        resnet50.load_state_dict(state_dict)
        resnet50.eval()
        self.model = torch.nn.Sequential(*list(resnet50.children())[:-1]).to(device) # 去掉fc层 获得特征
        self.preprocess = transforms.Compose([
            transforms.Resize(256),  # 先resize短边256
            transforms.CenterCrop(224),  # 再中心裁剪到224×224
            transforms.ToTensor(),  # 转tensor，C*H*W格式，且归一化到[0,1]
            transforms.Normalize(  # 归一化，ImageNet均值方差
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    def forward(self, image_path):

     # 读图
        if isinstance(image_path, str):

            img = Image.open(image_path).convert('RGB')

            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        else:
            input_tensor = self.preprocess(image_path).unsqueeze(0).to(self.device)

        # self.model.to(self.device)
        with torch.no_grad():
            features = self.model(input_tensor)


        features = features.squeeze()  # [2048]
        return features

class ResNet2(nn.Module):
    def __init__(self, weights_path=None, device=None):
        super(ResNet2, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        self.device = device
        state_dict = torch.load(weights_path, map_location=device)
        resnet50.load_state_dict(state_dict)
        resnet50 = resnet50.to(device).eval()
        self.model = torch.nn.Sequential(*list(resnet50.children())[:-1])  # 去掉fc层
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, image_inputs):
        # 支持单张图片 (PIL or path) 或 List[PIL or path]
        if isinstance(image_inputs, (str, Image.Image)):
            image_inputs = [image_inputs]  # 转成列表方便处理

        # 预处理所有图像
        tensors = []
        for img in image_inputs:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            tensor = self.preprocess(img)
            tensors.append(tensor)

        batch_tensor = torch.stack(tensors).to(self.device)  # [B, 3, 224, 224]

        with torch.no_grad():
            features = self.model(batch_tensor)  # [B, 2048, 1, 1]
            features = features.view(batch_tensor.size(0), -1)  # [B, 2048]

        return features  # 返回多个特征向量

class ResNetGlobal(nn.Module):
    def __init__(self, weights_path=None, device=None, out_dim=512):
        super().__init__()
        self.device = device

        # 加载resnet50预训练权重
        resnet50 = models.resnet50(pretrained=False)
        if weights_path:
            state_dict = torch.load(weights_path)
            resnet50.load_state_dict(state_dict)
        resnet50.eval()

        # 去掉fc和avgpool，只保留卷积层输出feature map
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2]).to(device)  # 输出[B, 2048, 7, 7]

        # # 线性层降维2048->out_dim
        # self.linear_proj = nn.Linear(2048, out_dim).to(device)

        # 预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, image):
        # image可以是PIL图或者tensor
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
            input_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        else:
            # 假设已是tensor
            input_tensor = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat_map = self.backbone(input_tensor)  # [1, 2048, 7, 7]

        B, C, H, W = feat_map.shape
        feat_map = feat_map.flatten(2).permute(0, 2, 1)  # [1, H*W, 2048] = [B, 49, 2048]

        # 线性降维
        # patch_feats = self.linear_proj(feat_map)  # [1, 49, out_dim]

        return feat_map[0]  # 返回patch序列特征