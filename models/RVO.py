import numpy as np
from scipy.optimize import minimize
import torch
from torch.nn import Module
import data.data as DATA

class RVOModule(Module, DATA.Pedestrians):
    """RVO速度修正模块（无嵌套函数版）"""
    def __init__(self,tau=3, fix=0.2):
        super().__init__()
        self.tau = tau
        self.max_speed = 2.0
        self.fix = fix


    def correct_velocity(self, p_cur, v_cur, v_desire, near_ped_idx,neigh_ped_mask, collision_threshold):
        """
        批量处理RVO速度修正
        Args:
            p_cur: 当前帧位置 (32, 144, 2)
            v_cur: 上一帧速度 (32, 144, 2)
            v_desire: 期望速度 (32, 144, 2)
            near_ped_idx: 邻居索引 (32, 144, 6)
        Returns:
            v_next: 修正后的速度 (32, 144, 2)
        """
        B, N, K = near_ped_idx.shape

        p_for_gather = p_cur.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 2)
        v_for_gather = v_cur.unsqueeze(1).expand(-1,N,-1,-1)

        idx = near_ped_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B, N, K, 2)

        # 从 dim=2 上 gather（N→K）
        valid_neighbors_pos = torch.gather(p_for_gather, 2, idx)  # (B, N, K, 2)
        valid_neighbors_vel = torch.gather(v_for_gather, 2, idx)

        mask = neigh_ped_mask.unsqueeze(-1)  # (B, N, K, 1)
        valid_neighbors_pos = valid_neighbors_pos * mask
        # 1. 提取有效邻居的位置和速度 (B,N,K,2)
        valid_neighbors_vel = valid_neighbors_vel * mask


        # 3. 计算相对位置和速度 (B,N,K,2)
        rel_pos = p_cur.unsqueeze(2) - valid_neighbors_pos  # (B,N,K,2)
        rel_vel = v_desire.unsqueeze(2) - valid_neighbors_vel  # (B,N,K,2)

        # 4. 计算避撞约束 (B,N,K)
        dot_pos_vel = torch.sum(rel_pos * rel_vel, dim=-1)  # (B,N,K)
        dot_vel_vel = torch.sum(rel_vel * rel_vel, dim=-1) + 1e-6
        t_min = dot_pos_vel / (dot_vel_vel + 1e-6)
        t_min = torch.where(torch.isnan(t_min), torch.full_like(t_min, self.tau), t_min)
        t_clamp = torch.clamp(t_min, 0, self.tau)
        min_dist = torch.norm(rel_pos + t_clamp.unsqueeze(-1) * rel_vel, dim=-1)  # (B,N,K)

        min_dist = torch.where(torch.isnan(min_dist), torch.full_like(min_dist, 1e6), min_dist)
        # 5. 构建避撞修正向量 (B,N,2)
        collision_mask = (min_dist < collision_threshold) & neigh_ped_mask.bool()  # (B,N,K)
        normal = torch.stack([-rel_pos[..., 1], rel_pos[..., 0]], dim=-1)  # 垂直方向 (B,N,K,2)
        denom = torch.norm(normal, dim=-1, keepdim=True)
        normal = torch.where(
            collision_mask.unsqueeze(-1),
            normal / (denom + 1e-6),
            torch.zeros_like(normal)
        )
        # 建议打断点打印 denom 是否接近 0
        if (denom < 1e-5).any():
            print("Warning: norm too small in normal vector.")
        correction = torch.sum(normal, dim=2)  # (B,N,2)
        correction = torch.nan_to_num(correction, nan=0.0)
        # 6. 应用修正并限制速度
        v_next = v_desire + correction * self.fix # 修正系数可调
        # v_next = torch.nan_to_num(v_desire + correction * self.fix, nan=0.0, posinf=0.0, neginf=0.0)

        # v_next = torch.clip(v_next, -self.max_speed, self.max_speed)




        return v_next

    def correct_velocity2(self, p_cur, v_cur, v_desire, near_ped_idx, neigh_ped_mask, collision_threshold):
        """
            批量处理RVO速度修正
            Args:
                p_cur: 当前帧位置 (B, N, 2)
                v_cur: 上一帧速度 (B, N, 2)
                v_desire: 期望速度 (B, N, 2)
                near_ped_idx: 邻居索引 (B, N, K)
                neigh_ped_mask: 邻居有效性掩码 (B, N, K)
                collision_threshold: 碰撞距离阈值 (float)
            Returns:
                v_next: 修正后的速度 (B, N, 2)
            """
        p_cur = torch.nan_to_num(p_cur, nan=0.0)
        B, N, K = near_ped_idx.shape
        p_for_gather = p_cur.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 2)
        v_for_gather = v_cur.unsqueeze(1).expand(-1, N, -1, -1)
        idx = near_ped_idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B, N, K, 2)

        neighbors_pos = torch.gather(p_for_gather, 2, idx)  # (B, N, K, 2)
        neighbors_vel = torch.gather(v_for_gather, 2, idx)

        #只在有效邻居
        mask = neigh_ped_mask.unsqueeze(-1)  # (B, N, K, 1)
        valid_neighbors_pos = neighbors_pos * mask
        valid_neighbors_vel = neighbors_vel * mask

        rel_pos = p_cur.unsqueeze(2) - valid_neighbors_pos  # (B, N, K, 2)
        rel_vel = v_desire.unsqueeze(2) - valid_neighbors_vel

        # 3. 时间最小化计算 t_min
        dot_pos_vel = torch.sum(rel_pos * rel_vel, dim=-1)  # (B, N, K)
        dot_vel_vel = torch.sum(rel_vel * rel_vel, dim=-1) + 1e-6  # 防止除0
        t_min = dot_pos_vel / dot_vel_vel
        t_min = torch.clamp(t_min, 0, self.tau)  # (B, N, K)

        # 4. 预测最小距离
        extrapolated = rel_pos + t_min.unsqueeze(-1) * rel_vel  # (B, N, K, 2)
        min_dist = torch.norm(extrapolated, dim=-1)  # (B, N, K)
        min_dist = torch.nan_to_num(min_dist, nan=1e6)

        # 5. 碰撞检测掩码
        collision_mask = (min_dist < collision_threshold) & neigh_ped_mask.bool()

        # 6. 计算修正方向（法向量，rel_pos旋转90度）
        normal = torch.stack([-rel_pos[..., 1], rel_pos[..., 0]], dim=-1)  # (B, N, K, 2)
        norm = torch.norm(normal, dim=-1, keepdim=True)
        normal_unit = torch.where(collision_mask.unsqueeze(-1),
                                  normal / (norm + 1e-6),
                                  torch.zeros_like(normal))

        # 7. 累计修正向量 (B, N, 2)
        correction = torch.sum(normal_unit, dim=2)
        correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)

        # 8. 应用修正并返回
        v_next = v_desire + correction * self.fix

        return v_next