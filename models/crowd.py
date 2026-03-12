import torch
import torch.nn as nn
import torch.nn.functional as F

class Crowd_Fusion(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        if num_heads > 1:
            self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # 加入 Dropout（新增）
        self.dropout = nn.Dropout(p=dropout)

        # 加入门控机制（新增）
        self.gate_layer = nn.Linear(hidden_dim, hidden_dim)
            # 残差 + 归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, H_hisr, c_curr):
        """
        Args:
              H_hisr: 历史状态， B,N,D
              c_curr: 当前状态: B,N,D
        returns:
            h_fused:融合后 B,N, D
        """
        B, N, D = H_hisr.shape

        query = self.query(c_curr)
        key = self.key(H_hisr)
        value = self.value(H_hisr)

        #计算权重
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.hidden_dim ** 0.5)
        alpha = F.softmax(scores, dim=-1)

        h_attn = torch.matmul(alpha, value)  # [B, N, D]

        # Dropout（新增）
        h_attn = self.dropout(h_attn)

        # 门控残差融合（新增）
        gate = torch.sigmoid(self.gate_layer(c_curr))  # [B, N, D]
        output = gate * h_attn + (1 - gate) * c_curr  # 融合当前特征和注意力特征

        return output


class WeightedGraphLayer2(nn.Module):
    def __init__(self, in_dim=64, out_dim=64,crowd_dim =5, sim=0):
        super(WeightedGraphLayer2, self).__init__()
        a = 6
        self.hist_decay = 0.8
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim+a, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        # node update MLP 输入：h_i + agg + crowd_feature
        self.node_update = nn.Sequential(
            nn.Linear(in_dim*2+crowd_dim, out_dim),
            nn.ReLU()
        )
        self.norm_crowd = torch.nn.LayerNorm(crowd_dim)

        self.sim = sim
        self.sigma = 1.0

    def safe_norm(self, x, dim=-1, keepdim=True, eps=1e-6):
        return torch.clamp(torch.norm(x, dim=dim, keepdim=keepdim), min=eps)
    def forward(self, h, pos, vel,acc,crowd, mask, idex, hist_feature):
        """
        h:     [B, N, d]       - 节点特征
        pos:   [B, N, 2]       - 行人位置
        vel:   [B, N, 2]       - 行人速度（方向）
        mask:  [B, N, N]       - 邻接矩阵掩码
        """
        B, N, d = h.shape
        K = mask.shape[-1]
        H = hist_feature.shape[-2]
        m = idex * mask # B,N,K
        device = h.device


        pos_expand = pos.unsqueeze(-2).expand(-1,-1,K,-1) # B,N，K 2
        vel_expand = vel.unsqueeze(-2).expand(-1,-1,K,-1)
        acc_expand = acc.unsqueeze(-2).expand(-1,-1,K,-1)
        idx_expanded = m.long().unsqueeze(-1)#B N K 1
        idx_ex = m.long().unsqueeze(-1).expand(-1,-1,-1,h.shape[-1])#B N K 1
        neigh_pos = torch.gather(pos_expand, dim=1, index=idx_expanded.expand(-1, -1, -1, 2))#B n K 2
        neigh_vel = torch.gather(vel_expand, dim=1, index=idx_expanded.expand(-1, -1, -1, 2))
        neigh_acc = torch.gather(acc_expand, dim=1, index=idx_expanded.expand(-1, -1, -1, 2))

        h_neigh = torch.take_along_dim(h.unsqueeze(1).expand(-1, N, -1, -1), idx_ex, dim=2)

        rel_pos = neigh_pos - pos_expand
        distance = torch.norm(rel_pos, dim=-1) + 1e-6 # B N K


        rel_speed = torch.norm(vel.unsqueeze(2) - neigh_vel, dim=-1, keepdim=True)

        # == crowd sim ==
        ped_motion = torch.cat([vel, acc],dim=-1)# BN4
        ped_norm = torch.norm(ped_motion, dim=-1,keepdim=True)#B N
        crowd_motion = crowd[...,:4]
        crowd_norm = torch.norm(crowd_motion, dim=-1,keepdim=True)

        dot = (ped_motion * crowd_motion).sum(dim=-1, keepdim=True)

        cos_sim = dot / (ped_norm * crowd_norm + 1e-6)
        crowd_sim = (cos_sim + 1) / 2.0
        crowd_sim = crowd_sim.unsqueeze(-2).expand(-1, -1, K, -1)

        hist_i = hist_feature.unsqueeze(2).expand(-1, -1, K, -1, -1)  # [B,N,K,H,6]
        idx_exp_hist = idex.long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, H, hist_feature.shape[-1])
        hist_j = torch.take_along_dim(hist_feature.unsqueeze(2).expand(-1, -1, K, -1, -1),
                                      idx_exp_hist, dim=1)  # [B,N,K,H,6]

        weights = torch.tensor([0.1, 0.1, 1.0, 1.0, 0.5, 0.5], device=device)  # 可调
        diff = (hist_i - hist_j) * weights
        dist = diff.norm(dim=-1)  # [B,N,K,H]
        sim_t = torch.exp(-dist / self.sigma)  # 越近越相似
        lambda_decay = self.hist_decay
        w = torch.pow(torch.tensor(lambda_decay, device=device),
                      torch.arange(H - 1, -1, -1, device=device))
        w = (w / (w.sum() + 1e-6)).view(1, 1, 1, H)

        hist_sim = (sim_t * w).sum(-1, keepdim=True)  # [B,N,K,1]
        hist_sim = hist_sim * 0.1

        edge_input = torch.cat([h_neigh, rel_pos, distance.unsqueeze(-1),  crowd_sim, hist_sim, rel_speed], dim=-1)

        edge_feat = self.edge_mlp(edge_input) * mask.unsqueeze(-1) #
        agg = edge_feat.sum(2) / (mask.sum(2, keepdim=True) + 1e-6)
        node_input = [h, agg]
        crowd1 = self.norm_crowd(crowd)
        node_input.append(crowd1)
        node_input = torch.cat(node_input, dim=-1)
        return self.node_update(node_input)


class WeightedGraphLayer(nn.Module):
    def __init__(self, in_dim=64, out_dim=64,crowd_dim =5, sim=0):
        super(WeightedGraphLayer, self).__init__()
        a = 7
        if sim==12:
            a = a-1
        if sim==1:
            a = a-2
        if sim==0:
            a = a-3
        if sim==3:
            a = a-2
        if sim==13:
            a = a-1
        if sim==123:
            a = a
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim+a, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

        # node update MLP 输入：h_i + agg + crowd_feature
        self.node_update = nn.Sequential(
            nn.Linear(in_dim*2+crowd_dim, out_dim),
            nn.ReLU()
        )



        self.norm_crowd = torch.nn.LayerNorm(crowd_dim)


        self.sim = sim


    def forward(self, h, pos, vel,acc,crowd, mask,idex, hist):
        """
        h:     [B, N, d]       - 节点特征
        pos:   [B, N, 2]       - 行人位置
        vel:   [B, N, 2]       - 行人速度（方向）
        mask:  [B, N, N]       - 邻接矩阵掩码
        """
        B, N, d = h.shape
        K = mask.shape[-1]

        m = idex * mask # B,N,K

        pos_expand = pos.unsqueeze(-2).expand(-1,-1,K,-1) # B,N，K 2
        vel_expand = vel.unsqueeze(-2).expand(-1,-1,K,-1)
        acc_expand = acc.unsqueeze(-2).expand(-1,-1,K,-1)
        idx_expanded = m.long().unsqueeze(-1)#B N K 1
        idx_ex = m.long().unsqueeze(-1).expand(-1,-1,-1,h.shape[-1])#B N K 1
        neigh_pos = torch.gather(pos_expand, dim=1, index=idx_expanded.expand(-1, -1, -1, 2))#B n K 2
        neigh_vel = torch.gather(vel_expand, dim=1, index=idx_expanded.expand(-1, -1, -1, 2))
        neigh_acc = torch.gather(acc_expand, dim=1, index=idx_expanded.expand(-1, -1, -1, 2))

        # h_expand = h.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, d]
        # h_neigh = torch.gather(h_expand, dim=2, index=idx_ex)  # [B, N, K, d]
        h_neigh = torch.take_along_dim(h.unsqueeze(1).expand(-1, N, -1, -1), idx_ex, dim=2)

        rel_pos = neigh_pos - pos_expand
        distance = torch.norm(rel_pos, dim=-1) + 1e-6 # B N K

        #邻居朝向行人走动
        rel_dir = F.normalize(rel_pos, dim=-1)
        vj_dir = F.normalize(neigh_vel, dim=-1)
        flow_sim1 = ((rel_dir * vj_dir).sum(-1,keepdim=True) +1)/2.0

        #与邻居同行
        vi_dir = F.normalize(vel, dim=-1)
        vi_dir = vi_dir.unsqueeze(2).expand_as(vj_dir)
        flow_sim2 = ((vi_dir * vj_dir).sum(-1, keepdim=True) + 1) / 2.0

        rel_speed = torch.norm(vel.unsqueeze(2) - neigh_vel, dim=-1, keepdim=True)

        # == crowd sim ==
        ped_motion = torch.cat([vel, acc],dim=-1)# BN4
        ped_norm = torch.norm(ped_motion, dim=-1,keepdim=True)#B N
        crowd_motion = crowd[...,:4]
        crowd_norm = torch.norm(crowd_motion, dim=-1,keepdim=True)

        dot = (ped_motion * crowd_motion).sum(dim=-1, keepdim=True)

        cos_sim = dot / (ped_norm * crowd_norm + 1e-6)
        crowd_sim = (cos_sim + 1) / 2.0
        crowd_sim = crowd_sim.unsqueeze(-2).expand(-1, -1, K, -1)
        if self.sim ==0:
            edge_input = torch.cat(
                [h_neigh, rel_pos, distance.unsqueeze(-1),rel_speed], dim=-1)
        if self.sim ==1:
            edge_input = torch.cat(
                [h_neigh, rel_pos, distance.unsqueeze(-1), flow_sim1, rel_speed], dim=-1)
        if self.sim==12:
            edge_input = torch.cat(
                [h_neigh, rel_pos, distance.unsqueeze(-1), flow_sim1, flow_sim2, rel_speed], dim=-1)
        if self.sim==3:
            edge_input = torch.cat([h_neigh, rel_pos, distance.unsqueeze(-1), crowd_sim, rel_speed],dim=-1)
        if self.sim==13:
            edge_input = torch.cat([h_neigh, rel_pos, distance.unsqueeze(-1), flow_sim1,crowd_sim, rel_speed],dim=-1)
        if self.sim ==123:
            edge_input = torch.cat([h_neigh, rel_pos, distance.unsqueeze(-1), flow_sim1,flow_sim2, crowd_sim, rel_speed], dim=-1)

        edge_feat = self.edge_mlp(edge_input) * mask.unsqueeze(-1) #
        agg = edge_feat.sum(2) / (mask.sum(2, keepdim=True) + 1e-6)
        node_input = [h, agg]
        crowd1 = self.norm_crowd(crowd)
        node_input.append(crowd1)
        node_input = torch.cat(node_input, dim=-1)
        return self.node_update(node_input)







class PedestrianInteractionModule(nn.Module):
    def __init__(self, in_feat_dim, time_dim, noise_dim, out_feat_dim ,hidden_dim=64,device=None,num_layers=3,crowd_dim=5,sim=0, type_num=1):
        super(PedestrianInteractionModule, self).__init__()
        self.in_dim = in_feat_dim + time_dim + noise_dim +crowd_dim
        self.out_dim = out_feat_dim
        if type_num==1:
            self.layers = nn.ModuleList([
                WeightedGraphLayer(in_dim=hidden_dim, out_dim=hidden_dim,sim=sim) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                WeightedGraphLayer2(in_dim=hidden_dim, out_dim=hidden_dim, sim=sim) for _ in range(num_layers)
            ])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.input_encoder= nn.Linear(self.in_dim, hidden_dim,device=device)

    def forward(self, ped_feat,  neigh_ped_mask,neigh_idx, time_emb, noise,crowd_feature, hist):
        """
        ped_feat:      [B, N, 6]
        neigh_ped_mask:[B, N, 6]  (0-1 mask)
        time_emb:      [B, N, 3]
        noise:         [B, N, 2]
        """
        B, N, _ = ped_feat.shape
        ped_feat = torch.nan_to_num(ped_feat, nan=0.0)
        # invalid = torch.isnan(ped_feat).any(dim=-1) # 原始 NaN 位置仍能被检测出来
        # neigh_ped_mask = neigh_ped_mask.masked_fill(
        #     invalid.unsqueeze(1) | invalid.unsqueeze(2), 0
        # )
        # 加入时间和噪声特征
        h = torch.cat([ped_feat, crowd_feature,time_emb, noise], dim=-1)  # [B, N, 16]
        p = ped_feat[...,:2]
        v = ped_feat[...,2:4]
        a = ped_feat[...,4:6]

        h = self.input_encoder(h)

        for layer in self.layers:
            h = layer(h, p, v,a,crowd_feature, neigh_ped_mask,neigh_idx,hist)



        return h