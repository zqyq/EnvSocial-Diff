import torch
import torch.nn.functional as F
from IPython.core.release import description
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb
# from utils.TrajectoryDS import anorm
from utils.utils import mask_mse_func
from tqdm.auto import tqdm
from .egnn import *
from .vnn_models import *
from .crowd import *
from .bright import  *
from .image import *
from  .PositionAware import *
from .Fusion import *
from .Text import *
from .bright2 import *
from .interest import *
from .Env import *
from .obs import *


class VarianceSchedule(Module):
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i] # posterior variance
        sigmas_inflex = torch.sqrt(sigmas_inflex)
        posterior_mean_coef1 = torch.zeros_like(sigmas_flex)
        posterior_mean_coef2 = torch.zeros_like(sigmas_flex)
        for i in range(1, posterior_mean_coef1.size(0)):
            posterior_mean_coef1[i] = torch.sqrt(alpha_bars[i-1]) * betas[i] / (1 - alpha_bars[i])
        posterior_mean_coef1[0] = 1.0
        for i in range(1, posterior_mean_coef2.size(0)):
            posterior_mean_coef2[i] = torch.sqrt(alphas[i]) * (1 - alpha_bars[i-1]) / (1 - alpha_bars[i])

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex) 
        self.register_buffer('sigmas_inflex', sigmas_inflex)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

class TrajNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(2, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 2, context_dim+3),

        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class Diffuser_ped_inter_geometric_cond_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'ucy' in config.data_dict_path:
            self.tau = 5 / 6
        else:
            self.tau = 2
        config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        self.egnn = NetEGNN_hid2(hid_dim=config.egnn_hid_dim, n_layers=config.egnn_layers)

        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)

        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        self.env_emb_dim = 0
        # if config.has_obstacles == True:
        #     self.egnn_obs = NetEGNN_hid_obs2(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
        #     self.env_emb_dim = self.env_emb_dim + config.egnn_hid_dim_obs
        if config.use_bright:
            self.bright_enco = BrightnessEncoder(in_dim=5, hidden_dim=config.bright_hid, out_dim=config.bright_out)
            self.env_emb_dim = self.env_emb_dim + config.bright_out
            self.relu3 = nn.ReLU()
            self.decode_env = nn.Linear(self.env_emb_dim, 2)
        self.concat_ped1 = nn.Linear(config.spatial_emsize + config.context_dim + 3, config.spatial_emsize // 2)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize // 2, 2)


    def forward(self, x, beta, context: tuple, nei_list, t):
        # context encoding
        hist_feature = context[4]
        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature)))  # bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0, end_dim=1)  # bs*N,his_len,embsize
        _, (hist_embedded, _) = self.history_LSTM(hist_embedded)  # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2], -1)  # bs,N,embsize # TODO**
        hist_embedded = self.lstm_output(hist_embedded)  # bs,N,embsize_out
        bright_features = context[11]
        self_features = context[2]
        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_  # des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1, x.shape[-2], 1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)

        acce_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)

        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))

        context_emb = torch.cat((acce_emb, hist_embedded), dim=-1)

        spatial_input_embedded = self.concat_ped1(torch.cat((context_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))

        if self.config.use_bright:
            bright_emb = self.bright_enco(bright_features)
            # env_emb = torch.cat((obs_emb, bright_emb), dim=-1)
            env_acc = self.decode_env(self.relu3(bright_emb))
            output_ped = output_ped + env_acc

        return output_ped + pred_acc_dest
    
class Diffuser_ped_inter_geometric_cond_w_obs_w_history(Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2

        # config.context_dim = config.ped_encode_dim2 + config.history_lstm_out
        # config.context_dim = 2
        if config.use_base:
            config.context_dim = config.egnn_hid_dim + config.history_lstm_out
        else:
            if config.use_crowd_lstm:
                config.context_dim = config.egnn_hid_dim + config.history_lstm_out + self.config.crowd_emsize
            else:
                config.context_dim = config.history_lstm_out


        if config.use_obs_json:
            self.env_dim = 0

            if config.use_obs_json:
                #版本1
                self.env_dim += config.egnn_hid_dim
                # if config.obs_num==1:
                self.image_text_fusion = LocalFusion(device=config.device, text_dim=config.text_dim, image_dim=config.image_dim,out_dim=config.out_dim, dropout=config.dropout)
                self.image_text_fusion = self.image_text_fusion.to(config.device)
                # self.position_aware = PositionAwareEnvAttention(device=config.device,person_dim=6, env_dim=config.out_dim,out_dim=64,hidden_dim=config.hidden_dim, config=config)
                self.position_aware = PositionAwareEnvAttention(device=config.device,person_dim=6, env_dim=config.out_dim,out_dim=config.egnn_hid_dim,hidden_dim=config.hidden_dim, config=config)
                self.position_aware = self.position_aware.to(config.device)
                self.big_fusion =  GlobalFusion(device=config.device, text_dim=config.text_dim, image_dim=config.image_dim, dropout=config.dropout)
                self.big_fusion = self.big_fusion.to(config.device)
                # # self.env_dim += 64
                # else:
                #     self.obs_aware = Obs_Aware(img_feat_dim=config.image_dim, text_feat_dim=config.text_dim, hidden_dim=config.hidden_dim, out_dim=config.egnn_hid_dim_obs)


                #
                #old
                # self.egnn_obs_json = NetEGNN_json_obs(hid_dim=config.egnn_hid_dim_obs, n_layers=config.egnn_layers, obs_dim=64).to(device=config.device)
                # self.egnn_obs_json = NetEGNN_json_obs(hid_dim=config.egnn_hid_dim_obs, n_layers=config.egnn_layers, obs_dim=config.out_dim_json).to(device=config.device)
            if config.use_in_json:
                self.env_dim += 64
                # config.context_dim = config.egnn_hid_dim_obs + config.context_dim
                self.in_aware = Intest_Aware(img_feat_dim=config.image_dim, text_feat_dim=config.text_dim, hidden_dim=config.hidden_dim, out_dim=64,config=config)
            if config.use_bright:
                # self.env_dim += 64
                config.context_dim +=config.bright_out
                # self.bright_enco = BrightnessEncoder(in_dim=5,hidden_dim=config.hidden_dim, out_dim=64)
                self.bright_enco = BrightnessEncoder(in_dim=5,hidden_dim=config.bright_hid, out_dim=config.bright_out)
                # config.context_dim = config.context_dim + config.bright_out
                # self.egnn_bri2 = NetEGNN_bright2(hidden_dim=config.egnn_hid_dim, config=config).to(device=config.device)
            if config.use_env:
                self.env_ecoder = PedEnvGNN(ped_dim=6,env_dim=self.env_dim, hidden_dim=config.hidden_dim, num_layers=config.egnn_layers_obs, out_dim=config.egnn_hid_dim)
                # self.env_ecoder = PedEnvGNN_v2(ped_dim=6,env_dim=self.env_dim, hidden_dim=config.hidden_dim, num_layers=config.egnn_layers_obs, out_dim=config.egnn_hid_dim)
                config.context_dim += config.egnn_hid_dim
            else:
                config.context_dim += self.env_dim

        # if config.use_bright2:
        #     self.bright2 = SceneBrightnessAwareContext(scene_feat_dim=config.image_dim,hidden_dim=config.bright_hidden_dim, device=config.device,config=config)
        #     config.context_dim = config.context_dim + config.bright_hidden_dim
        #     self.egnn_bri = NetEGNN_bright(hid_dim=config.bright_hidden_dim, n_layers=config.egnn_layers,bri_dim=config.bright_hidden_dim )



        # self.egnn = NetEGNN_hid2(hid_dim = config.egnn_hid_dim, n_layers = config.egnn_layers)

        # self.egnn2 = NetEGNN_hid3(hid_dim=config.egnn_hid_dim, n_layers=config.egnn_layers)  # 64 3
        if config.use_ped:
            self.ped_interation = PedestrianInteractionModule(in_feat_dim=6,out_feat_dim=config.egnn_hid_dim,hidden_dim=config.egnn_hid_dim, time_dim=3, noise_dim=2,num_layers=config.egnn_layers, device=config.device, sim=config.sim, type_num=config.ped_num)
            config.context_dim += config.egnn_hid_dim




        # self.egnn3 = NetEGNN_hid4(hid_dim= config.egnn_hid_dim, n_layers=config.egnn_layers)

        self.has_obstacles = config.has_obstacles
        if config.has_obstacles:
            self.egnn_obs = NetEGNN_hid_obs2(hid_dim = config.egnn_hid_dim_obs, n_layers = config.egnn_layers_obs)
            config.context_dim = config.egnn_hid_dim_obs + config.context_dim

        self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        # self.history_encoder2 = nn.Linear(11, config.history_emsize)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        # self.history_encoder = nn.Linear(config.history_dim, config.history_emsize)
        self.history_LSTM = nn.LSTM(config.history_emsize, config.history_lstm, 1, batch_first=True)
        self.lstm_output = nn.Linear(config.history_lstm, config.history_lstm_out)
        if config.use_crowd_lstm:

            self.crowd_encoder = nn.Linear(config.crowd_dim, config.crowd_emsize)
            self.relu_crowd = nn.ReLU()
            self.dropout_crowd = nn.Dropout(config.dropout)
            self.history_crowd_LSTM = nn.LSTM(config.crowd_emsize, config.history_crowd_lstm, 1, batch_first=True)
            self.crowd_output = nn.Linear(config.history_crowd_lstm, config.history_crowd_lstm_out)
            #
        if config.use_crowd_fusion:
            self.crowd_curr_encoder = nn.Linear(config.crowd_dim, config.crowd_emsize,)
            self.relu_crowd_curr = nn.ReLU()
            self.dropout_crowd_curr = nn.Dropout(config.dropout)

            self.crowd_fusion = Crowd_Fusion(config.crowd_emsize, num_heads=4,dropout=config.dropout)


        self.input_embedding_layer_spatial = nn.Linear(2, config.spatial_emsize)
        self.relu = nn.ReLU()
        self.dropout_in = nn.Dropout(config.dropout)
        # self.concat2 = AdaptiveFusion(config.spatial_emsize, config.spatial_emsize, config.context_dim)
        # self.k_emb = lambda x:timestep_embedding(x, dim=config.kenc_dim)



        self.concat_ped1 = nn.Linear(config.spatial_emsize + config.context_dim + 3, config.spatial_emsize // 2)
        self.relu2 = nn.ReLU()
        self.decode_ped1 = nn.Linear(config.spatial_emsize // 2, 2)

        self.concat_ped2 = nn.Linear(
            config.spatial_emsize + config.context_dim + 3, config.spatial_emsize // 2)
        self.relu3 = nn.ReLU()
        self.decode_ped2 = nn.Linear(config.spatial_emsize // 2, 2)


        
    def forward(self, x, beta, context:tuple, nei_list,t):
        # context encoding
        #context[0] curr p,v,a
        crowd_hist_features = context[10]
        hist_feature = context[4]

        hist_embedded = self.dropout1(self.relu1(self.history_encoder(hist_feature))) #bs,N,his_len,embsize
        # hist_embedded = self.dropout1(self.relu1(self.history_encoder2(torch.cat([hist_feature,crowd_hist_features],dim=-1)))) #bs,N,his_len,embsize
        origin_shape = hist_embedded.shape
        hist_embedded = hist_embedded.flatten(start_dim=0,end_dim=1) #bs*N,his_len,embsize
        _, (hist_embedded,_) = self.history_LSTM(hist_embedded) # 1, bs*N, lstm hidden size
        hist_embedded = hist_embedded.squeeze().view(*origin_shape[:2],-1) # bs,N,embsize # TODO**
        hist_embedded = self.lstm_output(hist_embedded)
        
        self_features = context[2]
        crowd_feature = context[9]
        bright_features = context[11]

        H = context[18]


        if self.config.use_crowd_lstm:
            crowd_embedded = self.dropout_crowd(self.relu_crowd(self.crowd_encoder(crowd_hist_features)))
            origin_crowd_shape = crowd_embedded.shape
            crowd_embedded = crowd_embedded.flatten(start_dim=0, end_dim=1)
            _, (crowd_embedded, _) = self.history_crowd_LSTM(crowd_embedded)
            crowd_embedded = crowd_embedded.squeeze().view(*origin_crowd_shape[:2], -1)
            crowd_embedded = self.crowd_output(crowd_embedded) #32 144 64
        if self.config.use_crowd_fusion:
            crowd_curr_embedded = self.dropout_crowd_curr(self.relu_crowd_curr(self.crowd_curr_encoder(crowd_feature)))
            crowd_curr_embedded_shape = crowd_curr_embedded.shape

            crowd_fusion_embedded = self.crowd_fusion(crowd_embedded, crowd_curr_embedded)

        desired_speed = self_features[..., -1].unsqueeze(-1)
        temp = torch.norm(self_features[..., :2], p=2, dim=-1, keepdim=True)
        temp_ = temp.clone()
        temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
        dest_direction = self_features[..., :2] / temp_ #des,direction
        pred_acc_dest = (desired_speed * dest_direction - self_features[..., 2:4]) / self.tau

        ped_features = context[0]
        neigh_ped_mask = context[1]
        near_ped_idx = context[3]
        beta = beta.view(x.shape[0], 1, 1).repeat([1,x.shape[-2],1])
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        # ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        # ped_emb = self.egnn([ped_features, neigh_ped_mask, near_ped_idx], time_emb)
        if self.config.use_ped:
            ped_emb = self.ped_interation(ped_features, neigh_ped_mask,near_ped_idx, time_emb,x,crowd_feature, hist_feature)
            ctx_emb = torch.cat((hist_embedded, ped_emb), dim=-1)
        else:
            ctx_emb = hist_embedded

        mask = context[21]
        if self.config.use_json:
            if self.config.use_obs_json:
                image_big = context[19]
                text_big = context[20]
                if self.config.use_obs_json:

                    text_obs_emb = context[12]
                    image_obs_emb  = context[14]
                    box1 = context[16]
                    if self.config.obs_num == 1:
                    #版本1
                        # image_fusion = self.big_fusion(text_big, image_big)
                        env_fusion = self.image_text_fusion(text_emb=text_obs_emb, image_emb=image_obs_emb)

                        # env_emb = self.position_aware(context[0],neigh_ped_mask, env_fusion,box1, H,image_big,text_big,self.config.label, self.config.bg_image,self.config)
                        # obs_emb = self.egnn_obs_json([ped_features, env_emb], time_emb)
                        # ctx_emb = torch.cat((ctx_emb, obs_emb),dim=-1)
                        # obs_emb = self.position_aware(context[0], env_fusion,box1, H,image_big,text_big,self.config.label, self.config.bg_image,self.config)

                        obs_emb = self.position_aware(context[0], env_fusion,box1, H,image_big,text_big,self.config.label, self.config.bg_image,self.config)

                    #版本2
                    else:
                        obs_emb = self.obs_aware(image_obs_emb,text_obs_emb, box1,
                                                 context[0],
                                                 image_big,text_big,
                                                 H)
                    env = torch.cat([obs_emb], dim=-1)
            if self.config.use_in_json:
                text_in_emb = context[13]
                image_in_emb = context[15]
                box2 = context[17]
                in_emb = self.in_aware(image_in_emb, text_in_emb, box2,
                                       context[0],
                                       image_big,text_big,
                                       H,self.config.label)
                # ctx_emb = torch.cat((ctx_emb, in_emb),dim=-1)
                env = torch.cat((env,in_emb), dim=-1)

            if self.config.use_bright:
                bright_emb = self.bright_enco(bright_features)
                # mask_p = context[18]
                # bright_emb = self.egnn_bri2([ped_features, bright_features], time_emb, H,mask_p)
                # bright_emb = torch.nan_to_num(bright_emb, nan=0.0)
                # ctx_emb = torch.cat((ctx_emb, bright_emb), dim=-1)
                ctx_emb = torch.cat([ctx_emb,bright_emb], dim=-1)
            if self.config.use_env:
                env_emb = self.env_ecoder(ped_features, env)
                ctx_emb = torch.cat((ctx_emb, env_emb), dim=-1)




        if self.config.use_crowd_lstm:
            # ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb, crowd_embedded), dim=-1)

            # ctx_emb = torch.cat((ped_emb, obs_emb, crowd_fusion_embedded, hist_embedded), dim=-1)
            # ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb, crowd_fusion_embedded), dim=-1)
            if self.config.use_crowd_fusion:
                ctx_emb = torch.cat((ctx_emb, crowd_fusion_embedded), dim=-1)
            else:
                ctx_emb = torch.cat(( ctx_emb, crowd_embedded,), dim=-1)


        # if self.config.use_bright2:
        #     bright2 = self.bright2(ped_features, context[16])
        #     bright_emb = self.egnn_bri([ped_features, bright2], time_emb)
        #     ctx_emb = torch.cat((ctx_emb, bright_emb), dim=-1)
        #     # ctx_emb = torch.cat((ctx_emb, bright2), dim=-1)

        # if self.has_obstacles:
        #   obs_features = context[5] #B,M,2
        #   near_obstacle_idx = context[6]
        #   neigh_obs_mask = context[7]
        #   obs_emb = self.egnn_obs([ped_features, obs_features, neigh_obs_mask, near_obstacle_idx], time_emb)
        #   if self.config.use_crowd_lstm:
        #       if self.config.use_base:
        #           ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)
        #       else:
        #           ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb, crowd_embedded), dim=-1)
        #           # ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb, crowd_fusion_embedded), dim=-1)
        #   else:
        #       ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb), dim=-1)
        #       # ctx_emb = torch.cat((hist_embedded, ped_emb, obs_emb, crowd_fusion_embedded), dim=-1)
        # else:
        #   # ctx_emb = torch.cat((hist_embedded, ped_emb, crowd_fusion_embedded), dim=-1)
        #   ctx_emb = torch.cat((hist_embedded, ped_emb, crowd_embedded), dim=-1)





        spatial_input_embedded = self.dropout_in(self.relu(self.input_embedding_layer_spatial(x)))
        
        
        spatial_input_embedded = self.concat_ped1(torch.cat((ctx_emb, spatial_input_embedded, time_emb), dim=-1))
        output_ped = self.decode_ped1(self.relu2(spatial_input_embedded))
        return output_ped + pred_acc_dest
        
        # return output_ped + pred_acc_dest





class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule, config):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.config = config
    def denoise_fn(self, x_0, curr=None, context=None, timestep=0.08, t=None, mask=None):
        batch_size = x_0.shape[0]
        # point_dim = x_0.shape[-1]
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1)

        

        e_rand = torch.randn_like(x_0).cuda()  # (B, N, d)******T
        x_t = c0 * x_0 + c1 * e_rand
        nei_list = None
        p0 = curr[...,:2] # B, N, 2

        if self.config.nei_padding_mask==True:
            nei_list = self.curr_padding(p0)

        x_0_hat = self.net(x_t, beta=beta, context=context, nei_list = nei_list, t=t)
        return x_0_hat
        
    def get_loss(self, x_0, curr=None, context=None, timestep=0.08, t=None, mask=None):

        # batch_size = x_0.shape[0]
        point_dim = x_0.shape[-1]
        x_0_hat = self.denoise_fn(x_0, curr=curr, context=context, timestep=timestep, t=t, mask=mask)

        if mask==None:
            loss = F.mse_loss(x_0_hat.view(-1,point_dim), x_0.continuous().view(-1,point_dim), reduction='mean')
        else:
            loss = mask_mse_func(x_0_hat, x_0, mask)       
        return loss
    
    def cur_mask(self, p0):
        assert p0.dim()==3
        attn_mask = torch.tensor((),device=p0.device)
        for i in range(p0.shape[0]):
            isnan = torch.isnan(p0[i,:,0]) # N
            index = torch.where(~isnan) 
            #**y
            y = torch.zeros((p0.shape[1], p0.shape[1]), device=p0.device) # N, N
            index = torch.meshgrid(index[0],index[0])
            y[index[0], index[1]] = 1
            y = y.unsqueeze(0) # 1, N, N
            attn_mask = torch.cat((attn_mask,y), dim=0)
        return attn_mask
        
    def curr_padding(self, p0):
        return p0[...,0].isnan()

    def sample(self, context, curr = None, bestof=True, point_dim=2, timestep=0.08, ret_traj=False):
        """_summary_

        Args:
            num_points (_type_): _description_
            context (_type_): [bs(sample), obs_len+1(destination),N,2]
            sample (_type_): _description_
            bestof (_type_): _description_
            point_dim (int, optional): _description_. Defaults to 2.
            flexibility (float, optional): _description_. Defaults to 0.0.
            ret_traj (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        traj_list = []
        history = context[0]
        ped_features = context[1]
        assert context[0].shape[0]==context[1].shape[0]
        if curr!=None:
            assert len(curr.shape)==3 # t, n, 6
        batch_size = history.shape[0]
        num_ped = context[1].shape[1]
        if bestof:
            x_T = torch.randn([batch_size, num_ped, point_dim]).to('cuda')
        else:
            x_T = torch.zeros([batch_size, num_ped, point_dim]).to('cuda')
        traj = {self.var_sched.num_steps: x_T}
        pbar = range(self.var_sched.num_steps, 0, -1)
        nei_list = None
        p0 = curr[...,:2] # B, N, 2

        if self.config.nei_padding_mask==True:
            nei_list = self.curr_padding(p0)
        for t in pbar:
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha_bar_ = self.var_sched.alpha_bars[t-1]

            # c0 = torch.sqrt(alpha_bar_).view(-1, 1, 1).cuda()       # (B, 1, 1)
            # c1 = torch.sqrt(1 - alpha_bar_).view(-1, 1, 1).cuda()

            x_t = traj[t]
            beta = self.var_sched.betas[[t]*batch_size]
            

            x_0_hat = self.net(x_t, beta=beta, context=context ,nei_list=nei_list, t=t)
            mean, var = self.p_mean_variance(x_0_hat,x_t,t)
            x_next = mean + torch.sqrt(var)*z
            assert x_next.shape == x_t.shape
            traj[t-1] = x_next
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            

        if ret_traj:
            traj_list.append(traj)
        else:
            traj_list.append(traj[0])
        
        return torch.stack(traj_list).squeeze()

    def p_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self.var_sched.posterior_mean_coef1[t] * x_start +
            self.var_sched.posterior_mean_coef2[t]  * x_t
        )
        posterior_variance = self.var_sched.sigmas_inflex[t].view(x_start.shape[0],*[1]*(x_start.ndim-1))
        
        assert (posterior_mean.shape[0] == posterior_variance.shape[0]  ==
                x_start.shape[0])
        return posterior_mean, posterior_variance
