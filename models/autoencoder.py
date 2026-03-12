import torch
from IPython.core.release import description
from torch.nn import Module
import torch.nn as nn
# from .encoders.trajectron import Trajectron
# from .encoders import dynamics as dynamic_module
import models.diffusion as diffusion
from models.RVO import RVOModule
from models.diffusion import DiffusionTraj,VarianceSchedule
import pdb
import numpy as np
import data.data as DATA
from tqdm.auto import tqdm
import torch.nn.functional as F
from .Text import *
from .image import *

def clear_nan(tensor:torch.Tensor):
    tensor[tensor.isnan()]=0
    return tensor

class AutoEncoder(Module, DATA.Pedestrians):

    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = config
        self.device = config.device
        self.encoder = encoder # 
        self.diffnet = getattr(diffusion, config.diffnet)  #Diffuser_ped_inter_geometric_cond_w_history

        self.diffusion = DiffusionTraj( 
            # net = self.diffnet(point_dim=2, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False),
            net = self.diffnet(config),
            var_sched = VarianceSchedule(
                num_steps=config.diffusion_steps,
                beta_T=5e-2,
                mode=config.variance_mode #'linear', 'cosine'

            ),
            config=config
        )
        if 'ucy' in config.data_dict_path:
            self.tau=5/6
        else:
            self.tau=2
        self.RVO = RVOModule(tau=0.24, fix=self.config.fix)

    def encode(self, batch,node_type):
        z = self.encoder.get_latent(batch, node_type)
        return z

    def generate(self, batch, node_type, num_points, sample, bestof,flexibility=0.0, ret_traj=False):

        dynamics = self.encoder.node_models_dict[node_type].dynamic
        encoded_x = self.encoder.get_latent(batch, node_type)
        predicted_y_vel =  self.diffusion.sample(num_points, encoded_x,sample,bestof, flexibility=flexibility, ret_traj=ret_traj)
        predicted_y_pos = dynamics.integrate_samples(predicted_y_vel)
        return predicted_y_pos.cpu().detach().numpy()
    def generate2(self, batch, node_type, sample:int,bestof,flexibility=0.0, ret_traj=False):
        if self.config.train_mode == 'origin':
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,\
            obs_vel, pred_vel_gt, obs_acc, pred_acc_gt, non_linear_ped,\
            loss_mask,V_obs,A_obs,Nei_obs,V_tr,A_tr,Nei_tr = batch
            obs_traj, pred_traj_gt, pred_traj_gt_rel = obs_traj.squeeze().cuda(), pred_traj_gt.squeeze().cuda(), pred_traj_gt_rel.squeeze().cuda()
            sample_outputs = torch.Tensor().cuda()
            
            for i in range(self.config.pred_seq_len):
                if i==0:
                    context = obs_traj[...,-1]
                    context = torch.stack([context]*sample, dim=0)
                else:
                    context = context + sample_outputs[:,-1,...].squeeze()
                
                pred_traj = self.diffusion.sample(context.type(torch.float32).cuda(),bestof, flexibility=flexibility, ret_traj=ret_traj)
                sample_outputs = torch.cat((sample_outputs, pred_traj.unsqueeze(1)),dim=1)
        
        elif self.config.train_mode=='multi':
            ped_features,obs_features,self_features, labels, self_hist_features = batch
            self_features = self_features[0] # N,dim
            self_hist_features = self_hist_features[0,:,:-1,:] # N, len, dim(6)
            sample_outputs = torch.Tensor().cuda()
            for i in range(self.config.pred_seq_len):

                if self.config.esti_goal == 'acce': #TODO
                    context = torch.tensor([]).cuda()
                    pred_traj = self.diffusion.sample(context) 
                elif self.config.esti_goal == 'pos':
                    if i == 0:
                        contexts = self_hist_features[...,:-1,:2].unsqueeze(0).repeat(sample,1,1,1) # sample, N, len, 2
                        contexts = contexts.permute(0,2,1,3)
                        past_traj = contexts.clone()
                        mask = contexts[:,-1,:,:]!=contexts[:,-1,:,:]
                        mask = mask[...,0] # bs, N (mask[...,0]==mask[...,1]TODO?)
                        assert len(mask.shape)==2
                    else:
                        contexts = past_traj[:,-self.config.obs_seq_len:,:,:]
                    assert self.config.obs_seq_len==contexts.shape[1]
                    dest = self_features[...,:2].unsqueeze(-3)
                    contexts = torch.cat((contexts,dest), dim=-3) # bs, obs_len+1, N, 2
                    contexts = clear_nan(contexts)
                    pred_traj = self.diffusion.sample(contexts) #bs, N, 2
                    past_traj = torch.cat((past_traj,pred_traj.unsqueeze(-3)), dim=-3)
                    sample_outputs = torch.cat((sample_outputs, pred_traj.unsqueeze(-3)),dim=-3)
        return sample_outputs.cpu().detach().numpy() # sample, pred_len, N, 2

    def generate_multistep(self, data: DATA.TimeIndexedPedData, t_start=0, load_model=True):
        """
        Args:
            data:
            t_start: rollout starts from frame #t
        """
        args = self.config
        # if load_model:
        #     self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)

        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        brights = data.brights
        shops = data.shops
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n

        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1

        history_features = data.self_hist_features[...,t_start, :, :, :]
        history_features = clear_nan(history_features)
        
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_feature = data.self_features[...,t_start, :, :]
        
        near_ped_idx = data.near_ped_idx[...,t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[...,t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[...,t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[...,t_start, :, :]
        
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2
        dest_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        ped_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1

            
            
            # a_next = self.diffusion.sample(*state_features)[0]
            if self.config.esti_goal=='acce':
                a_next = self.diffusion.sample(context = (history_features.unsqueeze(0), 
                                                          ped_features.unsqueeze(0), 
                                                          self_feature.unsqueeze(0),
                                                          obs_features.unsqueeze(0)), 
                                               curr = curr.unsqueeze(0)) 
                # print(a_next.sum())
                
                # dest force part
                self_feature = self_feature
                desired_speed = self_feature[..., -1].unsqueeze(-1)
                temp = torch.norm(self_feature[..., :2], p=2, dim=-1, keepdim=True)
                temp_ = temp.clone()
                temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
                dest_direction = self_feature[..., :2] / temp_ #des,direction
                pred_acc_dest = (desired_speed * dest_direction - self_feature[..., 2:4]) / self.tau
                pred_acc_ped = a_next - pred_acc_dest
                if t < data.num_frames-1:
                    dest_force_res[..., t+1, :, :] = pred_acc_dest
                    ped_force_res[..., t+1, :, :] = pred_acc_ped
                        
                v_next = v_cur + a_cur * data.time_unit
                p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                # if self.config.history_dim==6:
                #     p_next = self.diffusion.sample(history_features.unsqueeze(0), dest_features.unsqueeze(0))
                # elif self.config.history_dim==2:
                #     p_next = self.diffusion.sample(history_features[...,:2].unsqueeze(0), dest_features.unsqueeze(0))
                #     v_next = v_cur
                #     a_next = a_cur

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n
            # TODO: currently don't delete?
            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived 

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1 
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze()  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next
            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
            # update hist_v
            hist_v = self_feature[..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            # update hist_features
            if self.config.esti_goal=='acce':
                history_features[..., :-1, :] = history_features[..., 1:,:].clone()  # history_features: n, len, 6
                history_features[..., -1, :2] = p_cur.clone()
                history_features[..., -1, 2:4] = v_cur.clone()
                history_features[..., -1, 4:6] = a_cur.clone()
                history_features = clear_nan(history_features)
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur.clone()
            
            # calculate features
            if self.config.esti_goal=='acce':
                ped_features, obs_features, dest_features,crowd_features, shops_features,brights_features,\
                near_ped_idx, neigh_ped_mask= self.get_relative_features(data,
                    p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                    dest_cur.unsqueeze(-3), obstacles,shops,brights, args.topk_ped, args.sight_angle_ped,
                    args.dist_threshold_ped, args.topk_obs,
                    args.sight_angle_obs, args.dist_threshold_obs,args)
                ped_features = ped_features.squeeze()
                obs_features = obs_features.squeeze()
                dest_features = dest_features.squeeze() 
                self_feature = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                dest_features = dest_cur - p_cur
                dest_features[dest_features.isnan()] = 0.

            

        output = DATA.RawData(p_res, v_res, a_res, destination, destination, obstacles,
                                mask_p_new, meta_data=data.meta_data)
        return output, dest_force_res, ped_force_res
    
    def generate_multistep_geo(self, data: DATA.TimeIndexedPedData, t_start=0, load_model=True):
        """
        Args:
            data:
            t_start: rollout starts from frame #t
        """
        args = self.config
        # if load_model:
        #     self.load_model(args, set_model=False, finetune_flag=self.finetune_flag)

        destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        brights = data.brights
        shops = data.shops

        # image_path = data.image
        # id = data.id
        # description = data.description
        # box = data.box
        # if args.use_json:
        #     if args.Image_encoder == "ResNet":
        #         self.image_encoder = ResNet(weights_path=args.image_weights_path, device=args.device)
        #     if args.Text_encoder == "BERT":
        #         self.text_encoder = BERTEncoder(model_path=args.text_weights_path, device=args.device)
        #     image_big = self.image_encoder(image_path)  # 2048
        #     combined = [" This is a  " + a + '. ' + b for a, b in zip(id, description)]
        #
        #     texts_emb = self.text_encoder(combined)  # 6,768
        #     images_emb = self.image_base.get_image_patch(image_path, box)
        #     fusion_emb = self.image_text_fusion(text_emb=texts_emb, image_emb=images_emb)



        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n
        mask = data.mask_p_pred[...,t_start,:].clone().long()  # *c, t, n

        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1

        if self.config.esti_goal=='acce':
            history_features = data.self_hist_features[...,t_start, :, :, :]
            history_features = clear_nan(history_features)
        elif self.config.esti_goal=='pos': 
            raise NotImplementedError
            # hist_pos = data.self_hist_features[...,t_start, :, :, :2]
            # hist_vel = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_acce = torch.zeros_like(hist_pos, device=hist_pos.device)
            # hist_vel[:,1:,:] = data.self_hist_features[...,t_start, :, :-1, 2:4]
            # hist_acce[:,2:,:] = data.self_hist_features[...,t_start, :, :-2, 4:6]
            # history_features = torch.cat((hist_pos, hist_vel, hist_acce), dim=-1)
            # history_features = clear_nan(history_features)
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        # shops_features = data.shops_features[..., t_start, :, :]
        shops_features = torch.zeros_like(obs_features,device=obs_features.device)
        crowd_features = data.crowd_features[..., t_start, :, :]
        crowd_hist_features = data.crowd_hist_features[...,t_start, :, :, :]
        self_feature = data.self_features[...,t_start, :, :]
        brights_features = data.brights_features[..., t_start, :, :,:]
        
        near_ped_idx = data.near_ped_idx[...,t_start, :, :]
        neigh_ped_mask = data.neigh_ped_mask[...,t_start, :, :]
        near_obstacle_idx = data.near_obstacle_idx[...,t_start, :, :]
        neigh_obs_mask = data.neigh_obs_mask[...,t_start, :, :]
        
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        p_res = torch.zeros(data.position.shape, device=args.device)  # *c, t, n, 2
        v_res = torch.zeros(data.velocity.shape, device=args.device)  # *c, t, n, 2
        a_res = torch.zeros(data.acceleration.shape, device=args.device)  # *c, t, n, 2
        dest_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        ped_force_res = torch.zeros(data.acceleration.shape, device=args.device)
        

        p_res[..., :t_start + 1, :, :] = data.position[..., :t_start + 1, :, :]
        v_res[..., :t_start + 1, :, :] = data.velocity[..., :t_start + 1, :, :]
        a_res[..., :t_start + 1, :, :] = data.acceleration[..., :t_start + 1, :, :]

        mask_p_new = torch.zeros(mask_p_.shape, device=mask_p_.device)
        mask_p_new[..., :t_start + 1, :] = data.mask_p[..., :t_start + 1, :].long()

        new_peds_flag = (data.mask_p.long() - data.mask_p_pred.long())  # c, t, n

        for t in tqdm(range(t_start, data.num_frames)):
            p_res[..., t, :, :] = p_cur
            v_res[..., t, :, :] = v_cur
            a_res[..., t, :, :] = a_cur
            # mask_p_new[..., t, ~p_cur[:, 0].isnan()] = 1
            mask_p_new[..., t, :][~p_cur[..., 0].isnan()] = 1



            # a_next = self.diffusion.sample(*state_features)[0]
            if self.config.esti_goal=='acce':
                # pdb.set_trace()
                if self.config.use_json:
                    a_next = self.diffusion.sample(context = (curr.unsqueeze(0),
                                                              neigh_ped_mask.unsqueeze(0),
                                                              self_feature.unsqueeze(0),
                                                              near_ped_idx.unsqueeze(0),
                                                              history_features.unsqueeze(0),
                                                              obstacles.unsqueeze(0),
                                                              near_obstacle_idx.unsqueeze(0),
                                                              neigh_obs_mask.unsqueeze(0),
                                                              shops_features.unsqueeze(0),
                                                              crowd_features.unsqueeze(0),
                                                              crowd_hist_features.unsqueeze(0),
                                                              brights_features.unsqueeze(0),
                                                              data.text_obs_emb.detach(),  # 12
                                                              self.safe_detach(data.text_in_emb),#13
                                                              data.image_obs_emb.detach(),#14
                                                              self.safe_detach(data.image_in_emb),#15
                                                              data.box.detach(),  # 16
                                                              self.safe_detach(data.in_box),#17
                                                              data.meta_data["H"],  # 18,
                                                              data.image_big.detach(),  # 19
                                                              data.text_big.detach() , # 20
                                                              mask.detach()
                                                              ),

                                                        curr = curr.unsqueeze(0))
                else:
                    a_next = self.diffusion.sample(context=(curr.unsqueeze(0),
                                                              neigh_ped_mask.unsqueeze(0),
                                                              self_feature.unsqueeze(0),
                                                              near_ped_idx.unsqueeze(0),
                                                              history_features.unsqueeze(0),
                                                              obstacles.unsqueeze(0),
                                                              near_obstacle_idx.unsqueeze(0),
                                                              neigh_obs_mask.unsqueeze(0),
                                                              shops_features.unsqueeze(0),
                                                              crowd_features.unsqueeze(0),
                                                              crowd_hist_features.unsqueeze(0),
                                                              brights_features.unsqueeze(0) ),

                                                       curr=curr.unsqueeze(0))

                # print(a_next.sum())
                # if a_next[mask_p_[t]].max()>15 or a_next[mask_p_[t]].isnan().any():
                #     pdb.set_trace()
                # dest force part
                self_feature = self_feature
                desired_speed = self_feature[..., -1].unsqueeze(-1)
                temp = torch.norm(self_feature[..., :2], p=2, dim=-1, keepdim=True)
                temp_ = temp.clone()
                temp_[temp_ == 0] = temp_[temp_ == 0] + 0.1  # to avoid zero division
                dest_direction = self_feature[..., :2] / temp_ #des,direction
                pred_acc_dest = (desired_speed * dest_direction - self_feature[..., 2:4]) / self.tau
                pred_acc_ped = a_next - pred_acc_dest
                if t < data.num_frames-1:
                    dest_force_res[..., t+1, :, :] = pred_acc_dest
                    ped_force_res[..., t+1, :, :] = pred_acc_ped


                v_next = v_cur + a_cur * data.time_unit
                p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                # if self.config.history_dim==6:
                #     p_next = self.diffusion.sample(history_features.unsqueeze(0), dest_features.unsqueeze(0))
                # elif self.config.history_dim==2:
                #     p_next = self.diffusion.sample(history_features[...,:2].unsqueeze(0), dest_features.unsqueeze(0))
                #     v_next = v_cur
                #     a_next = a_cur

            # update destination & mask_p
            out_of_bound = torch.tensor(float('nan'), device=args.device)
            dis_to_dest = torch.norm(p_cur - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] += 1  # *c, n
            # TODO: currently don't delete?
            p_next[dest_idx_cur > dest_num - 1, :] = out_of_bound  # destination arrived 

            dest_idx_cur[dest_idx_cur > dest_num - 1] -= 1 
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze()  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            v_cur = v_next
            a_cur = a_next
            # curr = torch.cat((p_cur, v_cur, a_cur), dim=-1)
            # update hist_v
            hist_v = self_feature[..., :, 2:-3]  # *c, n, 2*x
            hist_v[..., :, :-2] = hist_v[..., :, 2:]
            hist_v[..., :, -2:] = v_cur

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]

                    # update hist_v
                    hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]
            
            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6


            # update hist_features
            if self.config.esti_goal=='acce':
                history_features[..., :-1, :] = history_features[..., 1:,:].clone()  # history_features: n, len, 6
                history_features[..., -1, :2] = p_cur.clone()
                history_features[..., -1, 2:4] = v_cur.clone()
                history_features[..., -1, 4:6] = a_cur.clone()
                history_features = clear_nan(history_features)

                dist = self.get_relative_quantity(p_cur, p_cur)  # T,N,N,2
                distance = torch.norm(dist, dim=-1)
                in_radius_mask = distance < args.perception_radius
                num_neighbors = in_radius_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                N = p_cur.shape[0]  #107

                vel_j = v_cur.unsqueeze(1).expand( -1, N, -1) #N,N,2
                masked_vel = vel_j * in_radius_mask.unsqueeze(-1)
                mean_vel = masked_vel.sum(dim=-2) / num_neighbors

                acc_j = a_cur.unsqueeze(1).expand( -1, N, -1)
                masked_acc = acc_j * in_radius_mask.unsqueeze(-1)
                mean_acc = masked_acc.sum(dim=-2) / num_neighbors  # (B_flat, T, N, 2)

                masked_dist = distance * in_radius_mask  # (B_flat, T, N, N)
                masked_dist = torch.nan_to_num(masked_dist, nan=0.0)
                mean_dist = masked_dist.sum(dim=-2) / num_neighbors.squeeze(-1)  #
                new_crowd = torch.cat((mean_vel, mean_acc, mean_dist.unsqueeze(-1)), dim=-1).unsqueeze(-2)
                crowd_hist_features = torch.cat([crowd_hist_features[..., 1:, :], new_crowd], dim=-2)
                crowd_hist_features[crowd_hist_features.isnan()] = 0
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur.clone()
            
            # calculate features
            if self.config.esti_goal=='acce':
                ped_features, obs_features, dest_features,crowd_features, shops_features,brights_features,\
                near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(data,
                    p_cur.unsqueeze(-3), v_cur.unsqueeze(-3), a_cur.unsqueeze(-3),
                    dest_cur.unsqueeze(-3), obstacles,shops,brights, args.topk_ped, args.sight_angle_ped,
                    args.dist_threshold_ped, args.topk_obs,
                    args.sight_angle_obs, args.dist_threshold_obs,args)
                ped_features = ped_features.squeeze()
                obs_features = obs_features.squeeze()
                crowd_features = crowd_features.squeeze()
                brights_features = brights_features.squeeze()
                dest_features = dest_features.squeeze() 
                near_ped_idx = near_ped_idx.squeeze() 
                neigh_ped_mask = neigh_ped_mask.squeeze() 
                near_obstacle_idx = near_obstacle_idx.squeeze() 
                neigh_obs_mask =  neigh_obs_mask.squeeze()
                # neigh_bright_idx = neigh_bright_idx.squeeze()
                # neigh_bright_mask = neigh_bright_mask.squeeze()
                
                self_feature = torch.cat((dest_features, hist_v, a_cur, desired_speed), dim=-1)
            elif self.config.esti_goal=='pos':
                raise NotImplementedError
                dest_features = dest_cur - p_cur
                dest_features[dest_features.isnan()] = 0.

            

        output = DATA.RawData(p_res, v_res, a_res, destination, destination, obstacles,
                                mask_p_new, meta_data=data.meta_data)
        return output, dest_force_res, ped_force_res


    # def forward_for_flops(self,x):
    #     traj_pred, _, _ = self.generate_multistep_geo(x, t_start=0)
    #     return traj_pred
    
    def generate_onestep(self, context:tuple, curr=None):
        history = context[0]
        ped_features = context[1]
        assert len(history.shape)==4 and len(ped_features.shape)==4 
        history = clear_nan(history)
        context = list(context)
        context[0] = history
        context = tuple(context)
        with torch.no_grad():
            if self.config.esti_goal =='pos':
                pred_traj=self.diffusion.sample(context = context) # t, N, 2
            elif self.config.esti_goal =='acce':
                pred_traj=self.diffusion.sample(context = context, curr = curr) # t, N, 2
        return pred_traj
        
        
    def get_loss(self, batch, node_type=None, timestep=0.08):
        if self.config.train_mode == 'origin':
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,\
            obs_vel, pred_vel_gt, obs_acc, pred_acc_gt, non_linear_ped,\
            loss_mask,V_obs,A_obs,Nei_obs,V_tr,A_tr,Nei_tr = batch 

            # feat_x_encoded = self.encode(batch,node_type) # B * 64
            loss = self._teacher_loss(obs_traj,pred_traj_gt,pred_traj_gt_rel)
            
        elif self.config.train_mode == 'multi':
            ped_features,obs_features,self_features, labels, self_hist_features,\
            mask_p_pred, mask_v_pred, mask_a_pred = batch
            y_t = labels[1:,:,4:6]
            # y_t = labels[:-1,:,4:6]
            y_t = clear_nan(y_t)
            curr = labels[:-1,:,:6] 

            history = self_hist_features[:-1,:,:,:6] #bs, N, obs_len, 6
            ped_features = ped_features[:-1] #bs, N, k_near, 6
            obs_features = obs_features[:-1]
            self_feature = self_features[:-1,:,:]
            history = clear_nan(history)
            mask = mask_a_pred[:-1]
            loss = self.diffusion.get_loss(y_t, curr=curr,context=(history, ped_features, self_feature, obs_features),timestep=timestep,mask = mask)
    
        else:
            raise NotImplementedError
        return loss
    
    def _teacher_loss(self, obs_traj,pred_traj_gt,pred_traj_gt_rel):
        """do teacher force training

        Args:
            obs_traj (np.array): (N,2,obs_len)
            pred_traj_gt (np.array): (N,2,pred_len)
            pred_traj_gt_rel (np.array): (N,2,pred_len)
        """
        obs_traj, pred_traj_gt, pred_traj_gt_rel = obs_traj.squeeze(), pred_traj_gt.squeeze(), pred_traj_gt_rel.squeeze()
        contexts = np.zeros([self.config.pred_seq_len, obs_traj.shape[0],obs_traj.shape[1]]) # (pred_len,N,2)
        contexts[0] = obs_traj[...,-1]
        contexts[1:] = pred_traj_gt[...,:-1].permute(2,0,1)
        contexts = torch.from_numpy(contexts).type(torch.float32)
        y_t = pred_traj_gt_rel.permute(2,0,1) # (pred_len,N,2)
        # torch.set_default_dtype(torch.float64)
        loss = self.diffusion.get_loss(y_t.type(torch.float32).cuda())
        return loss

    def test_multiple_rollouts_for_training(self, data: DATA.TimeIndexedPedData, t_start=0):
        """
        dynamic weighting;
        Args:
            data:
            t_start:

        Returns:

        """
        args = self.config

        # destination = data.destination
        waypoints = data.waypoints
        obstacles = data.obstacles
        shops = data.shops
        brights = data.brights
        mask_p_ = data.mask_p_pred.clone().long()  # *c, t, n
        mask_a_ = data.mask_a_pred.clone().long()
        
        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1)  # *c, n, 1
        history_features = data.self_hist_features[...,t_start, :, :, :]
        history_features[history_features.isnan()]=0
        ped_features = data.ped_features[..., t_start, :, :, :]
        obs_features = data.obs_features[..., t_start, :, :, :]
        self_features = data.self_features[...,t_start, :, :]
        a_cur = data.acceleration[..., t_start, :, :]  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :]  # *c, N, 2
        p_cur = data.position[..., t_start, :, :]  # *c, N, 2
        collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        label_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        p_cur_nonan = clear_nan(p_cur.clone())
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :]  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :]  # *c, N
        dest_num = data.dest_num

        new_peds_flag = (data.mask_p - data.mask_p_pred).long()  # c, t, n

        # loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        p_res = torch.zeros(data.position.shape, device=p_cur.device)
        
        a_res = torch.zeros(data.acceleration.shape, device=p_cur.device)
        
    
        for t in range(t_start, data.num_frames):
            
            collision = self.collision_detection(p_cur.clone().detach(), args.collision_threshold)  # c, n, n
            collision = torch.sum(collision, dim=-1)  # c, n
            collisions[:, t, :] = collision
            label_collision = self.collision_detection(data.labels[:, t, :, :2],
                                                           args.collision_threshold)  # c, n, n
            label_collision = torch.sum(label_collision, dim=-1)  # c, n
            label_collisions[:, t, :] = label_collision
            # a_next = self.diffusion.sample(context = (history_features.detach(), 
            #                                               ped_features.detach(), 
            #                                               self_features.detach()), 
            #                                    curr = curr.detach()).view(*a_cur.shape) 
            if t < data.num_frames - 1 + t_start:
                a_next = self.diffusion.denoise_fn(x_0 = data.acceleration[..., t+1, :, :] , #c, n, 2
                                                context = (history_features.detach(), 
                                                            ped_features.detach(), 
                                                            self_features.detach(),
                                                            obs_features.detach()), 
                                                curr = curr.detach()).view(*a_cur.shape)  
            # mask = mask_p_[:, t, :]  # c,n

            # if torch.sum(mask) > 0:
            p_res[:, t, ...] = p_cur_nonan
            a_res[:, t, ...] = a_cur

                # if args.reg_weight > 0: # mynote: regularization
                #     reg_loss += self.l1_reg_loss(p_msg, args.reg_weight, 'sum')
                #     loss = loss + reg_loss
                # if len(predictions) > 2:
                #     loss = loss + self.l1_reg_loss(o_msg, args.reg_weight, 'sum')


            v_next = v_cur + a_cur * data.time_unit
            p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            p_next_nonan = p_cur_nonan + v_cur * data.time_unit
            assert ~a_next.isnan().any(), print('find nan in epoch :', self.epoch, self.batch_idx)

            # update destination, yet do not delete people when they arrive
            dis_to_dest = torch.norm(p_cur.detach() - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] = dest_idx_cur[dis_to_dest < 0.5] + 1

            dest_idx_cur[dest_idx_cur > dest_num - 1] = dest_idx_cur[dest_idx_cur > dest_num - 1] - 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze().view(*p_cur.shape)  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            p_cur_nonan = p_next_nonan
            v_cur = v_next
            a_cur = a_next
            del p_next_nonan, p_next, v_next #ceshi
            curr = torch.cat((p_cur_nonan, v_cur, a_cur), dim=-1).detach()
            # update hist_v
            # hist_v = self_features[..., :, 2:-3]  # *c, n, 2*x
            # hist_v[..., :, :-2] = hist_v[..., :, 2:]
            # hist_v[..., :, -2:] = v_cur.detach()

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :]  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    p_cur_nonan[new_idx == 1] = data.position[..., t + 1, :, :][new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :][new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :][new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :][new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :][new_idx == 1]
                    # update hist_v
                    # hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]

            # update hist_features
            if self.config.esti_goal=='acce':
                new_traj = curr.clone().unsqueeze(-2)
                history_features = torch.cat([history_features[...,1:,:], new_traj],dim=-2)
                # history_features[..., :-1, :] = history_features[...,1:,:].detach().clone()  # history_features: n, len, 6
                # history_features[..., -1, :2] = p_cur_nonan.clone().detach()
                # history_features[..., -1, 2:4] = v_cur.clone().detach()
                # history_features[..., -1, 4:6] = a_cur.clone().detach()
                # history_features = clear_nan(history_features)   
                
            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur_nonan.clone()
            ped_features_shape = ped_features.shape
            obs_features_shape = obs_features.shape
            brights_features_shape = brights_features.shape
            crowd_features_shape = crowd_features.shape
            # calculate features
            ped_features, obs_features, dest_features,crowd_features,shops_features,brights_features,\
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(data,
                p_cur.unsqueeze(-3).detach(), v_cur.unsqueeze(-3).detach(), a_cur.unsqueeze(-3).detach(),
                dest_cur.unsqueeze(-3).detach(), obstacles,shops,brights, args.topk_ped, args.sight_angle_ped,
                args.dist_threshold_ped, args.topk_obs,
                args.sight_angle_obs, args.dist_threshold_obs,args)  # c,1,n,k,2

            self_features = torch.cat((dest_features.view(*v_cur.shape), v_cur, a_cur, desired_speed), dim=-1)
            ped_features, obs_features = ped_features.view(*ped_features_shape), obs_features.view(*obs_features_shape)
            brights_features = brights_features.view(*brights_features_shape)
            crowd_features = crowd_features.view(*crowd_features_shape)
            torch.cuda.empty_cache() #ceshi

        p_res[mask_p_ == 0] = 0.  # delete 'nan'
        data.labels[mask_p_ == 0] = 0.  # delete 'nan'
        a_res[mask_a_==0] = 0.
        
        loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        # mse_loss = self.multiple_rollout_mse_loss(p_res, data.labels[:, :, :, :2], args.time_decay, reduction='sum')
        # or
        mse_loss = self.multiple_rollout_mse_loss(a_res, data.labels[:, :, :, 4:6], args.time_decay, reduction='sum')

        loss = loss + mse_loss
        if self.config.use_col_focus_loss:
            collision_loss = self.multiple_rollout_collision_loss(
                    p_res, data.labels[:, :, :, :2], 1, args.collision_focus_weight, collisions,
                    reduction='sum')
            loss = loss + collision_loss
        
        return loss
    def safe_detach(self, x):
        return x.detach() if isinstance(x, torch.Tensor) else x
    def test_multiple_rollouts_for_training_geo(self, data: DATA.TimeIndexedPedData, t_start=0):
        """
        Args:
            data:
            t_start:0

        Returns:

        """
        args = self.config

        # destination = data.destination
        waypoints = data.waypoints.to(args.device) # B,t Ped,2
        obstacles = data.obstacles.to(args.device) # 84 2
        shops = data.shops.to(args.device)
        brights = data.brights.to(args.device)



        mask_p_ = data.mask_p_pred.clone().long() .to(args.device) # *c, t, n
        mask_a_ = data.mask_a_pred.clone().long().to(args.device)
        mask_v_ = data.mask_v_pred.clone().long().to(args.device)
        mask = mask_p_[...,t_start,:].to(args.device)
        
        desired_speed = data.self_features[...,t_start,:,-1].unsqueeze(-1) .to(args.device) # *c, n, 1
        history_features = data.self_hist_features[...,t_start, :, :, :].to(args.device)
        history_features[history_features.isnan()]=0
        ped_features = data.ped_features[..., t_start, :, :, :].to(args.device)
        obs_features = data.obs_features[..., t_start, :, :, :].to(args.device)
        self_features = data.self_features[...,t_start, :, :].to(args.device)
        shops_features = data.shops_features[..., t_start, :, :].to(args.device)
        brights_features = data.brights_features[..., t_start,:, :,:].to(args.device)
        crowd_hist_features = data.crowd_hist_features[...,t_start, :, :, :].to(args.device)

        crowd_features = data.crowd_features[..., t_start, :, :].to(args.device)
        
        near_ped_idx = data.near_ped_idx[...,t_start, :, :].to(args.device)
        neigh_ped_mask = data.neigh_ped_mask[...,t_start, :, :].to(args.device)
        near_obstacle_idx = data.near_obstacle_idx[...,t_start, :, :].to(args.device)
        neigh_obs_mask = data.neigh_obs_mask[...,t_start, :, :].to(args.device)
        
        a_cur = data.acceleration[..., t_start, :, :].to(args.device)  # *c, N, 2
        v_cur = data.velocity[..., t_start, :, :] .to(args.device) # *c, N, 2+
        p_cur = data.position[..., t_start, :, :] .to(args.device) # *c, N, 2

        collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        label_collisions = torch.zeros(mask_p_.shape, device=p_cur.device)  # c, t, n
        p_cur_nonan = clear_nan(p_cur.clone())
        curr = torch.cat((p_cur, v_cur, a_cur), dim=-1) # *c, N, 6
        dest_cur = data.destination[..., t_start, :, :].to(args.device)  # *c, N, 2
        dest_idx_cur = data.dest_idx[..., t_start, :] .to(args.device) # *c, N
        dest_num = data.dest_num.to(args.device)

        new_peds_flag = (data.mask_p.long() - data.mask_p_pred.long()).to(args.device)  # c, t, n

        # loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        p_res = torch.zeros(data.position.shape, device=p_cur.device)
        
        a_res = torch.zeros(data.acceleration.shape, device=p_cur.device)
        v_res = torch.zeros(data.velocity.shape, device=p_cur.device)
        obstacles = obstacles.unsqueeze(0).repeat(near_obstacle_idx.shape[0],1,1)
    
        for t in range(t_start, data.num_frames):
            mask_p_ = data.mask_p_pred.clone().long().to(args.device)  # *c, t, n
            mask_a_ = data.mask_a_pred.clone().long().to(args.device)
            mask_v_ = data.mask_v_pred.clone().long().to(args.device)
            mask = mask_p_[..., t, :].to(args.device)
            
            collision = self.collision_detection(p_cur.clone().detach(), args.collision_threshold)  # c, n, n
            collision = torch.sum(collision, dim=-1)  # c, n
            collisions[:, t, :] = collision
            label_collision = self.collision_detection(data.labels[:, t, :, :2],
                                                           args.collision_threshold)  # c, n, n
            label_collision = torch.sum(label_collision, dim=-1)  # c, n
            label_collisions[:, t, :] = label_collision
            # a_next = self.diffusion.sample(context = (history_features.detach(), 
            #                                               ped_features.detach(), 
            #                                               self_features.detach()), 
            #                                    curr = curr.detach()).view(*a_cur.shape)
            if t+1 < data.num_frames:
                if self.config.use_json:

                    a_next = self.diffusion.denoise_fn(x_0 = data.acceleration[..., t, :, :] , #c, n, 2
                                            context = (curr.detach(),  #0
                                                       neigh_ped_mask.detach(),  #1
                                                       self_features.detach(),  #2
                                                       near_ped_idx.detach(),  #3
                                                       history_features.detach(),  #4
                                                       obstacles.detach(),  #5
                                                       near_obstacle_idx.detach(),  #6
                                                       neigh_obs_mask.detach(),  #7
                                                       shops_features.detach(),  #8
                                                       crowd_features.detach(),  #9
                                                       crowd_hist_features.detach(),  #10
                                                       brights_features.detach(),  #11
                                                       data.text_obs_emb.detach(),  # 12
                                                       self.safe_detach(data.text_in_emb),  # 13
                                                       data.image_obs_emb.detach(), #14
                                                       self.safe_detach(data.image_in_emb), #15
                                                       data.box.detach(),  # 16
                                                       self.safe_detach(data.in_box),  # 17
                                                       data.meta_data["H"] , # 18,
                                                       data.image_big.detach(),#19
                                                       data.text_big.detach(), #20
                                                       mask.detach()#21
                                                       ),

                                            curr = curr.detach()).view(*a_cur.shape)  #chec**
                else:
                    a_next = self.diffusion.denoise_fn(x_0=data.acceleration[..., t, :, :],  # c, n, 2
                                                       context=(curr.detach(),  # 0
                                                                neigh_ped_mask.detach(),  # 1
                                                                self_features.detach(),  # 2
                                                                near_ped_idx.detach(),  # 3
                                                                history_features.detach(),  # 4
                                                                obstacles.detach(),  # 5
                                                                near_obstacle_idx.detach(),  # 6
                                                                neigh_obs_mask.detach(),  # 7
                                                                shops_features.detach(),  # 8
                                                                crowd_features.detach(),  # 9
                                                                crowd_hist_features.detach(),  # 10
                                                                brights_features.detach(),  # 11

                                                                ),

                                                       curr=curr.detach()).view(*a_cur.shape)  # chec**

            # p_res[:, t, ...] = p_cur_nonan
            a_res[:, t, ...] = a_next
            # a_res[:, t, ...] = a_cur
            if self.config.use_RVO:
                v_desire = v_cur + a_next * data.time_unit
                v_next = self.RVO.correct_velocity2(p_cur,v_cur, v_desire,near_ped_idx,neigh_ped_mask, self.config.collision_threshold)
                # a_next = (v_next - v_cur) / data.time_unit
            else:
                v_next = v_cur + a_next * data.time_unit

            v_res[:, t, ...] = v_next
            # v_res[:, t, ...] = v_cur
            # v_next = v_cur + a_cur * data.time_unit

            # p_next = p_cur + v_cur * data.time_unit  # *c, n, 2
            # p_next_nonan = p_cur_nonan + v_cur * data.time_unit
            p_next = p_cur + v_next * data.time_unit  # *c, n, 2
            p_next_nonan = p_cur_nonan + v_next * data.time_unit
            p_res[:, t, ...] = p_next_nonan
            # print("!!!!!!!!!!!!!!!!!!")
            assert ~ a_next.isnan().any(), print('find nan in epoch')

            # update destination, yet do not delete people when they arrive
            dis_to_dest = torch.norm(p_cur.detach() - dest_cur, p=2, dim=-1)
            dest_idx_cur[dis_to_dest < 0.5] = dest_idx_cur[dis_to_dest < 0.5] + 1

            dest_idx_cur[dest_idx_cur > dest_num - 1] = dest_idx_cur[dest_idx_cur > dest_num - 1] - 1
            dest_idx_cur_ = dest_idx_cur.unsqueeze(-2).unsqueeze(-1)  # *c, 1, n, 1
            dest_idx_cur_ = dest_idx_cur_.repeat(*([1] * (dest_idx_cur_.dim() - 1) + [2]))
            dest_cur = torch.gather(waypoints, -3, dest_idx_cur_).squeeze().view(*p_cur.shape)  # *c, n, 2

            # update everyone's state
            p_cur = p_next  # *c, n, 2
            p_cur_nonan = p_next_nonan


            v_cur = v_next
            a_cur = a_next
            del p_next_nonan, p_next, v_next #ceshi
            # update hist_v
            # hist_v = self_features[..., :, 2:-3]  # *c, n, 2*x
            # hist_v[..., :, :-2] = hist_v[..., :, 2:]
            # hist_v[..., :, -2:] = v_cur.detach()

            # add newly joined pedestrians
            if t < data.num_frames - 1:
                new_idx = new_peds_flag[..., t + 1, :].to(args.device)  # c, n
                if torch.sum(new_idx) > 0:
                    p_cur[new_idx == 1] = data.position[..., t + 1, :, :].to(args.device)[new_idx == 1, :]
                    p_cur_nonan[new_idx == 1] = data.position[..., t + 1, :, :].to(args.device)[new_idx == 1, :]
                    v_cur[new_idx == 1] = data.velocity[..., t + 1, :, :].to(args.device)[new_idx == 1, :]
                    a_cur[new_idx == 1] = data.acceleration[..., t + 1, :, :].to(args.device)[new_idx == 1, :]
                    dest_cur[new_idx == 1] = data.destination[..., t + 1, :, :].to(args.device)[new_idx == 1, :]
                    dest_idx_cur[new_idx == 1] = data.dest_idx[..., t + 1, :].to(args.device)[new_idx == 1]
                    # update hist_v
                    # hist_v[new_idx == 1] = data.self_features[..., t + 1, :, 2:-3][new_idx == 1]
            
            curr = torch.cat((p_cur, v_cur, a_cur), dim=-1).detach()

            # update hist_features
            if self.config.esti_goal=='acce':
                new_traj = curr.clone().unsqueeze(-2)
                dist = self.get_relative_quantity(p_cur,p_cur)   # T,N,N,2
                distance = torch.norm(dist, dim=-1)
                in_radius_mask = distance < args.perception_radius
                num_neighbors = in_radius_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                N = p_cur.shape[1]

                vel_j = v_cur.unsqueeze(2).expand(-1, -1, N, -1)
                masked_vel = vel_j * in_radius_mask.unsqueeze(-1)
                mean_vel = masked_vel.sum(dim=-2) / num_neighbors
                
                acc_j = a_cur.unsqueeze(2).expand(-1, -1, N, -1)
                masked_acc = acc_j * in_radius_mask.unsqueeze(-1)
                mean_acc = masked_acc.sum(dim=-2) / num_neighbors  # (B_flat, T, N, 2)


                masked_dist = distance * in_radius_mask  # (B_flat, T, N, N)
                masked_dist = torch.nan_to_num(masked_dist, nan=0.0)
                mean_dist = masked_dist.sum(dim=-2) / num_neighbors.squeeze(-1)#
                new_crowd = torch.cat((mean_vel, mean_acc, mean_dist.unsqueeze(-1)), dim=-1).unsqueeze(-2)
                crowd_hist_features = torch.cat([crowd_hist_features[..., 1:,:], new_crowd], dim=-2)
                crowd_hist_features[crowd_hist_features.isnan()] = 0

                history_features = torch.cat([history_features[..., 1:, :], new_traj], dim=-2)
                history_features[history_features.isnan()] = 0
                #print("Yes, corwd_hist_features is right")
                # history_features[..., -1, :2] = p_cur_nonan.clone().detach()
                # history_features[..., -1, 2:4] = v_cur.clone().detach()
                # history_features[..., -1, 4:6] = a_cur.clone().detach()
                # history_features = clear_nan(history_features)

            elif self.config.esti_goal=='pos':
                if self.config.history_dim==2:
                    history_features[..., :-1, :] = history_features[:,1:,:].clone()  # history_features: n, len, 6
                    history_features[..., -1, :2] = p_cur_nonan.clone()
            ped_features_shape = ped_features.shape
            crowd_features_shape = crowd_features.shape
            obs_features_shape = obs_features.shape
            brights_features_shape = brights_features.shape
            near_ped_idx_shape = near_ped_idx.shape
            neigh_ped_mask_shape = neigh_ped_mask.shape
            near_obstacle_idx_shape = near_obstacle_idx.shape
            neigh_obs_mask_shape = neigh_obs_mask.shape
            # neigh_bright_idx_shape = neigh_bright_idx.shape
            # neigh_bright_mask_shape = neigh_bright_mask.shape
            # calculate features
            ped_features, obs_features, dest_features,crowd_features, shops_features,brights_features,\
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask= self.get_relative_features(data,
                p_cur.unsqueeze(-3).detach(), v_cur.unsqueeze(-3).detach(), a_cur.unsqueeze(-3).detach(),
                dest_cur.unsqueeze(-3).detach(), obstacles,shops,brights, args.topk_ped, args.sight_angle_ped,
                args.dist_threshold_ped, args.topk_obs,
                args.sight_angle_obs, args.dist_threshold_obs,args)  # c,1,n,k,2
            
            self_features = torch.cat((dest_features.view(*v_cur.shape), v_cur, a_cur, desired_speed), dim=-1)
            ped_features, obs_features = ped_features.view(*ped_features_shape), obs_features.view(*obs_features_shape)
            crowd_features = crowd_features.view(*crowd_features_shape)
            brights_features = brights_features.expand(* brights_features_shape)
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = \
                            near_ped_idx.view(*near_ped_idx_shape), \
                            neigh_ped_mask.view(*neigh_ped_mask_shape), \
                            near_obstacle_idx.view(*near_obstacle_idx_shape), \
                            neigh_obs_mask.view(*neigh_obs_mask_shape)
                            # neigh_bright_idx.view(*neigh_bright_idx_shape),\
                            # neigh_bright_mask.view(*neigh_bright_mask_shape)

            


        p_res[mask_p_ == 0] = 0.  # delete 'nan'
        data.labels[mask_p_ == 0] = 0.  # delete 'nan'
        a_res[mask_a_==0] = 0.
        v_res[mask_v_==0] = 0.
        
        loss = torch.tensor(0., requires_grad=True, device=p_cur.device)
        mse_loss = self.multiple_rollout_mse_loss(p_res, data.labels[:, :, :, :2].to(args.device), args.time_decay, reduction='sum')
        # mse_loss = self.multiple_rollout_mse_loss(v_res, data.labels[:, :, :, 2:4], args.time_decay, reduction='sum')
        # or
        mse_loss2 = self.multiple_rollout_mse_loss(a_res, data.labels[:, :, :, 4:6].to(args.device), args.time_decay, reduction='sum')

        loss = loss +  mse_loss2 + mse_loss
        # loss = loss +  mse_loss2
        if self.config.use_col_focus_loss:
            collision_loss = self.local_collision_loss(
                    p_res, args.time_decay, args.collision_focus_weight,  args.collision_threshold,args.margin,
                    reduction='sum')
            #my_collision_loss2
            # collision_loss = self.my_collision_loss2(
            #     p_res, args.time_decay, args.collision_focus_weight, args.collision_threshold, args.margin,
            #     reduction='mean')
            loss = loss + collision_loss
        if self.config.use_obs_collision_loss:
            loss  = loss + self.obstacle_collision_loss_quad(p_res, data.box, data.meta_data["H"], args.time_decay,
                                                          args.collision_focus_obs, 0.5, reduction='sum')

        return loss
    
    
    def multiple_rollout_mse_loss(self, pred, labels, time_decay, reduction='none', reverse=False):
        """
        multiple rollout training loss with time decay
        Args:
            reverse:
            time_decay:
            pred: c, t, n, 2
            labels:
            reduction:

        Returns:

        """
        loss = (pred - labels) * (pred - labels)
        if not reverse: # mynote: reverse=False stands for the reverse long-term discounted factor in student-force training
            decay = torch.tensor([time_decay ** (pred.shape[1] - t - 1) for t in range(pred.shape[1])],
                                 device=pred.device)
        else:
            decay = torch.tensor([time_decay ** t for t in range(pred.shape[1])], device=pred.device)
        decay = decay.reshape(1, int(pred.shape[1]), 1, 1)
        loss = loss * decay
        return self.reduction(loss, reduction)
    
    @staticmethod
    def reduction(values, mode):
        if mode == 'sum':
            return torch.sum(values)
        elif mode == 'mean':
            return torch.mean(values)
        elif mode == 'none':
            return values
        else:
            raise NotImplementedError

    def compute_collision_number(self, pred, collision_threshold=0.5):
        """
        Args:
            pred: (c, t, n, 2) - 预测轨迹
            collision_threshold: 碰撞距离阈值
        Returns:
            collision_loss: 碰撞次数的损失
        """
        c, t, n, _ = pred.shape
        collision_loss = torch.zeros_like(pred)

        for i in range(n):
            # 计算智能体i与其他所有智能体的距离
            diff = pred - pred[:, :, i:i + 1, :]  # (c, t, n, 2)
            dist = torch.norm(diff, p=2, dim=-1, keepdim=True)  # (c, t, n, 1)

            # 检测碰撞（距离 < threshold）
            collisions = (dist < collision_threshold).float()  # (c, t, n, 1)

            # 排除自身（i != j）
            mask = torch.ones(n, device=pred.device)
            mask[i] = 0  # 屏蔽自己
            collisions = collisions * mask.view(1, 1, n, 1)  # (c, t, n, 1)

            # 对每个智能体i，累加来自其他智能体的碰撞风险
            collision_risk = torch.sum(collisions, dim=2, keepdim=True)  # (c, t, 1, 1)
            collision_loss[:, :, i:i + 1, :] = collision_risk  # 广播到 (c, t, 1, 2)

        return collision_loss
    def multiple_rollout_collision_avoidance_loss(self, pred, labels, time_decay, reduction='none'):
        """
        multiple rollout training loss with time decay
        Args:
            weight:
            time_decay:
            pred: c, t, n, 2
            labels: c, t, n, 2
            reduction:

        Returns:

        """
        ni = labels[:, -1:, :, :] - labels[:, 0:1, :, :] #真实的整体运动方向
        ni_norm = torch.norm(ni, p=2, dim=-1, keepdim=True)
        ni_norm = ni_norm + 1e-6
        ni = ni / ni_norm  # c,1,n,2#方向向量

        pred_ = pred - torch.sum(pred * ni, dim=-1, keepdim=True) * ni #减掉模拟的方向，得到垂直方向上的投影
        labels_ = labels - torch.sum(labels * ni, dim=-1, keepdim=True) * ni

        loss = self.multiple_rollout_mse_loss(pred_, labels_, time_decay, reduction='none')


        return self.reduction(loss, reduction)
    
    def multiple_rollout_collision_loss(self, pred, labels, time_decay, coll_focus_weight, collisions,
                                        reduction='none', abnormal_mask=None):
        """
        multiple rollout training loss with time decay
        Args:
            collisions: c, t, n
            time_decay:
            pred: c, t, n, 2
            labels: c, t, n, 2
            reduction:

        Returns:

        """
        collisions = torch.sum(collisions, dim=1)  # c, n
        collisions_sum = collisions
        collisions[collisions > 0] = 1.  #**
        collision_w = collisions
        collision_w = collision_w.unsqueeze(1).repeat(1, pred.shape[1], 1)
        collision_w = collision_w.unsqueeze(-1)  # c, t, n, 1

        # mse_loss = self.multiple_rollout_mse_loss(pred, labels, time_decay, reduction='none')
        collision_focus_loss = self.multiple_rollout_collision_avoidance_loss(pred, labels, time_decay,
                                                                              reduction='none')
        # loss = collision_w * (mse_loss + collision_focus_loss * coll_focus_weight)
        loss = collision_w * collision_focus_loss  # c,t,n,2

        if self.config.num_collision:
            num_loss = self.compute_collision_number(pred, self.config.collision_threshold)
            num_loss = num_loss *collision_w
            loss = loss + num_loss

        # if abnormal_mask is not None:
        #     abnormal_mask = abnormal_mask.reshape(1, 1, -1, 1)
        #     loss = loss * abnormal_mask


        return self.reduction(loss, reduction)*coll_focus_weight

    def my_collision_loss(self, pred, time_decay, coll_focus_weight, collision_threshold,margin,
                                        reduction='none', abnormal_mask=None):
        """
            Args:
                pred: Tensor of shape (c, t, n, 2) — predicted trajectories
                collision_threshold: threshold distance below which collision penalty starts
                margin: controls softness, small margin makes penalty grow earlier
                reduction: 'mean', 'sum' or 'none'

            Returns:
                Scalar tensor representing collision loss
            """
        c, t, n, _ = pred.shape
        loss_list = []

        for i in range(n):
            for j in range(i + 1, n):
                # Compute pairwise distance over all frames and samples
                dist = torch.norm(pred[:, :, i] - pred[:, :, j], dim=-1)  # shape (c, t)

                time_weights = torch.exp(-time_decay * torch.arange(t, device=pred.device).float())  # 时间衰减权重
                time_weights = time_weights.unsqueeze(0).unsqueeze(0)  # shape (1, 1, t)

                # Soft collision penalty: only penalize if too close
                penalty = F.relu(collision_threshold + margin - dist)
                penalty = penalty * time_weights
                loss_list.append(penalty)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=pred.device)

        loss_tensor = torch.stack(loss_list)  # shape (num_pairs, c, t)

        return self.reduction(loss_tensor, reduction)*coll_focus_weight

    def my_collision_loss2(self, pred, time_decay, coll_focus_weight, collision_threshold, margin,
                              reduction='none', abnormal_mask=None):
        """
            Args:
                pred: Tensor of shape (c, t, n, 2) — predicted trajectories
                collision_threshold: threshold distance below which collision penalty starts
                margin: controls softness, small margin makes penalty grow earlier
                reduction: 'mean', 'sum' or 'none'

            Returns:
                Scalar tensor representing collision loss
            """
        c, t, n, _ = pred.shape
        loss_list = []

        for i in range(n):
            for j in range(i + 1, n):
                # Compute pairwise distance over all frames and samples
                dist = torch.norm(pred[:, :, i] - pred[:, :, j], dim=-1)  # shape (c, t)

                # Soft collision penalty: only penalize if too close
                penalty = F.relu(collision_threshold + margin - dist)
                penalty = penalty
                loss_list.append(penalty)

        if len(loss_list) == 0:
            return torch.tensor(0.0, device=pred.device)

        loss_tensor = torch.stack(loss_list)  # shape (num_pairs, c, t)
        time_weights = torch.exp(-time_decay * torch.arange(t, device=pred.device).float())
        time_weights = time_weights / time_weights.sum()  # normalize
        loss_tensor = loss_tensor * time_weights[None, None, :]

        return self.reduction(loss_tensor, reduction) * coll_focus_weight

    def my_collision_loss_nodecay(self, pred, time_decay, coll_focus_weight, collision_threshold, margin,
                               reduction='none', abnormal_mask=None):
            """
                Args:
                    pred: Tensor of shape (c, t, n, 2) — predicted trajectories
                    collision_threshold: threshold distance below which collision penalty starts
                    margin: controls softness, small margin makes penalty grow earlier
                    reduction: 'mean', 'sum' or 'none'

                Returns:
                    Scalar tensor representing collision loss
                """
            c, t, n, _ = pred.shape
            loss_list = []

            for i in range(n):
                for j in range(i + 1, n):
                    # Compute pairwise distance over all frames and samples
                    dist = torch.norm(pred[:, :, i] - pred[:, :, j], dim=-1)  # shape (c, t)

                    # Soft collision penalty: only penalize if too close
                    penalty = F.relu(collision_threshold + margin - dist)
                    penalty = penalty
                    loss_list.append(penalty)

            if len(loss_list) == 0:
                return torch.tensor(0.0, device=pred.device)

            loss_tensor = torch.stack(loss_list)  # shape (num_pairs, c, t)
            # time_weights = torch.exp(-time_decay * torch.arange(t, device=pred.device).float())
            # time_weights = time_weights / time_weights.sum()  # normalize
            # loss_tensor = loss_tensor * time_weights[None, None, :]

            return self.reduction(loss_tensor, reduction) * coll_focus_weight

    # def local_collision_loss(self, pred, time_decay, coll_focus_weight, collision_threshold, margin,
    #                          reduction='none'):
    #     """
    #     局部关注的碰撞损失，仅对当前预测中潜在碰撞人对施加惩罚。
    #
    #     Args:
    #         pred: Tensor (c, t, n, 2) — predicted trajectories
    #         time_decay: float — 时间衰减因子
    #         coll_focus_weight: float — 总权重
    #         collision_threshold: float — 距离阈值
    #         margin: float — 软间隔
    #         reduction: str — 'sum', 'mean', or 'none'
    #
    #     Returns:
    #         scalar or tensor — 碰撞损失
    #     """
    #
    #     c, t, n, _ = pred.shape
    #     device = pred.device
    #     loss_list = []
    #
    #     # 时间衰减权重 (1 closer, e^-time_decay * t farther)
    #     time_weights = torch.exp(-time_decay * torch.arange(t, device=device).float())  # shape (t,)
    #     time_weights = time_weights.view(1, t)  # shape (1, t, 1) for broadcasting
    #
    #     for i in range(n):
    #         for j in range(i + 1, n):
    #             dist = torch.norm(pred[:, :, i] - pred[:, :, j], dim=-1)  # shape (c, t)
    #             # 只对距离小于阈值的部分惩罚（软 margin）
    #             mask = dist < (collision_threshold + margin)
    #             penalty = F.relu(collision_threshold + margin - dist)  # shape (c, t)
    #
    #             penalty = penalty * mask.float()  # 只保留潜在碰撞处
    #             weighted_penalty = penalty * time_weights  # 加时间衰减
    #             loss_list.append(weighted_penalty)  # (c, t)
    #
    #     if len(loss_list) == 0:
    #         return torch.tensor(0.0, device=pred.device)
    #
    #     loss_tensor = torch.stack(loss_list, dim=0)  # shape (num_pairs, c, t)
    #     loss_tensor = loss_tensor.sum(dim=0)  # (c, t)
    #
    #
    #     return self.reduction(loss_tensor, reduction)*coll_focus_weight
    def local_collision_loss(self, pred, time_decay, coll_focus_weight, collision_threshold, margin,
                             reduction='none'):
        c, t, n, _ = pred.shape
        device = pred.device

        # pairwise distance (c, t, n, n)
        diff = pred.unsqueeze(2) - pred.unsqueeze(3)
        dist = torch.norm(diff, dim=-1)

        # only upper triangle pairs
        triu_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        dist_pairs = dist[:, :, triu_mask]  # (c, t, num_pairs)

        # penalty with margin
        penalty = F.relu(collision_threshold + margin - dist_pairs)
        mask = dist_pairs < (collision_threshold + margin)
        penalty = penalty * mask.float()

        # time decay
        time_weights = torch.exp(-time_decay * torch.arange(t, device=device).float())  # (t,)
        penalty = penalty * time_weights.view(1, t, 1)  # (c, t, num_pairs)

        # sum over pairs
        loss_tensor = penalty.sum(dim=-1)  # (c, t)

        if reduction == 'sum':
            loss = loss_tensor.sum()
        elif reduction == 'mean':
            loss = loss_tensor.mean()
        else:
            loss = loss_tensor  # 'none'

        return loss * coll_focus_weight

        # return 0

        # 2) 图像坐标的旋转矩形 -> 四个角点 (图像坐标)

    def transform_points_torch(self, points: torch.Tensor, H) -> torch.Tensor:
        """
        points: (..., 2), torch.Tensor (dtype/设备自适应)
        H: (3, 3), 可以是 numpy.ndarray 或 torch.Tensor，表示 图像->世界 的单应矩阵
        return: 与 points 同形状的世界坐标 (..., 2)
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError("points 必须是 torch.Tensor")
        if not isinstance(H, torch.Tensor):
            H = torch.as_tensor(H, dtype=points.dtype, device=points.device)
        else:
            H = H.to(dtype=points.dtype, device=points.device)

        orig_shape = points.shape
        pts = points.reshape(-1, 2)  # (N, 2)
        ones = torch.ones(pts.size(0), 1, device=pts.device, dtype=pts.dtype)
        homo = torch.cat([pts, ones], dim=1)  # (N, 3)
        trans = homo @ H.t()  # (N, 3)
        w = trans[:, 2:3].clamp_min(1e-8)  # 避免除零
        xy = trans[:, :2] / w
        return xy.view(orig_shape)

    # 2) 图像坐标的旋转矩形 -> 四个角点 (图像坐标)
    def rbox_corners_image(self,obs_img: torch.Tensor) -> torch.Tensor:
        """
        obs_img: (m, 5) [cx, cy, w, h, angle(rad)] in image coords
        return: (m, 4, 2) 四个角点（按 CCW 顺序），图像坐标
        """
        cx, cy, w, h, ang = [obs_img[:, i] for i in range(5)]
        c, s = torch.cos(ang), torch.sin(ang)
        # 旋转矩阵（标准图像坐标，x 右，y 下；若你的角度定义不同，请相应调整）
        R = torch.stack([torch.stack([c, -s], dim=-1),
                         torch.stack([s, c], dim=-1)], dim=-2)  # (m, 2, 2)

        hx, hy = w * 0.5, h * 0.5
        # 局部未旋转矩形的角点顺序：(-hx,-hy),(+hx,-hy),(+hx,+hy),(-hx,+hy)
        base = torch.stack([
            torch.stack([-hx, -hy], dim=-1),
            torch.stack([hx, -hy], dim=-1),
            torch.stack([hx, hy], dim=-1),
            torch.stack([-hx, hy], dim=-1),
        ], dim=1)  # (m, 4, 2)

        # 旋转 + 平移到图像全局
        corners = base @ R.transpose(-1, -2)  # (m, 4, 2)
        center = torch.stack([cx, cy], dim=-1).unsqueeze(1)  # (m, 1, 2)
        return corners + center  # (m, 4, 2)

    # 3) 点到线段距离（批量）
    def point_to_segment_distance(self, P: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        P: (..., 2)
        A: (..., 2)
        B: (..., 2)
        return: (...) 点到 AB 线段的欧氏距离
        """
        AB = B - A
        AP = P - A
        denom = (AB * AB).sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (..., 1)
        t = (AP * AB).sum(dim=-1, keepdim=True) / denom  # (..., 1)
        t = t.clamp(0.0, 1.0)
        closest = A + t * AB  # (..., 2)
        return ((P - closest) ** 2).sum(dim=-1).sqrt()  # (...)


    def obstacle_collision_loss_quad(self, pred, obs_img, H, time_decay, obs_focus_weight,
                                     collision_threshold, reduction='none'):
        """
        pred:    (c, t, n, 2)  预测（世界坐标）
        obs_img: (m, 5)        [cx, cy, w, h, angle]（图像坐标）
        H:       (3, 3)        图像->世界 单应矩阵
        """
        device = pred.device
        dtype = pred.dtype
        c, t, n, _ = pred.shape

        # (a) 图像空间的四个角点 -> 世界坐标四边形
        obs_img = torch.as_tensor(obs_img, dtype=dtype, device=device)  # (m,5)
        corners_img = self.rbox_corners_image(obs_img)  # (m,4,2) image
        corners_world = self.transform_points_torch(corners_img.reshape(-1, 2), H) \
            .reshape(corners_img.shape)  # (m,4,2) world

        # (b) 构建边（按顶点顺序的四条边）
        A = corners_world[:, [0, 1, 2, 3], :]  # (m,4,2)
        B = corners_world[:, [1, 2, 3, 0], :]  # (m,4,2)

        # (c) 广播到 (c,t,n,m,4,2)
        P = pred.unsqueeze(3).unsqueeze(4)  # (c,t,n,1,1,2)
        A = A.view(1, 1, 1, -1, 4, 2).to(device=device, dtype=dtype)  # (1,1,1,m,4,2)
        B = B.view(1, 1, 1, -1, 4, 2).to(device=device, dtype=dtype)

        # (d) 点是否在多边形内部（凸四边形）
        AB = B - A  # (c,t,n,m,4,2)
        AP = P - A  # (c,t,n,m,4,2)
        cross = AB[..., 0] * AP[..., 1] - AB[..., 1] * AP[..., 0]  # (c,t,n,m,4)
        inside_ccw = (cross >= 0).all(dim=-1)
        inside_cw = (cross <= 0).all(dim=-1)
        inside = inside_ccw | inside_cw  # (c,t,n,m)

        # (e) 外部点到四条边的最近距离，内部为 0
        dists_edges = self.point_to_segment_distance(P, A, B)  # (c,t,n,m,4)
        dist_outside = dists_edges.min(dim=-1).values  # (c,t,n,m)
        dist = torch.where(inside, torch.zeros_like(dist_outside), dist_outside)

        # (f) 惩罚：不考虑 margin
        penalty = F.relu(collision_threshold - dist)  # (c,t,n,m)

        # (g) 时间衰减
        tw = torch.exp(-time_decay * torch.arange(t, device=device, dtype=dtype))
        penalty = penalty * tw.view(1, t, 1, 1)

        # (h) 聚合
        loss_tensor = penalty.sum(dim=-1).sum(dim=-1)  # (c,t)
        if reduction == 'sum':
            loss = loss_tensor.sum()
        elif reduction == 'mean':
            loss = loss_tensor.mean()
        else:
            loss = loss_tensor
        return loss * obs_focus_weight

