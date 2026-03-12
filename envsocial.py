import os
import argparse
import torch
import dill
# import pdb
import numpy as np
import cv2
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
from utils.draw_loss import draw_curve_train,draw_curve_test
# import pickle
# from environment.TrajectoryDS import TrajectoryDataset
from thop import  profile
import  time
from torch.utils.data import DataLoader
from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
# from models.trajectron import Trajectron
# from utils.model_registrar import ModelRegistrar
from utils import data_loader as LOADER
from utils.trajectron_hypers import get_traj_hypers
from utils.utils import mask_mse_func, post_process
from utils.visualization import plot_trajectory
import evaluation
import data.data as DATA
import data.dataset as DATASET
import torch.nn.functional as F
import utils.metrics as METRIC
from collections import OrderedDict
from collections import defaultdict

#

class EnvSocial():
    def __init__(self, config):


        # pass
        self.config = config
        torch.backends.cudnn.benchmark = True # explanation: https://blog.csdn.net/leviopku/article/details/121661020
        self._build()



        # self.load_current_model()


    def train(self):

        test = False
        
        device = self.config.device
        train_loaders = self.finetune_train_loaders
        val_list = self.finetune_valid_list
        optimizer = self.ft_optimizer
        scheduler = self.ft_scheduler
        # scheduler = self.scheduler
        # optimizer = self.optimizer
        total_epochs = self.config.total_epochs
        finetune_flag = True
        train_data_num = int(len(train_loaders)*self.config.train_data_ratio)
        train_loaders = train_loaders[:train_data_num]
        loss_epoch = {}
        best_MAE = 1e9
        best_OT = 1e9
        best_MMD = 1e9
        best_COL = 1e9
        best_DTW = 1e9
        lss = []
        scLr = []
        loss_epoch = {}
        # loss_path = 'tt/base.txt'
        # lr_path = 'tt/base.txt'
        test_mae = []
        test_OT = []
        test_mse = []
        test_FDE = []
        test_epoch = []
        test_coll = []
        test_MMD = []
        test_DTW = []
        train_loss = []
        train_iter = []
        train_path = os.path.join(self.logs_dir, "trainbrightcol0.5grid0.8.jpg")
        test_path = os.path.join(self.logs_dir, "testbrightcol0.5grid0.8.jpg")
        for epoch in range(total_epochs+1):

            if epoch!=0 or epoch==0:
                loss_epoch[epoch]=[]
                if self.config.train_mode=='multi':
                    mse_list = []
                    # tic = time.time()
                    for i, train_loader in enumerate(train_loaders):
                        if self.config.finetune_trainmode=='singlestep' or not finetune_flag:
                            pbar = tqdm(train_loader, ncols=90)
                            # pbar = tqdm(list(itertools.islice(train_loaders,1)), ncols=90)
                            for batch_idx, batch_data in enumerate(pbar):
                                self.optimizer.zero_grad()
                                self.batch_idx = batch_idx
                                train_loss = self.model.get_loss(batch_data)
                                if self.config.diffnet!='SpatialTransformer_dest_force':
                                    train_loss.backward()
                                    optimizer.step()
                                    scheduler.step()
                                    print('last lr:',scheduler.get_last_lr())
                                loss_epoch[epoch].append(train_loss.item())
                                # mse_list.append(train_loss.item())
                                pbar.set_description(f"Epoch {epoch}, {batch_idx+1}/{len(pbar)},{i+1}/{len(self.train_loaders)} MSE: {np.mean(loss_epoch[epoch])}")
                        elif self.config.finetune_trainmode=='multistep':

                            if test:
                                torch.cuda.empty_cache()
                                torch.cuda.reset_peak_memory_stats(device)

                            self.optimizer.zero_grad()
                            # torch.autograd.set_detect_anomaly(True)
                            assert type(train_loader)==DATA.ChanneledTimeIndexedPedData
                            loss = self.model.test_multiple_rollouts_for_training_geo(train_loader)
                            # loss = self.model.test_multiple_rollouts_for_training(train_loader)

                            loss_epoch[epoch].append(loss.item())
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            if test:
                                torch.cuda.synchronize(device)
                                peak_mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                                reserved_peak_mb = torch.cuda.max_memory_reserved(device) / 1024 ** 2
                                print(
                                    f"peak allocated: {peak_mb:.1f} MB, reserved peak: {reserved_peak_mb:.1f} MB")
                            if i == len(train_loaders) - 1:
                                mse_mean = np.mean(loss_epoch[epoch])
                                last_lr = scheduler.get_last_lr()
                                lss.append(mse_mean)
                                scLr.append(last_lr)
                                self.log.info(f"Epoch {epoch} loss: {mse_mean}")
                                self.log.info(f"Epoch:{epoch} lr: {last_lr}")
                                print(f"已训练{epoch}个epoch")
                                print(f"Epoch {epoch} mean seqs loss:{mse_mean}")
                                train_loss.append(mse_mean)
                                train_iter.append(epoch)
                                draw_curve_train(train_path, train_iter, train_loss)
                                print(f'Epoch {epoch}  lr:', last_lr)
                            del loss
                            # import gc
                            # gc.collect()
                            # torch.cuda.empty_cache()
                    # tic2 = time.time()
                    # print(f"Inference time: {tic2-tic:.3f} s")
                self.log_writer.add_scalar('train_MSE', np.mean(loss_epoch[epoch]), epoch)
            
            # self.train_dataset.augment = False
            
            if (epoch) % self.config.eval_every == 0  :
                self.model.eval()
                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []
                if self.config.train_mode =='origin':
                    pbar = tqdm(self.dataloader_eval)
                    i = 0
                    for batch in pbar:
                        i+=1
                        traj_pred = self.model.generate2(batch, node_type, num_points=0,sample=20,bestof=True)
                        gt = batch[3].squeeze().permute(2,0,1)
                elif self.config.train_mode == 'multi':
                    if self.config.val_mode=='multistep':
                        mse_list = []
                        mae_list = []
                        ot_list = []
                        FDE_list = []
                        mmd_list = []
                        collision_list = []
                        dtw_list = []
                        ipd_list = []
                        ipd_mmd_list = []

                        for i, val_data in enumerate(val_list):

                            # labels = val_data.labels[..., :2]
                            #
                            # if 1 == 1:
                            #     print("test label collison")
                            #     collision = METRIC.collision_count(labels, 0.5, reduction='sum')
                            #     print(collision)
                            #     return



                            with torch.no_grad():
                                #计算GFLOPs
                                if test:
                                    sample_input = val_data  # 或者 val_data 里取一个 batch
                                    wrapper_model = FlopsWrapper(self.model)
                                    flops, params = profile(wrapper_model, inputs=(sample_input,), verbose=False)
                                    print(f"GFLOPs: {flops / 1e9:.2f}, Params: {params / 1e6:.2f}M")
                                    start = time.time()
                                    #测内存，显存占用
                                    torch.cuda.synchronize(device)
                                    # 如果你是测单次 generate_multistep_geo，传入一个 representative batch
                                    start_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
                                    start_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)

                                traj_pred, dest_force, ped_force = self.model.generate_multistep_geo(val_data, t_start=self.config.skip_frames)
                                # traj_pred, dest_force, ped_force = self.model.generate_multistep(val_data, t_start=self.config.skip_frames)
                                if test:
                                    torch.cuda.synchronize(device)
                                    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                                    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                                    print(f"Start allocated: {start_mem:.1f} MB, Peak allocated: {peak_mem:.1f} MB")
                                    print(f"Start reserved: {start_reserved:.1f} MB, Peak reserved: {peak_reserved:.1f} MB")
                                    end = time.time()
                                    print(f"Inference time: {end-start:.3f} s")



                                ipd_mmd = METRIC.get_nearby_distance_mmd(traj_pred.position, traj_pred.velocity, 
                                                                val_data.labels[..., :2], val_data.labels[..., 2:4], 
                                                                val_data.mask_p_pred.long(), self.config.dist_threshold_ped, self.config.topk_ped*2, reduction='mean')
                                ipd_mmd_list.append(ipd_mmd)
                                p_pred = traj_pred.position
                                # p_pred_ = p_pred.clone()
                                # p_pred_[:-1, :, :] = p_pred_[1:, :, :].clone()
                                # p_pred = p_pred_
                                mask_p_pred = val_data.mask_p_pred.long()  # (*c) t, n
                                labels = val_data.labels[..., :2]





                                FDE = METRIC.fde_at_label_end(p_pred, labels, reduction='mean')
                                
                                p_pred = post_process(val_data, p_pred, traj_pred.mask_p, mask_p_pred)
                                dtw = METRIC.dtw_tensor(p_pred, labels, mask_p_pred, mask_p_pred,reduction='mean')
                                dtw_list.append(dtw)
                        
                                
                                func = lambda x: x*torch.exp(-x)
                                ipd = METRIC.inter_ped_dis(p_pred, labels, mask_p_pred,reduction='mean',applied_func=func)
                                ipd_list.append(ipd)
                                
                                loss = F.mse_loss(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1], reduction='mean')
                                loss = loss.item()
                                mse_list.append(loss)
                                mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                                mae_list.append(mae)

                            ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean', dvs=self.config.device)
                            mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                            collision = METRIC.collision_count(p_pred, 0.5, reduction='sum')
                            
                            ot_list.append(ot)
                            mmd_list.append(mmd)
                            FDE_list.append(FDE)
                            collision_list.append(collision)
                            # ade,fde = evaluation.compute_batch_statistics2(traj_pred,gt,best_of=True)
                            # eval_ade_batch_errors.append(ade)
                            # eval_fde_batch_errors.append(fde)
                        ade = np.mean(eval_ade_batch_errors)
                        fde = np.mean(eval_fde_batch_errors)
                        mse = np.mean(mse_list)
                        mae = np.mean(mae_list)
                        ot = np.mean(ot_list)
                        FDE = np.mean(FDE_list)
                        mmd = np.mean(mmd_list)
                        collision = np.mean(collision_list)
                        dtw = np.mean(dtw_list)
                        ipd = np.mean(ipd_list)
                        ipd_mmd = np.mean(ipd_mmd_list)
                        # if self.config.dataset == "eth":
                        #     ade = ade/0.6
                        #     fde = fde/0.6
                        # elif self.config.dataset == "sdd":
                        #     ade = ade * 50
                        #     fde = fde * 50
                        # print(f"Epoch {epoch} Best Of 20: FDE: {FDE}")
                        print(f"Epoch {epoch} MSE: {mse} MAE: {mae}")
                        print(f"Epoch {epoch} OT: {ot} MMD: {mmd}")
                        print(f"Epoch {epoch} collision: {collision} FDE: {FDE}")
                        print(f"Epoch {epoch} dtw: {dtw} inter ped distance mmd: {ipd_mmd}")

                        test_mae.append(mae)
                        test_mse.append(mse)
                        test_FDE.append(FDE)
                        test_OT.append(ot)
                        test_MMD.append(mmd)
                        test_DTW.append(dtw)
                        test_epoch.append(epoch)
                        test_coll.append(collision)

                        draw_curve_test(test_path, test_epoch, test_mse, test_mae, test_OT, test_MMD, test_FDE,test_DTW, test_coll)

                        # self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")
                        # self.log_writer.add_scalar('ADE', ade, epoch)
                        # self.log_writer.add_scalar('FDE', fde, epoch)
                        self.log.info(f"Epoch {epoch} MSE: {mse} MAE: {mae}")
                        self.log.info(f"Epoch {epoch} OT: {ot} MMD: {mmd}")
                        self.log.info(f"Epoch {epoch} collision: {collision} FDE: {FDE}")
                        self.log.info(f"Epoch {epoch} dtw: {dtw} inter ped distance mmd: {ipd_mmd}")
                        self.log.info(" ")
                        self.log_writer.add_scalar('MSE', mse, epoch)
                        self.log_writer.add_scalar('MAE', mae, epoch)
                        self.log_writer.add_scalar('OT', ot, epoch)
                        self.log_writer.add_scalar('MMD', mmd, epoch)
                        self.log_writer.add_scalar('Collision', collision, epoch)
                        self.log_writer.add_scalar('fde', FDE, epoch)
                        self.log_writer.add_scalar('dtw', dtw, epoch)
                        self.log_writer.add_scalar('ipd_mmd', ipd_mmd, epoch)
                        if self.config.save_model:
                            # save_dir = os.path.join(self.logs_dir,'chpt')
                            # print(f'save at epoch {epoch} to {save_dir}...')
                            # os.makedirs(save_dir, exist_ok=True)
                            # save_name = f'Epoch {epoch}' +'.pkl'
                            # save_path = os.path.join(save_dir, save_name)
                            # net_state_dict = self.model.state_dict()
                            # torch.save(net_state_dict, save_path)
                            if mae < best_MAE or ot < best_OT or mmd < best_MMD or dtw < best_DTW or collision< best_COL:
                                best_MMD = min(best_MMD, mmd)
                                best_OT = min(best_OT, ot)
                                best_MAE = min(best_MAE, mae)
                                save_dir1 = os.path.join(self.logs_dir, 'best')
                                best_DTW = min(best_DTW, dtw)
                                best_COL = min(best_COL, collision)

                                current_time = epoch
                                log_line = (
                                    f"{current_time} - ,"
                                    f"MAE: {mae:.4f}, "
                                    f"OT: {ot:.4f}, "
                                    f"MMD: {mmd:.4f}, "  # 可根据数值类型调整格式
                                    f"DTW: {dtw:.4f}, "
                                    f"COL: {collision: .4f}, "
                                    f"FDE: {FDE: .4f},"
                                    f"save_dir: {save_dir1}\n"
                                )

                                # 追加写入文件（确保 logs_dir 存在）
                                  # 确保目录存在
                                with open(os.path.join(self.logs_dir,"fubright_grid_0.8_col_1_debug.txt"), 'a') as f:
                                    f.write(log_line)

                                print(f"find a best performance, save at epoch{epoch} to {save_dir1} ...")
                                self.save_model(self.model, self.optimizer,self.scheduler, epoch,  save_dir1)




                            # if epoch %100 ==0 :
                            #     save_dir = os.path.join(self.logs_dir, 'last')
                            #     print(f'save at epoch {epoch} to {save_dir}...')
                            #     self.save_model(self.model, self.optimizer, epoch, save_dir)
                        del traj_pred
                        # import gc
                        # gc.collect()
                        # torch.cuda.empty_cache()
                    elif self.config.val_mode=='singlestep':
                        val_mses=[]
                        for i, val_data in enumerate(val_list):
                            gt = val_data.labels[1:,:,4:6]
                            hist_fea = val_data.self_hist_features[:-1,:,:,:6]
                            ped_features = val_data.ped_features[:-1]
                            obs_features = val_data.obs_features[:-1]
                            self_feature = val_data.self_features[:-1,:,:]
                            context = (hist_fea, ped_features, self_feature, obs_features)
                            curr = val_data.labels[:-1,:,:6]
                            traj_pred = self.model.generate_onestep(context,curr)
                            mask = val_data.mask_a_pred[:-1]
                            # mask[0] = mask[1]
                            val_mse = mask_mse_func(traj_pred, gt, mask).item()
                            print(f"Val Dataset {i} Epoch {epoch} val mse {val_mse}")
                            self.log.info(f"Val Dataset {i} Epoch {epoch} val mse: {val_mse}")
                            self.log_writer.add_scalar(f'val_mse_{i}', val_mse, epoch)
                            val_mses.append(val_mse)
                    
                        val_mse_mean = np.mean(val_mses)
                        if self.config.save_model:
                            save_dir = os.path.join(self.logs_dir,'ckpt')
                            print(f'save at epoch {epoch} to {save_dir}...')
                            os.makedirs(save_dir, exist_ok=True)
                            save_name = f'epoch{epoch}_val_mse_'+format(val_mse_mean,'.5f')+'.pkl'
                            save_path = os.path.join(save_dir, save_name)
                            net_state_dict = self.model.state_dict()
                            torch.save(net_state_dict, save_path)
                
                self.model.train()


    def _test(self):
        self.model.eval()

        node_type = "PEDESTRIAN"

        val_list = self.finetune_valid_list
        if self.config.train_mode == 'origin':
            pbar = tqdm(self.dataloader_eval)
            i = 0
            for batch in pbar:
                i += 1
                traj_pred = self.model.generate2(batch, node_type, num_points=0, sample=20, bestof=True)
                gt = batch[3].squeeze().permute(2, 0, 1)
        elif self.config.train_mode == 'multi':
            if self.config.val_mode == 'multistep':
                mse_list = []
                mae_list = []
                ot_list = []
                FDE_list = []
                mmd_list = []
                collision_list = []
                dtw_list = []
                ipd_list = []
                ipd_mmd_list = []
                path1 = os.path.join(self.logs_dir, "lable.npy")
                path2 = os.path.join(self.logs_dir, "pred.npy")
                for i, val_data in enumerate(val_list):
                    with torch.no_grad():
                        gtcollision = METRIC.collision_count(val_data.labels[..., :2], 0.5, reduction='sum')
                        # print("GT_collision", gtcollision)

                        traj_pred, dest_force, ped_force = self.model.generate_multistep_geo(val_data,
                                                                                             t_start=self.config.skip_frames)



                        # traj_pred, dest_force, ped_force = self.model.generate_multistep(val_data, t_start=self.config.skip_frames)
                        # print(traj_pred.position)
                        ipd_mmd = METRIC.get_nearby_distance_mmd(traj_pred.position, traj_pred.velocity,
                                                                 val_data.labels[..., :2], val_data.labels[..., 2:4],
                                                                 val_data.mask_p_pred.long(),
                                                                 self.config.dist_threshold_ped,
                                                                 self.config.topk_ped * 2, reduction='mean')
                        ipd_mmd_list.append(ipd_mmd)
                        p_pred = traj_pred.position
                        # p_pred_ = p_pred.clone()
                        # p_pred_[:-1, :, :] = p_pred_[1:, :, :].clone()
                        # p_pred = p_pred_
                        mask_p_pred = val_data.mask_p_pred.long()  # (*c) t, n
                        labels = val_data.labels[..., :2]
                        FDE = METRIC.fde_at_label_end(p_pred, labels, reduction='mean')
                        labels_np = labels.cpu().numpy()
                        pred_np = p_pred.cpu().numpy()
                        np.save(path1, labels_np)
                        np.save(path2, pred_np)
                        p_pred = post_process(val_data, p_pred, traj_pred.mask_p, mask_p_pred)

                        # np.save("predict_position.npy", p_pred.detach().cpu().numpy())
                        # np.save("label_position.npy", labels.detach().cpu().numpy())

                        func = lambda x: x * torch.exp(-x)
                        ipd = METRIC.inter_ped_dis(p_pred, labels, mask_p_pred, reduction='mean', applied_func=func)
                        ipd_list.append(ipd)

                        loss = F.mse_loss(p_pred[mask_p_pred == 1], labels[mask_p_pred == 1], reduction='mean')
                        loss = loss.item()
                        mse_list.append(loss)
                        mae = METRIC.mae_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                        mae_list.append(mae)

                        ot = METRIC.ot_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean',
                                                      dvs=self.config.device)
                        mmd = METRIC.mmd_with_time_mask(p_pred, labels, mask_p_pred, reduction='mean')
                        collision = METRIC.collision_count(p_pred, 0.5, reduction='sum')
                        dtw = METRIC.dtw_tensor(p_pred, labels, mask_p_pred, mask_p_pred,reduction='mean')
                        dtw_list.append(dtw)
                        print("mae:",mae,"ot",ot, "mmd:",mmd,"collision:",collision,"dtw:",dtw,'fde:',FDE)

    def load_current_model(self,model_name):
        model_pretrain = model_name  #gc grid 300

        self.log.info(f"加载预训练模型: {model_pretrain}")

        # 加载检查点
        checkpoint = torch.load(model_pretrain,
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        state_dict = checkpoint['model_state_dict']

        # 检查当前模型是否为 DataParallel
        is_data_parallel = isinstance(self.model, torch.nn.DataParallel)

        # 检查 state_dict 的键是否带有 'module.' 前缀
        has_module_prefix = list(state_dict.keys())[0].startswith('module.')

        if is_data_parallel and not has_module_prefix:
            # 当前模型是 DataParallel，但 state_dict 没有 'module.' 前缀，添加前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict['module.' + k] = v
            state_dict = new_state_dict
        elif not is_data_parallel and has_module_prefix:
            # 当前模型不是 DataParallel，但 state_dict 有 'module.' 前缀，移除前缀
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_key = k[7:]  # 移除 'module.' 前缀
                else:
                    new_key = k
                new_state_dict[new_key] = v
            state_dict = new_state_dict

        # 加载模型参数
        try:
            self.model.load_state_dict(state_dict)
            self.log.info("模型参数加载成功")
        except RuntimeError as e:
            self.log.error(f"加载模型参数时出错: {e}")
            raise e

        # 加载优化器状态
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.log.info("优化器,调度器状态加载成功")
        except RuntimeError as e:
            self.log.error(f"加载优化器状态时出错: {e}")
            raise e

        # 获取 epoch 信息
        epoch = checkpoint.get('epoch', 0)
        self.log.info(f"预训练模型加载完成，epoch: {epoch}")

        print("加载预训练模型成功")
        # print(f"当前模型状态: {self.model}")

    def _build(self):
        self._build_dir()

        # self._build_encoder_config()
        # self._build_encoder()
        self._build_model()
        # self._build_train_loader()
        # self._build_eval_loader()
        if self.config.train_mode=='origin':
            self._build_train_loader2()
            self._build_eval_loader2()
        elif self.config.train_mode == 'multi':
            self._build_data_loader()
        else:
            raise NotImplementedError

        self._build_optimizer()
        

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        import sys
        debug_flag = 'run' if sys.gettrace() ==None else 'debug'
        print('running in',debug_flag,'mode')
        logs_dir = osp.join(self.model_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))

        logs_dir += debug_flag

        self.logs_dir = logs_dir
        os.makedirs(logs_dir,exist_ok=True)
        self.log_writer = SummaryWriter(log_dir=logs_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(logs_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)
        self.log.info(time.strftime('%Y-%m-%d-%H-%M-%S'))
        self.log.info("Config:")
        for item in self.config.items():
            self.log.info(item)

        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([
                                    # {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                    {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.999)
        if 'ft_lr' not in self.config.keys():
            if 'ucy' in self.config.data_dict_path:
                # self.config.ft_lr=self.config.lr/100
                self.config.ft_lr = self.config.lr / 100
            elif 'gc' in self.config.data_dict_path:
                self.config.ft_lr=self.config.lr/10
            elif 'eth' in self.config.data_dict_path:
                self.config.ft_lr=self.config.lr/10
            else:
                # raise NotImplementedError
                self.config.ft_lr = self.config.lr / 100
            # self.config.ft_lr=self.config.lr/1000
            # self.config.ft_lr=self.config.lr
        print('ft_lr:',self.config.ft_lr)
        self.ft_optimizer = optim.Adam([
                                    # {'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                    {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.ft_lr,
                                    weight_decay=1e-5)
        if 'ucy' in self.config.data_dict_path:
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=20, gamma=0.999)
        elif 'gc' in self.config.data_dict_path:
            # self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=10, gamma=0.999)
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=20, gamma=0.99)
        elif 'eth' in self.config.data_dict_path:
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=20, gamma=0.999)
        else:
            # raise NotImplementedError
            self.ft_scheduler = optim.lr_scheduler.StepLR(self.ft_optimizer, step_size=20, gamma=0.999)
        self.log.info(f'(\'ft_lr\', {self.config.ft_lr})')
        print("> Optimizer built!")


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = None)
        self.model = model.to(self.config.device)
        # detailed_param_stats(self.model)
        # if self.config.eval_mode:
        #     self.model.load_state_dict(self.checkpoint['ddpm'])
        if self.config.model_sd_path:
            self.model.load_state_dict(torch.load(self.config.model_sd_path))
        train_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        trainable_params = np.sum([p.numel() for p in train_params])
        self.log.info(f'#Trainable Parameters: {trainable_params}')
        self.log.info("\n")
        print("> Model built!")

    def save_model(self, model, optimizer,scheduler, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        save_name = f'Epoch {epoch}' + '.pkl'
        save_path = os.path.join(save_path, save_name)
        if isinstance(model, torch.nn.DataParallel):
            net_state_dict = model.module.state_dict()
        else:
            net_state_dict = model.state_dict()

        torch.save({
            'model_state_dict':net_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
        }, save_path)

    def _build_data_loader(self):

        if self.config.rebuild_dataset == True:
            if self.config.dataset_type=='timeindex':
                finetune_dataset = DATASET.TimeIndexedPedDataset2()    
            else:
                raise NotImplementedError
            finetune_dataset.load_data(self.config.data_config, config=self.config)
            
            print('number of finetune training dataset: ', len(finetune_dataset.raw_data['train']))
            finetune_dataset.build_dataset(self.config, finetune_flag=(self.config.finetune_trainmode=='multistep'))
            
            with open(self.config.data_dict_path, 'wb') as f:
                dill.dump(finetune_dataset, f, protocol=dill.HIGHEST_PROTOCOL)
        elif self.config.finetune == True:
            with open(self.config.data_dict_path, 'rb') as f:
                finetune_dataset = dill.load(f)
        
        if self.config.finetune == True:
            
            finetune_train_list = finetune_dataset.train_data
            if self.config.finetune_trainmode=='singlestep':
                self.finetune_train_loaders=[]
                assert type(finetune_train_list)==list
                for item in finetune_train_list:
                    self.finetune_train_loaders.append(DataLoader(
                                                item,
                                                batch_size=self.config.batch_size,
                                                num_workers=8,
                                                shuffle =False,
                                                drop_last=True))
            elif self.config.finetune_trainmode=='multistep':
                assert type(finetune_train_list[0])==DATA.ChanneledTimeIndexedPedData
                self.finetune_train_loaders = LOADER.data_loader(finetune_train_list, self.config.batch_size,

                                                self.config.seed, shuffle=False, drop_last=False)

            self.finetune_valid_list = finetune_dataset.valid_data

        return

    # def load_data_all(self):
    #     if self.config.rebuild_dataset == True:
    #         finetune_dataset = DATASET.TimeIndexedPedDataset2()
    #     finetune_dataset.load_data(path = self.config.data_path, begin_time= self.config.begin_time, simulate_time=self.config.simulate_time)


    def _build_offline_scene_graph(self):
        if self.hyperparams['offline_scene_graph'] == 'yes':
            print(f"Offline calculating scene graphs")
            for i, scene in enumerate(self.train_scenes):
                scene.calculate_scene_graph(self.train_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(self.eval_scenes):
                scene.calculate_scene_graph(self.eval_env.attention_radius,
                                            self.hyperparams['edge_addition_filter'],
                                            self.hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")


    def image2world(self, traj, H):
        num_frame, num_people, _ = traj.shape
        H_inv = np.linalg.inv(H)
        image_point = np.zeros_like(traj)
        for frame_idx in range(num_frame):
            frame = traj[frame_idx]
            world_point = np.hstack([frame, np.ones((num_people, 1))]) # m,n,3
            image = H_inv @ world_point.T #3,
            image /= image[2]
            image = image[:2].T  # N,2
            image[:, 1] = image[:, 1] * -1 + 288
            image += [360, 0]
            image_point[frame_idx] = np.round(image).astype(int)


        return image_point

    def draw_image_frame(self, pred, lable):
        pred_p = np.load(pred)
        label_p = np.load(lable)
        H = np.array([[2.84217540e-02, 2.97335273e-03, 6.02821031e+00],
                      [-1.67162992e-03, 4.40195878e-02, 7.29109248e+00],
                      [-9.83343172e-05, 5.42377797e-04, 1.00000000e+00]])

        image_pred = self.image2world(pred_p, H).astype(int)
        image_label = self.image2world(label_p, H).astype(int)


        background = cv2.imread(self.config.bg_image)
        # H_inv = np.linalg.inv(H)
        out_dir = "test_frame"
        os.makedirs(out_dir, exist_ok=True)

        num_frame, num_people, _ = pred_p.shape

        for frame_idx in range(num_frame):

            ppred = image_pred[frame_idx]
            llabel = image_label[frame_idx]

            img = background.copy()
            for ii in range(num_people):
                x, y = ppred[ii]
                xx, yy = llabel[ii]

                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= xx < img.shape[1] and 0 <= yy < img.shape[0] :
                    cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # 红色实心圆
                    cv2.circle(img, (xx,yy), radius=4, color=(0,255,0), thickness=-1)

                # 保存图像
            out_path = os.path.join(out_dir, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(out_path, img)
            print("num frame:",frame_idx)


    def draw_image_person(self, pred, lable):
        pred_p = np.load(pred)
        label_p = np.load(lable)
        H = np.array([[2.84217540e-02, 2.97335273e-03, 6.02821031e+00],
                      [-1.67162992e-03, 4.40195878e-02, 7.29109248e+00],
                      [-9.83343172e-05, 5.42377797e-04, 1.00000000e+00]])

        image_pred = self.image2world(pred_p, H).astype(int)
        image_label = self.image2world(label_p, H).astype(int)


        background = cv2.imread(self.config.bg_image)
        # H_inv = np.linalg.inv(H)
        out_dir = "test_person"
        os.makedirs(out_dir, exist_ok=True)

        num_frame, num_people, _ = pred_p.shape

        for people_idx in range(num_people):

            ppred = image_pred[:, people_idx, :]
            llabel = image_label[:, people_idx, :]

            img = background.copy()
            for ii in range(num_frame):
                x, y = ppred[ii]
                xx, yy = llabel[ii]

                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0] and 0 <= xx < img.shape[1] and 0 <= yy < img.shape[0] :
                    cv2.circle(img, (x, y), radius=4, color=(0, 0, 255), thickness=-1)  # 红色实心圆
                    cv2.circle(img, (xx,yy), radius=2, color=(0,255,0), thickness=-1)

                # 保存图像
            out_path = os.path.join(out_dir, f'person_{people_idx:04d}.jpg')
            cv2.imwrite(out_path, img)
            print("num person:",people_idx)


