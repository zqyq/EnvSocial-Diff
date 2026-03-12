import numpy as np
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,os
import pandas as pd
sys.path.append(os.path.dirname(__file__) + os.sep + '../../')
from data.dataset import RawData
from utils.visualization import state_animation
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
from scipy.interpolate import CubicSpline

def get_args():
    parser = argparse.ArgumentParser(description='UCY dataset processor')
    parser.add_argument('-i', '--input', type=str, default="../raw_ucy/",
                        help='input file path')
    parser.add_argument("--static", type=str, default="../UCY/")
    parser.add_argument('-o', '--output', type=str, default='../../data_origin/UCY_dataset/raw/',
                        help='output file path')
    parser.add_argument('-d', '--duration', type=float, default='54',
                        help='length of time snippet to save')
    parser.add_argument('-t', '--time', type=float, default='0',
                        help='begining of time snippet to save. Recommendation parameter: 760, 1000, 1100, 1280, 1560')
    # parser.add_argument('-r', '--range', action='store_true',
    #                     help='whether to limit range to [[5, 15], [25, 35]]')
    parser.add_argument('-v', '--visulization', action='store_true',
                        help='whether to generate visulization animation')
    args, unknown = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = get_args()
    time_range = (int(args.time), int(args.time + args.duration))
    frame_range = [time_range[0] * 25, time_range[1] * 25]
    time_unit = 0.4
    # savename = args.output + f"UCY_Dataset_time{time_range[0]}-{time_range[1]}_timeunit{time_unit:.2f}"

    meta_data = {
        "time_unit": time_unit,
        "version": "v2.2",
        "begin_time": time_range[0],
        "source": "UCY dataset"
    }

    trajectories = []
    print('Processing...')
    with open(args.input + "crowds_zara01.txt") as f:
        for line in f:
            trajs = line.split()
            t = float(trajs[0])
            id = float(trajs[1])
            x = float(trajs[2])
            y = float(trajs[3])
            trajectories.append((id, t, x, y))

    # 转换为 Tensor
    trajectories = torch.tensor(trajectories, dtype=torch.float32)
    scale_factor = 10  # **每两个原始点之间插入 9 个数据**

    # 计算唯一的 ID 和 时间
    ids, id_indices = torch.unique(trajectories[:, 0], return_inverse=True)
    times, time_indices = torch.unique(trajectories[:, 1], return_inverse=True)
    num_ids = len(ids)
    num_times = len(times)
    t_min, t_max = times.min().item(), times.max().item()
    num_new_times = (len(times) - 1) * scale_factor + 1
    t_new = np.arange(t_max+1)

    # # 创建存储轨迹的 Tensor，并填充 NaN
    coords = torch.full((num_ids, int(t_max)//10 +1 , 2), float('nan'), dtype=torch.float32)

    # 修正索引数据类型
    id_indices = id_indices.to(dtype=torch.long)
    time_indices = time_indices.to(dtype=torch.long)

    # 填充坐标
    coords[id_indices, time_indices, 0] = trajectories[:, 2]  # x 坐标
    coords[id_indices, time_indices, 1] = trajectories[:, 3]  # y 坐标

    # 插值
    destinations = np.full((num_ids, 3), np.nan, dtype=np.float32)
    interpolated_data = np.full((num_ids, int(t_max)+1, 2), np.nan, dtype=np.float32)
    for n, pid in enumerate(ids):
        # 获取当前人的轨迹数据
        mask = trajectories[:, 0] == pid
        person_traj = trajectories[mask]

        # 按时间排序
        person_traj = person_traj[person_traj[:, 1].argsort()]

        # 原始时间点和坐标
        t_original = person_traj[:, 1].numpy()
        xy_original = person_traj[:, 2:].numpy()

        if len(t_original) < 2:
            continue  # 少于两个时间点无法插值，跳过

        # 插值 X 和 Y 坐标
        for c in range(2):  # 遍历 x 和 y
            interp_kind = 'cubic' if len(t_original) >= 4 else 'linear'
            interpolator = interp1d(t_original, xy_original[:, c], kind=interp_kind, bounds_error=False,
                                    fill_value=np.nan)
            interpolated_values = interpolator(t_new)

            # 只填充有效插值数据，默认 NaN 位置不变
            interpolated_data[n, :, c] = interpolated_values

        valid_indices = np.where(~np.isnan(interpolated_data[n, :, 0]))[0]  # 找到非 NaN 的索引
        if len(valid_indices) > 0:
            last_valid_idx = valid_indices[-1]  # 取最后一个有效索引
            destinations[n][:2]= interpolated_data[n, last_valid_idx, :]  # 存储最终目的地
            destinations[n][2] = last_valid_idx

    # 转换回 Torch Tensor（可选）
    interpolated_data = torch.tensor(interpolated_data, dtype=torch.float32)
    t_new = torch.tensor(t_new)

    t_new_expanded = t_new.unsqueeze(0).unsqueeze(-1)  # (1, 9011, 1)

    # 复制到 (148, 9011, 1)，让每个人都有相同的时间维度
    t_new_broadcasted = t_new_expanded.expand(interpolated_data.shape[0], -1, -1)  # (148, 9011, 1)
    t_new_broadcasted = t_new_broadcasted.int()
    interpolated_data = torch.cat([interpolated_data,t_new_broadcasted], dim=-1)


    # x,y,t  destination x,y,t
    savename = args.output + "UCY_all_"

    data = np.array((meta_data, interpolated_data, destinations,[],int(t_max)), dtype=object)
    np.save(savename + ".npy", data)
    print(f'Saved processed data to {savename + ".npy"}\n')

    saved_data = RawData()
    saved_data.load_trajectory_data(savename + ".npy")

    if(args.visulization):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.subplot()
        ax.grid(linestyle='dotted')
        ax.set_aspect(1.0, 'datalim')
        ax.set_axisbelow(True)
        ax.set_xlim(-5, 20)
        ax.set_ylim(-5, 16.5)
        video = state_animation(ax, saved_data, show_speed=False, movie_file=savename+".gif")
