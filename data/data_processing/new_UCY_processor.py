import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
def load_homography_matrix(homog_file):
    """ 读取单应性矩阵（Homography Matrix） """
    if not os.path.exists(homog_file):
        raise FileNotFoundError(f"Homography matrix file {homog_file} not found!")
    return np.loadtxt(homog_file)


def apply_homography(x, y, H):
    """ 将像素坐标 (x, y) 映射到真实世界坐标 """
    pixel_coords = np.array([x, y, 1.0]).reshape(3, 1)
    world_coords = np.dot(H, pixel_coords)
    world_coords /= world_coords[2, 0]  # 归一化
    return world_coords[0, 0], world_coords[1, 0]



def load_vsp_file(vsp_file, H=None, time_unit=0.08, time= 0, duratime=0):

    """ 解析 VSP 轨迹文件 """
    if not os.path.exists(vsp_file):
        raise FileNotFoundError(f"VSP file {vsp_file} not found!")

    trajectories = []
    time_range = (int(time), int(time+duratime))
    frame_range = [time_range[0]* 25, time_range[1] * 25]
    to_begin = frame_range[0] / 25 / time_unit
    print("frame_range", frame_range)
    # time_unit = 0.04
    with open(vsp_file, "r") as f:
        num_peds = int(f.readline().split(' ')[0])
        index = 0

        while index < num_peds:
            S = int(f.readline().split(' ')[0])
            # print(S)
            traj = np.zeros([S, 4])
            trajs = []
            for j in range(S):

                traj[j, :] = np.array(f.readline().split(' ')[0:4], dtype=float)
                traj[j , 0], traj[j,1] = apply_homography(traj[j,0], traj[j,1], H)


            index += 1
            begin_frame, end_frame = int(traj[0, 2]), int(traj[-1, 2])
            sample_frame = np.arange(begin_frame, end_frame + 1, time_unit * 25)
            traj_ = np.zeros([len(sample_frame), 3])

            traj_[:, 2] = sample_frame
            try:
                traj_[:, 0] = interp1d(traj[:, 2], traj[:, 0], kind='cubic')(traj_[:, 2])

                traj_[:, 1] = interp1d(traj[:, 2], traj[:, 1], kind='cubic')(traj_[:, 2])

                # traj_[:, 3] = interp1d(traj[:,2], traj[:, 3], kind='cubic')(traj_[:, 2])
            except (ValueError):  # traj_.shape[0] is too less to do high order interpolate
                traj_[:, 0] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 0])
                traj_[:, 1] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 1])

            traj = [(x, y, int(f / time_unit / 25 - to_begin)) for x, y, f in traj_ if ((f >= frame_range[0]) and (f <= frame_range[1]))]
            if (traj):
                trajectories.append(traj)



    return trajectories







def load_obs_file(obs_file, H):
    obs = []
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"VSP file {obs_file} not found!")
    with open(obs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split()

            x,y = float(x), float(y)
            x = x - 360
            y = (y - 288) * -1
            x,y = apply_homography(x,y,H)
            obs.append([x,y])
    return obs

def split_and_ave_trans(bg_road, mask, shop_mask,obs_mask, block_size, H):
    h, w = bg_road.shape  # 获取图像尺寸
    h_blocks = h // block_size  # 计算纵向块数
    w_blocks = w // block_size  # 计算横向块数

    # avg_brightness = np.zeros((h_blocks, w_blocks), dtype=np.float32)
    # new_mask = np.zeros((h_blocks, w_blocks), dtype=np.uint8)
    world_coords_list = []
    shop_list = []
    obs_list = []
    bb = block_size//2
    hh = h // bb  # 计算纵向块数
    ww = w // bb  # 计算横向块数
    for i in range(hh):
        for j in range(ww):
            y_start, y_end = i * bb, (i + 1) * bb
            x_start, x_end = j * bb, (j + 1) * bb
            obs_block = obs_mask[y_start:y_end, x_start:x_end]
            center_x = x_start + bb // 2
            center_y = y_start + bb // 2
            center_x = center_x - 360
            center_y = (center_y - 288) * -1
            world_x, world_y = apply_homography(center_x, center_y, H)

            # 是道路部分的区域，超过一半的像素输入road,进行计算亮度值
            if np.sum(obs_block == 255) > (bb * bb) / 2:
                obs_list.append((world_x, world_y))



    for i in range(h_blocks):
        for j in range(w_blocks):
            # 计算块的坐标范围
            y_start, y_end = i * block_size, (i + 1) * block_size
            x_start, x_end = j * block_size, (j + 1) * block_size

            # 计算该区域的平均亮度
            block = bg_road[y_start:y_end, x_start:x_end]
            avg_value = np.mean(block)/255.0
            # avg_brightness[i, j] = np.mean(block)

            # 计算该区域的 mask 投票
            mask_block = mask[y_start:y_end, x_start:x_end]
            shop_block = shop_mask[y_start:y_end, x_start:x_end]
            # 计算该区域的世界坐标（取块的中心点）
            center_x = x_start + block_size // 2
            center_y = y_start + block_size // 2
            center_x = center_x - 360
            center_y = (center_y - 288) * -1
            world_x, world_y = apply_homography(center_x, center_y, H)

            #是道路部分的区域，超过一半的像素输入road,进行计算亮度值
            if np.sum(mask_block == 255) > (block_size * block_size) / 2:
                world_coords_list.append((world_x, world_y, avg_value))

            if np.sum(shop_block == 255) > (block_size * block_size) / 2:
                shop_list.append((world_x, world_y))

    return  np.array(world_coords_list), np.array(shop_list), np.array(obs_list)





def read_image(image_file,H, block_size):
    bg_path = os.path.join(image_file, "bg.png")
    road_path = os.path.join(image_file, "road.png")
    shop_path = os.path.join(image_file, "shop.png")
    obs_path = os.path.join(image_file, "obs.png")
    bg = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE) # 576,720灰度图，越亮的地方数值越高
    road_mask = cv2.imread(road_path, cv2.IMREAD_GRAYSCALE)
    shop_mask = cv2.imread(shop_path, cv2.IMREAD_GRAYSCALE)
    obs_mask = cv2.imread(obs_path, cv2.IMREAD_GRAYSCALE)

    #进行分块 得到新的亮度和mask


    return split_and_ave_trans(bg, road_mask,shop_mask,obs_mask, block_size, H)






def main():
    parser = argparse.ArgumentParser(description="Load and interpolate UCY dataset trajectories")
    parser.add_argument("--vsp_file", type=str, default="/mnt/d/SPD_2025/data/UCY/students03/annotation.vsp", help="Path to the VSP file")
    parser.add_argument("--homog_file", type=str, default="/mnt/d/SPD_2025/data/UCY/students03/H.txt", help="Path to the homography matrix file")
    parser.add_argument("--fps", type=float, default=2.5, help="Frame rate for interpolation")
    # parser.add_argument("--obs", type=str, default="E:/test/Dataset/UCY/zara01/static.txt")
    parser.add_argument("--savename", type=str, default="/mnt/d/SPD_2025/data_origin/UCY_dataset/UCY/ucy_students03_brights_shop")
    parser.add_argument("--image_file", type=str, default="/mnt/d/SPD_2025/data/UCY/students03")
    parser.add_argument("--block_size", type=int, default=36)
    parser.add_argument("--time", type=int, default=0) # 0 54 108 162
    parser.add_argument("--duratime", type=int, default=54)
    args = parser.parse_args()
    time_range = [int(args.time), int(args.time + args.duratime)]
    time_unit = 0.08
    savename = args.savename + f"UCY_Dataset_time{time_range[0]}-{time_range[1]}_timeunit{time_unit:.2f}"
    print("savename", savename)
    # pixel_pts = np.array([
    #     [639,411],
    #     [634,390],
    #
    #     [566,390],
    #     [562, 360],
    #     [414, 391],
    #     [386,431],
    #     [300,415],
    #     [206, 268]
    #
    #
    # ], dtype = np.float32)
    # pixel_pts = pixel_pts - [360,0]
    # pixel_pts[:,1] = (pixel_pts[:,-1]-288)*-1
    # world_pts = np.array([
    #     [13.4487205051, 3.93788669527],
    #     [13.3434879503, 4.43907227467],
    #     [11.9123252048, 4.43907227467],
    #     [11.828139161, 5.15505167381],
    #     [8.7132555385, 4.4152062947],
    #     [8.12395323155, 3.46056709585],
    #     [6.31395328877, 3.84242277539],
    #     [4.33558125829, 7.35072183117]
    # ], dtype=np.float32)
    # H, status = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC)
    # print(H)
    # print(H)
    # print(status)
    # 读取 Homography 矩阵
    # H = load_homography_matrix(args.homog_file)
    #
    # # 读取 VSP 文件并转换坐标
    H = np.array([[2.84217540e-02, 2.97335273e-03, 6.02821031e+00],
                  [-1.67162992e-03, 4.40195878e-02, 7.29109248e+00],
                  [-9.83343172e-05, 5.42377797e-04, 1.00000000e+00]])
    brights,shops,obstacles = read_image(args.image_file, H, args.block_size)
    trajectories = load_vsp_file(args.vsp_file, H, time_unit=0.08, time=args.time, duratime=args.duratime)

    ped_num = len(trajectories)
    destinations = []
    for i, arr in enumerate(trajectories):
        destinations.append( [arr[-1][0], arr[-1][1], arr[-1][2]])


    # obstacles = load_obs_file(args.obs, H)
    max_length = int(max(arr[-1][-1] for arr in trajectories) + 1)
    # max_length = 0
    # for arr in trajectories:
    #     max_length = int(max_length, arr[-1][-1] + 1)


    meta_data = {
        "time_unit": 0.08,
        "version": "v2.2",
        "max_time": max_length,
        "source": "UCY dataset",
        "ped_num": ped_num,
        "H":H
    }

    data = np.array((meta_data,trajectories, destinations,obstacles, brights, shops),dtype=object)
    # trajectories = torch.tensor(trajectories)
    # 可视化轨迹
    np.save(savename + ".npy", data)

    # save_data = Raw_data()
    # save_data.load_trajectories()

    # padded_trajectories = [pad_sequence(arr, max_length) for arr in trajectories]
    # padded_tensor = torch.tensor(padded_trajectories)
    # # padded_tensor = padded_tensor.permute(1,0,2) # T,
    # mask = torch.zeros((148, max_length), dtype=torch.bool)
    # for i, arr in enumerate(trajectories):
    #     begin = int(arr[0][2])
    #     end = int(arr[-1][2] + 1)
    #     mask[i,begin:end] = True
    #
    # np.savetxt('zara1.txt', trajectories, fmt='%10f', delimiter=',')


if __name__ == "__main__":
    main()
