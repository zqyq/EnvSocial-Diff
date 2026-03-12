# -*- coding: utf-8 -*-
import cv2
import math
import torch
import torch.utils
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pdb
import  json
from pathlib import Path
from typing import List, Dict,Any
from models.image import *
from models.Text import *
from models.PositionAware import *
from models.Fusion import *
from PIL import Image
class RawData(object):
    """
    Attributes:
        position: (t, N, 2)
        velosity: (t, N, 2)
        acceleration: (t, N, 2)
        destination: (t, N, 2)
        waypoints: (D, N, 2)
        dest_num: (N): the number of waypoints of each pedestrian
        dest_idx: (t, N): the index of the waypoint a user is heading at time t
        obstacles: (M * 2)
        mask_a: (t, N), If someone is not in the frame, mask_a is zero too.
        mask_v: (t, N)
        mask_p: (t, N):****mask_**1
        num_steps: total number of time steps
        num_pedestrains: total number of pedestrains
        time_unit
    Notes：
        If an agent is not in the frame, then its position and destination is
        assigned as 'nan'
    """

    def __init__(
            self,
            position=torch.tensor([]),
            velocity=torch.tensor([]),
            acceleration=torch.tensor([]),
            destination=torch.tensor([]),
            waypoints=torch.tensor([]),
            obstacles=torch.tensor([]),
            mask_p=torch.tensor([]),
            mask_v=torch.tensor([]),
            mask_a=torch.tensor([]),
            meta_data=None):
        super(RawData, self).__init__()
        if meta_data is None:
            self.meta_data = dict()
        else:
            self.meta_data = meta_data
            self.time_unit = meta_data['time_unit']
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.destination = destination
        self.grid_origin = [5.8982, 6.5514]
        self.waypoints = waypoints
        # self.device = "cuda:2"

        # self.image_obs = "/mnt/d/SPD_2025/data/UCY/students03/obs.png"
        # self.image_bg = "/mnt/d/SPD_2025/data/UCY/students03/bg.png"
        # self.image_bg_gray = cv2.imread(self.image_bg)
        # self.image= self.image_bg
        # self.mask_obs = cv2.imread(self.image_obs, 0) #只读灰度图
        # inv_mask = cv2.bitwise_not(self.mask_obs)
        # self.foreground = cv2.bitwise_and(self.image_bg_gray, self.image_bg_gray, mask=inv_mask)
        # # hsv = cv2.cvtColor(self.foreground, cv2.COLOR_BGR2HSV)
        # hsv = cv2.cvtColor(self.image_bg_gray, cv2.COLOR_BGR2HSV)
        # self.brights = hsv[:,:,2]
        # self.brights = self.brights.astype(float)
        # self.brights[self.mask_obs == 255] =  np.nan

        # X范围: [5.8982, 28.5104]
        # Y范围: [6.5514, 25.3242]
        self.world_min_x = 5.8982
        self.world_max_x = 28.5104
        self.world_min_y = 6.5514
        self.world_max_x = 25.3242
        self.world_size = [22.6122, 18.7729]
        self.grid_size = 0.1

        self.obstacles = obstacles
        self.mask_p = mask_p
        self.mask_v = mask_v
        self.mask_a = mask_a
        if position.shape[-1] > 0:
            self.num_steps = position.shape[0]
            self.num_pedestrians = position.shape[1]
            self.destination_flag = torch.zeros(self.num_pedestrians, dtype=int)
        if waypoints.shape[-1] > 0:
            self.num_destinations = waypoints.shape[0]
            self.dest_idx = self.get_waypoints_index_matrix()
            self.dest_num = self.get_dest_num()

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')

    def get_waypoints_index_matrix(self):
        pass

    def get_dest_num(self):
        pass
    def mask_to_occpuancy_grid(self,grid_size=0.2):
        # 读取图像和元数据
        H = self.meta_data['H']  # 齐次变换矩阵（图像坐标 -> 世界坐标）
        mask_img = cv2.imread(self.image_obs, cv2.IMREAD_GRAYSCALE)

        # 二值化处理（0或1）
        _, mask_binary = cv2.threshold(mask_img, 127, 1, cv2.THRESH_BINARY)

        # 获取图像尺寸和网格划分
        height, width = mask_binary.shape
        grid_x_pixels = int(grid_size * width)  # 网格的像素宽度
        grid_y_pixels = int(grid_size * height)  # 网格的像素高度

        occupied_coords = []

        # 遍历每个网格
        for i in range(0, height, grid_y_pixels):
            for j in range(0, width, grid_x_pixels):
                # 检查当前网格是否被占据
                grid_patch = mask_binary[i:i + grid_y_pixels, j:j + grid_x_pixels]
                if np.any(grid_patch > 0):  # 如果有障碍物
                    # 计算网格中心的世界坐标
                    x = (j + grid_x_pixels // 2) - 360
                    y = (i + grid_y_pixels // 2-288) * -1
                    center_pixel = np.array([x, y, 1.0])
                    # center_pixel
                    world_homo = H @ center_pixel
                    world_x, world_y = world_homo[:2] / world_homo[2]
                    occupied_coords.append([world_x, world_y])

        # 转换为 Tensor（形状为 [n_occupied, 2]）
        return torch.tensor(occupied_coords, dtype=torch.float32, device=self.device)




    # def average_pool_brights(self, brights, grid_size):
    #     grid = {}
    #     for x,y, b in brights:
    #         gx = int(x // grid_size)
    #         gy = int(y // grid_size)
    #         key = (gx, gy)
    #         if key not in grid:
    #             grid[key] = []
    #         grid[key].append(b)
    #
    #     pooled = []
    #     for (gx, gy), b_list in grid.items():
    #         avg_b = torch.mean(torch.tensor(b_list))
    #         if avg_b.isnan():
    #             continue
    #         cx = (gx + 0.5) * grid_size
    #         cy = (gy + 0.5) * grid_size
    #         pooled.append([cx, cy, avg_b])
    #
    #     return np.array(pooled)

    def stat_pool_brights(self, brights, grid_size, device=None):
        H, W = brights.shape
        self.device = device
        # 创建像素坐标网格
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        brights= torch.tensor(brights, device= self.device)
        # 将坐标和亮度展平
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        xx = torch.tensor(xx, device=self.device)
        yy = torch.tensor(yy, device=self.device)
        b = brights.reshape(-1)
        points = torch.stack([xx, yy, b], dim=1)
        grid = {}
        for x, y, b in points:
            gx = int(x.item() // grid_size)
            gy = int(y.item() // grid_size)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append(b.item())

        pooled = []
        for (gx, gy), b_list in grid.items():
            if not b_list:
                continue
            b_tensor = torch.tensor(b_list, dtype=torch.float32)
            avg_b = torch.mean(b_tensor)
            max_b = torch.max(b_tensor)
            min_b = torch.min(b_tensor)

            if torch.isnan(avg_b):
                continue  # skip invalid values

            # Grid cell center position
            cx = (gx + 0.5) * grid_size
            cy = (gy + 0.5) * grid_size
            # x = cx - 360
            # y = (cy - 288) * -1
            # center_pixel = np.array([x, y, 1.0])
            # # center_pixel
            # world_homo = self.meta_data["H"] @ center_pixel
            # world_x, world_y = world_homo[:2] / world_homo[2]
            # pooled.append([world_x, world_y, avg_b.item()/255.0, max_b.item()/255.0, min_b.item()/255.0])
            pooled.append([cx, cy, avg_b.item() / 255.0, max_b.item() / 255.0, min_b.item() / 255.0])
        return np.array(pooled)


    def load_json(self,json_path:str)->Dict[str, Any]:
        """
        读取一个障碍物jsonw文件，返回结构化字典
        """
        json_path = Path(json_path)
        with open(json_path, 'r') as f:
            data = json.load(f)
        id_list = []
        bbox_list = []
        desc_list = []
        image_list = []
        id_list2 = []
        bbox_list2 = []
        desc_list2= []
        image_list2 = []
        info = data["image_information"]
        for obj in data["obstacles"]:
            x, y, w, h, theta = obj["bbox"]
            category = obj["category"]
            # cat_id = obj["category"]  # -1 表示未知类别

            bbox_list.append([x, y, w, h, theta])
            desc_list.append(obj["description"])
            id_list.append(category)
            image_list.append(obj["image"])
        for obj in data["interest"]:
            x, y, w, h, theta = obj["bbox"]
            category = obj["category"]
            # cat_id = obj["category"]  # -1 表示未知类别

            bbox_list2.append([x, y, w, h, theta])
            desc_list2.append(obj["description"])
            id_list2.append(category)
            image_list2.append(obj["image"])

        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)  # (N, 6)
        bbox_tensor2 = torch.tensor(bbox_list2, dtype=torch.float32)  # (N, 6)
        border = torch.tensor(data["boundary_line"]).reshape(-1, 2)

        return bbox_tensor, desc_list, id_list,info, bbox_tensor2, desc_list2,border,image_list,image_list2


    def get_env_features(self,args):
        self.image_base = ImageBase(args.device, args.image_weights_path)
        if args.Image_encoder == "ResNet":
            # self.image_encoder = ResNet(weights_path=args.image_weights_path, device=args.device)
            self.image_encoder = ResNet2(weights_path=args.image_weights_path, device=args.device)
            self.image_global = ResNetGlobal(weights_path=args.image_weights_path,device=args.device)
        if args.Text_encoder == "BERT":
            self.text_encoder = BERTEncoder(model_path=args.text_weights_path, device=args.device)
        # self.image_text_fusion = LocalFusion(device=args.device, text_dim=args.text_dim, image_dim=args.image_dim,
        #                                      out_dim=args.out_dim, dropout=args.dropout)
        # self.image_text_fusion = self.image_text_fusion.to(args.device)
        self.image_big = self.image_global(args.bg_image).detach().to(args.device)  #49, 2048
        self.text_big = self.text_encoder(self.info).detach().to(args.device)
        combined_obs = [" This is a  " + a + '. ' + b for a, b in zip(self.id, self.description)]


        self.text_obs_emb = self.text_encoder(combined_obs).detach().to(args.device)  # M,768
        if len(self.in_destcrip)!=0:
            self.text_in_emb = self.text_encoder(self.in_destcrip).detach().to(args.device)  # N,768
            self.image_in_emb = self.image_encoder(self.images2).detach().to(args.device)
        else:
            self.text_in_emb = None
            self.image_in_emb = None
        self.image_obs_emb = self.image_encoder(self.images1).detach().to(args.device)

        # fusion_emb = self.image_text_fusion(text_emb=texts_emb, image_emb=images_emb) #512
        # return text_obs_emb,text_in_emb, image_obs_emb,image_in_emb, text_big, image_big


    def load_trajectory_data(self, data_path,config):
        """
        Process the raw data to get velocity and acceleration. If an agent is
        not in the frame, then its position is assigned as 'nan'

        Dataset format description: see
        https://tsingroc-wiki.atlassian.net/wiki/spaces/TSINGROC/pages/2261120#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%AD%98%E5%82%A8%E6%A0%BC%E5%BC%8F%E8%AF%B4%E6%98%8E

        """
        self.devide=config.device

        device = torch.device(config.device)
        print(f"Loading from '{data_path}'...")
        out_of_bound = torch.tensor(float('nan'))

        data = np.load(data_path, allow_pickle=True)
        assert ('version' in data[0] and data[0]['version'] == 'v2.2'), f"'{data_path}' is out of date."

        if 'gc' in config.data_dict_path:
            meta_data, trajectories, destinations, obstacles = data
            meta_data["H"] = np.array([[3.54477751e-02, 1.73477252e-02, -1.82112170e+01],
                  [6.03523702e-04, -5.58259424e-02, 5.12654156e+01],
                  [1.00205219e-05, 1.25487966e-03, 1.00000000e+00]])
        # elif 'eth' in config.data_dict_path:
        #     meta_data, trajectories, destinations, obstacles = data
        #     meta_data["H"] = np.array([[2.84217540e-02, 2.97335273e-03, 6.02821031e+00],
        #                   [-1.67162992e-03, 4.40195878e-02, 7.29109248e+00],
        #                   [-9.83343172e-05, 5.42377797e-04, 1.00000000e+00]])
        elif "zara02" in config.data_dict_path:
            meta_data, trajectories, destinations, obstacles = data
            meta_data["H"] = np.array([[-2.595651699999999840e-02, -5.157280400000000145e-18, 7.838868099999996453e+00],
                                [-1.095387399999999886e-03, 2.166433000000000247e-02, 5.566045600000004256e+00],
                                [1.954012500000000506e-20, 4.217141000000002596e-19, 1.000000000000000444e+00]])
        elif 'citystreet' in config.data_dict_path:
            meta_data, trajectories, destinations, obstacles = data

            meta_data["H"] = np.array([[0.0001714141245861959, -0.002523602874974581, 1.6630292472203838],
                                       [0.0008291509119256762, -0.00025926739732050276, -1.0121581548759886],
                                       [-1.025482980223859e-08, 1.1168721358491362e-07, -2.33573253766503e-05]])
        elif "eth" in config.data_dict_path:
            meta_data, trajectories, destinations, obstacles = data
            meta_data["H"] = np.array([[2.8128700e-02 ,  2.0091900e-03,  -4.6693600e+00],
                                        [8.0625700e-04 ,  2.5195500e-02 , -5.0608800e+00],
                                        [3.4555400e-04 ,  9.2512200e-05 ,  4.6255300e-01
                                        ]])
        else:
            meta_data, trajectories, destinations, obstacles, brights, shops = data

        self.meta_data = meta_data

        if config.use_json:
            self.box, self.description, self.id, self.info,self.in_box,self.in_destcrip,self.border,self.images1,self.images2 = self.load_json(config.json_path)
            # self.text_emb, self.image_emb, self.text_big, self.image_big = self.get_env_features(config)
            self.get_env_features(config)
        else:
            self.box= torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.description = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.id = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.info =torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.text_emb =torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.image_emb =torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.text_big =torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
            self.image_big =torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)
        # obstacles = torch.tensor(obstacles, dtype=torch.float)

        # if obstacles.shape[-1] == 0:
        # obstacles = self.mask_to_occpuancy_grid(grid_size=self.grid_size)
        obstacles = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=device)

        obstacles = torch.tensor(obstacles, device=device)
        self.num_steps = int(max([u[-1][-1] for u in trajectories]) + 1)
        self.num_pedestrians = int(len(trajectories))
        self.num_destinations = max([len(u) for u in destinations])
        position = torch.zeros((self.num_steps, self.num_pedestrians, 2),device=device)
        velocity = torch.zeros((self.num_steps, self.num_pedestrians, 2),device=device)
        acceleration = torch.zeros((self.num_steps, self.num_pedestrians, 2),device=device)
        mask_p = torch.zeros((self.num_steps, self.num_pedestrians),dtype=torch.bool,device=device)
        mask_v = torch.zeros((self.num_steps, self.num_pedestrians),dtype=torch.bool,device=device)
        mask_a = torch.zeros((self.num_steps, self.num_pedestrians),dtype=torch.bool,device=device)
        for i, traj in enumerate(tqdm(trajectories)):
            for x, y, t in traj:
                i= int(i)
                t = int(t)
                position[t, i, :] = torch.tensor([x, y])
                mask_p[t, i] = 1
                mask_v[t, i] = 1
                mask_a[t, i] = 1
            mask_v[t, i] = 0
            mask_a[t, i] = 0
            if t >= 1:
                mask_a[t - 1, i] = 0

        assert (not (position.isnan().any())), "ValueError: Find nan in raw data. Raw data should not contain" \
                                               "any nan values! "

        # self.image = config.image
        destination = torch.zeros((self.num_steps, self.num_pedestrians, 2),device=device)
        waypoints = torch.zeros((self.num_destinations, self.num_pedestrians, 2),device=device) + out_of_bound
        dest_idx = torch.zeros((self.num_steps, self.num_pedestrians), dtype=torch.long,device=device)

        #new
        # brights = self.brights
        # H,W = brights.shape
        # yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        # xx = xx -360
        # yy = (yy - 288) * -1
        # points_img = np.stack([xx,yy], axis=-1). reshape(-1,1,2).astype(np.float32)
        # points_world = cv2.perspectiveTransform(points_img, self.meta_data["H"].astype(np.float32))
        # points_world = points_world.reshape(-1, 2)
        # points_world = torch.tensor(points_world, device=self.device)
        # brights = torch.tensor(brights, device=self.device)
        # brights = brights.reshape(-1,1)
        # self.brights = torch.cat((points_world, brights), dim=-1)
        # self.brights = self.average_pool_brights(self.brights, grid_size=self.grid_size)



        image = Image.open(config.bg_image).convert("HSV")
        image = np.array(image)
        self.brights = image[...,-1]
        if 'gc' in config.data_dict_path:

            self.brights = self.stat_pool_brights(self.brights, grid_size=config.bright_grid, device=config.device)#96 220
        else:
            self.brights = self.stat_pool_brights(self.brights, grid_size=config.bright_grid, device=config.device)
        self.brights = torch.tensor(self.brights, device=self.device)
        # self.brights = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=config.device)
        self.shops = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float, device=config.device)

        # self.shops = torch.tensor(shops,device=config.device)
        dest_num = torch.tensor([1 for relays in destinations])



        if  'ucy' in config.data_dict_path:
            if "zara" in config.data_dict_path:
                for i, relays in enumerate(destinations):
                    relays = torch.tensor(relays)
                    d = relays[:, 0:2]
                    t = relays[:, 2].type(torch.int)
                    waypoints[:d.shape[0], i, :] = d
                    j = -1  # If len(relays) == 1, the loop below will not be executed, and an error will be reported without this statement.
                    for j in range(d.shape[0] - 1):
                        destination[t[j]:t[j + 1], i, :] = d[j]
                        dest_idx[t[j]:t[j + 1], i] = j
                    destination[t[j + 1]:, i, :] = d[j + 1]
                    dest_idx[t[j + 1]:, i] = j + 1
            else:
                for i, relays in enumerate(destinations):
                    relays = torch.tensor(relays)
                    d = relays[0:2]
                    t = relays[2].type(torch.int)
                    waypoints[:d.shape[0], i, :] = d
                    j = -1  # If len(relays) == 1, the loop below will not be executed, and an error will be reported without this statement.
                    # for j in range(d.shape[0] - 1):
                    #     destination[t[j]:t[j + 1], i, :] = d[j]
                    #     dest_idx[t[j]:t[j + 1], i] = j
                    destination[t:, i, :] = d
                    dest_idx[t:, i] = j + 1
        else:
            for i, relays in enumerate(destinations):
                relays = torch.tensor(relays)
                d = relays[:, 0:2]
                t = relays[:, 2].type(torch.int)
                waypoints[:d.shape[0], i, :] = d
                j = -1  # If len(relays) == 1, the loop below will not be executed, and an error will be reported without this statement.
                for j in range(d.shape[0] - 1):
                    destination[t[j]:t[j + 1], i, :] = d[j]
                    dest_idx[t[j]:t[j + 1], i] = j
                destination[t[j + 1]:, i, :] = d[j + 1]
                dest_idx[t[j + 1]:, i] = j + 1
        #old
        # brights = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float)
        # shops = torch.tensor([[1e4, 1e4], [1e4 + 1, 1e4 + 1]], dtype=torch.float)
        # self.brights = torch.tensor(brights)
        # self.shops = torch.tensor(shops)
        # dest_num = torch.tensor([len(relays) for relays in destinations])
        # for i, relays in enumerate(destinations):
        #     relays = torch.tensor(relays)
        #     d = relays[:, 0:2]
        #     t = relays[:, 2].type(torch.int)
        #     waypoints[:d.shape[0], i, :] = d
        #     j = -1  # If len(relays) == 1, the loop below will not be executed, and an error will be reported without this statement.
        #     for j in range(d.shape[0] - 1):
        #         destination[t[j]:t[j + 1], i, :] = d[j]
        #         dest_idx[t[j]:t[j + 1], i] = j
        #     destination[t[j + 1]:, i, :] = d[j + 1]
        #     dest_idx[t[j + 1]:, i] = j + 1

        destination[mask_p == 0] = out_of_bound
        # destination[mask_p == 0] = 0
        # dest_idx[mask_p == 0] = out_of_bound  # in this way we can directly use dest_idx to slice dest
        position[mask_p == 0] = out_of_bound
        # position[mask_p == 0] = 0
        velocity = torch.cat(
            (position[1:, :, :], position[-1:, :, :]), 0) - position
        velocity /= meta_data['time_unit']
        velocity[mask_v == 0] = 0
        acceleration = torch.cat(
            (velocity[1:, :, :], velocity[-1:, :, :]), 0) - velocity
        acceleration /= meta_data['time_unit']
        acceleration[mask_a == 0] = 0

        assert (not (velocity.isnan().any())), f"find nan in velocity."
        assert (not (acceleration.isnan().any())), f"find nan in acceleration."



        self.position, self.velocity = position, velocity
        self.position_desc = [position.masked_fill(mask_p.unsqueeze(-1), 0).mean(),
                              position.masked_fill(mask_p.unsqueeze(-1), 0).std()
                              ]
        self.velocity_desc = [velocity.masked_fill(mask_v.unsqueeze(-1), 0).mean(),
                              velocity.masked_fill(mask_v.unsqueeze(-1), 0).std()
                              ]
        self.acceleration_desc = [acceleration.masked_fill(mask_a.unsqueeze(-1), 0).mean(),
                                  acceleration.masked_fill(mask_a.unsqueeze(-1), 0).std()
                                  ]
        self.acceleration, self.destination = acceleration, destination
        self.waypoints, self.dest_idx, self.dest_num = waypoints, dest_idx, dest_num
        self.obstacles, self.mask_v, self.mask_a = obstacles, mask_v, mask_a
        self.mask_p = mask_p
        self.destination_flag = torch.zeros(self.num_pedestrians, dtype=int)
        self.time_unit = meta_data['time_unit']
        # self.ped_num_each_frame = mask_p.

    def get_frame(self, f: int) -> dict:
        """Get the data of frame f in dictionary format.

        Return:
            - "position": (N, 2)
            - "velocity": (N, 2)
            - "acceleration": (N, 2)
            - "destination": (N, 2), include current destination for ped N
            - "destinations": (R, N, 2), include all R destinations for ped N
            - "destination_flag": (N), include index of current destination for ped N
            - "mask_p": (N)
            - "num_destinations": R
            - "num_pedestrians": N
            - "num_pedestrians": T
            - "meta_data": dict
        """
        frame = {
            "position": self.position[f, :, :],
            "velocity": self.velocity[f, :, :],
            "acceleration": self.acceleration[f, :, :],
            "destination": torch.stack(
                [self.waypoints[int(self.destination_flag[i]), i, :] for i in range(self.num_pedestrians)], dim=0),
            "destinations": self.waypoints,
            "destination_flag": self.destination_flag,
            "num_destinations": self.num_destinations,
            "obstacles": self.obstacles,
            "id": self.id,
            "description": self.description,
            "info": self.info,
            "box": self.box,
            "num_pedestrians": self.num_pedestrians,
            "num_steps": self.num_steps,
            "mask_p": self.mask_p[f, :],
            "meta_data": self.meta_data,
            "shops": self.shops,
            "text_obs_emb": self.text_obs_emb,
            "text_in_emb": self.text_in_emb,
            "image_obs_emb": self.image_obs_emb,
            "image_in_emb": self.image_in_emb,
            # "image_emb": self.image_emb,
            "image_big": self.image_big,
            "text_big": self.text_big,
            "image": self.image_bg,
            "mask_obs": self.mask_obs,
            "brights": self.brights
        }



        return frame

    def get_current_frame(self) -> dict:
        '''Get the data of current frame in dictionary format.'''
        return self.get_frame(self.num_steps)

    def add_frame(self, frame: dict) -> None:
        """Add a frame discribed in dictionary format to data.

        Input:
            - "position": (N, 2)
            - "velocity": (N, 2)
            - "acceleration": (N, 2)
            - "destinations": (R, N, 2).
            - "destination_flag": (N).
            - "mask_p": (N).
            - "num_destinations": D
            - "num_pedestrians": N+dN
            - "meta_data": dict

            - "add_position": (dN, 2)
            - "add_velocity": (dN, 2)
            - "add_acceleration": (dN, 2)
            - "add_destination": (D', dN, 2)

        In this function, we first add new data of raw N pedestrians(e.g. position, saved in frame["position"]). And then, if dN > 0(i.e. frame['num_pedestrians'] > self.num_pedestrians), call self.add_pedestrians() to add data of new dN pedestrians(saved in frame["add_position"]).
        """
        nan = torch.tensor(float('nan'), device=self.position.device)

        # Add data for all attributes who has a time dimention.
        self.num_steps += 1
        self.position = torch.cat((self.position, frame["position"].unsqueeze(0)), dim=0)
        self.velocity = torch.cat((self.velocity, frame["velocity"].unsqueeze(0)), dim=0)
        self.acceleration = torch.cat((self.acceleration, frame["acceleration"].unsqueeze(0)), dim=0)
        self.destination = torch.cat(
            (self.destination, nan + torch.zeros(1, self.num_pedestrians, 2, device=nan.device)), dim=0)

        # A pedestrian whose current destination is nan or current destination's index equals num_destinations has arrived its final destination, so set its masks to zero.
        arrived_final_destination = torch.tensor([(frame['destination_flag'][i] == frame[
            'num_destinations'] or torch.any(torch.isnan(frame['destinations'][frame['destination_flag'][i], i, :])))
                                                  for i in range(self.num_pedestrians)])
        self.destination_flag[arrived_final_destination] = -1
        for i in range(self.destination.shape[1]):
            self.destination[self.num_steps - 1, i, :] = self.waypoints[self.destination_flag[i], i, :]

        mask_ = frame['mask_p'].clone()
        mask_[arrived_final_destination] = 0
        self.mask_p = torch.cat((self.mask_p, mask_.unsqueeze(0)), dim=0)
        self.mask_v = torch.cat((self.mask_v, mask_.unsqueeze(0)), dim=0)
        self.mask_a = torch.cat((self.mask_a, mask_.unsqueeze(0)), dim=0)
        self.position[self.num_steps - 1, mask_ == 0, :] = nan
        self.velocity[self.num_steps - 1, mask_ == 0, :] = 0
        self.acceleration[self.num_steps - 1, mask_ == 0, :] = 0
        self.destination[self.num_steps - 1, mask_ == 0, :] = nan

        # If dN > 0(i.e. frame["num_pedestrians"] > self.num_pedestrians), then it implies that 'frame' has keys like 'add_xxxxx', pass it to self.add_pedestrians to add pedestrians.
        add_num_pedestrians = frame["num_pedestrians"] - self.num_pedestrians
        if (add_num_pedestrians > 0):
            self.add_pedestrians(add_num_pedestrians, **frame)

        return add_num_pedestrians

    def add_pedestrians(self, add_num_pedestrians, add_position, add_destination, add_velocity=torch.tensor([]),
                        add_acceleration=torch.tensor([]), **kwargs):
        '''Add pedestrians with given initial state.

        Input:
             - add_position: (dN, 2)
             - add_destination: (D', dN, 2). If D' > D, we first expand self.destination to (D', N, 2), then concatenate it with add_destination to get (D', N+dN, 2).
             - add_velocity: (dN, 2)
             - add_acceleration: (dN, 2)

        Note: Do NOT change the formal parameters' names, as they should be same with the update_functions defined in data.scenarios.
        '''
        nan = torch.tensor(float('nan'), device=self.position.device)
        self.num_pedestrians += add_num_pedestrians

        self.position = torch.cat(
            (self.position, nan + torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        self.position[-1, -add_num_pedestrians:, :] = add_position

        self.velocity = torch.cat(
            (self.velocity, torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        if (add_velocity.numel()):
            self.velocity[-1, -add_num_pedestrians:, :] = add_velocity

        self.acceleration = torch.cat(
            (self.acceleration, torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        if (add_acceleration.numel()):
            self.acceleration[-1, -add_num_pedestrians:, :] = add_acceleration

        self.mask_p = torch.cat((self.mask_p, torch.zeros((self.num_steps, add_num_pedestrians), device=nan.device)),
                                dim=1)
        self.mask_p[-1, -add_num_pedestrians:] = 1

        self.mask_v = torch.cat((self.mask_v, torch.zeros((self.num_steps, add_num_pedestrians), device=nan.device)),
                                dim=1)
        self.mask_v[-1, -add_num_pedestrians:] = 1

        self.mask_a = torch.cat((self.mask_a, torch.zeros((self.num_steps, add_num_pedestrians), device=nan.device)),
                                dim=1)
        self.mask_a[-1, -add_num_pedestrians:] = 1

        self.waypoints = torch.cat(
            (self.waypoints, nan + torch.zeros((self.num_destinations, add_num_pedestrians, 2), device=nan.device)),
            dim=1)
        add_num_destinations = add_destination.shape[0] - self.num_destinations
        if (add_num_destinations > 0):
            self.num_destinations += add_num_destinations
            self.waypoints = torch.cat(
                (self.waypoints, nan + torch.zeros((add_num_destinations, self.num_pedestrians, 2), device=nan.device)),
                dim=0)
        self.waypoints[:add_destination.shape[0], -add_num_pedestrians:, :] = add_destination

        self.destination = torch.cat(
            (self.destination, nan + torch.zeros((self.num_steps, add_num_pedestrians, 2), device=nan.device)), dim=1)
        self.destination[self.num_steps - 1, -add_num_pedestrians:, :] = add_destination[0, :]

        self.destination_flag = torch.cat(
            (self.destination_flag, torch.zeros((add_num_pedestrians), device=nan.device)), dim=0).type(torch.int)

    def to_trajectories(self):
        trajectories = []
        for n in range(self.num_pedestrians):
            trajectory = []
            for f in range(self.num_steps):
                if (self.mask_p[f, n] == 1):
                    trajectory.append((self.position[f, n, 0].item(),
                                       self.position[f, n, 1].item(), f))
            trajectories.append(trajectory)
        return trajectories

    def to_destinations(self):
        destinations = []
        frame_id = torch.arange(self.num_steps)
        for i, relays in enumerate(torch.transpose(self.waypoints, 0, 1)):
            destination = []
            for des in relays:
                if (torch.any(torch.isnan(des))):
                    continue
                tmp = frame_id[torch.norm(des - self.destination[:, i, :], dim=1) < 0.01]
                if (tmp.numel() > 0):
                    t = tmp[0]
                    destination.append((des[0].item(), des[1].item(), t.item()))
                else:
                    break
            if (destination):
                destinations.append(destination)
        return destinations

    def save_data(self, data_path: str):
        self.meta_data["version"] = "v2.2"
        data = np.array((self.meta_data, self.to_trajectories(),
                         self.to_destinations(),
                         self.obstacles.tolist()), dtype=object)
        np.save(data_path, data)
        print(f"Saved to '{data_path}'.")


class Pedestrians(object):
    """
    """

    def __init__(self):
        super(Pedestrians, self).__init__()

    @staticmethod
    def get_heading_direction(velocity):
        """
        Function: infer people's heading direction (without normalization);
        Using linear smoothing
        Args:
            velocity: (*c, t, N, 2)
        Return:
            heading_direction: (*c, t, N, 2)

        """
        heading_direction = velocity.clone()
        if heading_direction.dim() == 3:
            for i in range(heading_direction.shape[-2]):
                tmp_direction = torch.tensor([0, 0], dtype=float, device=velocity.device)
                for t in range(heading_direction.shape[-3] - 1, -1, -1):  # 749~0
                    if torch.norm(heading_direction[t, i, :], p=2, dim=0) == 0:
                        heading_direction[t, i, :] = tmp_direction
                    else:
                        tmp_direction = heading_direction[t, i, :]
                for t in range(heading_direction.shape[-3]):
                    if torch.norm(heading_direction[t, i, :], p=2, dim=0) == 0:
                        heading_direction[t, i, :] = tmp_direction
                    else:
                        tmp_direction = heading_direction[t, i, :]
        elif heading_direction.dim() == 4:
            for j in range(heading_direction.shape[-4]):
                for i in range(heading_direction.shape[-2]):
                    tmp_direction = torch.tensor([0, 0], dtype=float, device=velocity.device)
                    for t in range(heading_direction.shape[-3] - 1, -1, -1):
                        if torch.norm(heading_direction[j, t, i, :], p=2, dim=0) == 0:
                            heading_direction[j, t, i, :] = tmp_direction
                        else:
                            tmp_direction = heading_direction[j, t, i, :]
                    for t in range(heading_direction.shape[-3]):
                        if torch.norm(heading_direction[j, t, i, :], p=2, dim=0) == 0:
                            heading_direction[j, t, i, :] = tmp_direction
                        else:
                            tmp_direction = heading_direction[j, t, i, :]

        tmp_direction = torch.norm(heading_direction, p=2, dim=-1, keepdim=True)
        tmp_direction_ = tmp_direction.clone()
        tmp_direction_[tmp_direction_ == 0] += 0.1
        heading_direction = heading_direction / tmp_direction_
        return heading_direction

    @staticmethod
    def get_relative_quantity(A, B):
        """
        Function:
            The relative amount among all objects in A and all objects in B at
            each moment get relative vector xj - xi fof each pedestrain i (B - A)
        Args:
            A: (*c, t, N, dim)
            B: (*c, t, M, dim)
        Return:
            relative_A: (*c, t, N, M, dim)
        """
        dim = A.dim()

        # A = A.unsqueeze(-2).repeat(*([1] * (dim - 1) + [B.shape[-2]] + [1]))  # *c, t, N, M, dim
        # B = B.unsqueeze(-3).repeat(*([1] * (dim - 2) + [A.shape[-3]] + [1, 1]))
        A = A.unsqueeze(-2)
        B = B.unsqueeze(-3)

        # 动态构造 expand 参数，保证维度匹配
        expand_sizes_A = list(A.shape)
        expand_sizes_A[-2] = B.shape[-2]  # 扩展这一维

        expand_sizes_B = list(B.shape)
        expand_sizes_B[-3] = A.shape[-3]  # 扩展这一维

        # 执行 expand（广播，不复制数据）
        A_expanded = A.expand(*expand_sizes_A)
        B_expanded = B.expand(*expand_sizes_B)

        relative = B_expanded - A_expanded
        # relative_A = B - A

        return relative

    @staticmethod
    def get_relative_quantity_expand(A, B):
        """
        Function:
            The relative amount among all objects in A and all objects in B at
            each moment get relative vector xj - xi fof each pedestrain i (B - A)
        Args:
            A: (*c, t, N, dim)
            B: (*c, t, M, dim)
        Return:
            relative_A: (*c, t, N, M, dim)
        """
        dim = A.dim()
        sizeA = A.shape[:-1] + B.shape[-2:-1] + A.shape[-1:]

        A = A.unsqueeze(-2).expand(sizeA)  # *c, t, N, M, dim
        sizeB = B.shape[:-2] + A.shape[-3:-2] + B.shape[-2:]

        B = B.unsqueeze(-3).expand(sizeB)
        relative_A = B - A

        return relative_A.contiguous()

    def get_nearby_obj_in_sight(self, position, objects, heading_direction, k, angle_threshold):
        """
        Function: get The k closest people's index at time t:
            Calculate the relative position between every two people, and then look
            at the angle between this relative position and the speed direction of the
            current person. If the angle is within the threshold range, then it is
            judged to be in the field of view.
        Args:
            k: get the nearest k persons
            position: (*c, t, N, 2)
            objects: (*c, t, M, 2)
            heading_direction: (*c, t, N, 2)
        Return:
            neighbor_index: (*c, t, N, k), The k closest objects' index at time t
        """
        num_ped = position.shape[-2]
        # k = min(k, num_ped)
        relative_pos = self.get_relative_quantity(position, objects[..., :2])  # *c,t,N,M,2
        relative_pos[relative_pos.isnan()] = float('inf')
        distance = torch.norm(relative_pos, p=2, dim=-1)  # *c,t,N,M

        dim = heading_direction.dim()
        heading_direction = heading_direction.unsqueeze(-2).repeat(
            *[[1] * (dim - 1) + [relative_pos.shape[-2]] + [1]])
        view_field = torch.cosine_similarity(
            relative_pos, heading_direction, dim=-1)
        view_field[view_field.isnan()] = -1
        distance[view_field < math.cos(
            3.14 * angle_threshold / 180)] = float('inf')

        sorted_dist, indices = torch.sort(distance, dim=-1)

        return sorted_dist[..., :k], indices[..., :k]

    def get_bright(self, position, bright, args):
        dim = position.dim()
        N = position.shape[-2]
        M = torch.tensor(args.M,device=args.device)
        num_steps = position.shape[-3]
        bright = bright.float()
        bright_pos = bright[:, :2]
        bright_pos[..., 0] = bright_pos[..., 0] -360
        bright_pos[..., 1] = (bright_pos[..., 1]-288 )*-1
        ones = torch.ones_like(bright_pos[...,0]).unsqueeze(-1)
        bright_pos = torch.cat([bright_pos, ones], dim=-1)
        bright_pos_T = bright_pos.T
        world = torch.matmul(M, bright_pos_T)  # [3, N]
        world = world[:2, :] / (world[2:, :] + 1e-6)  # 防除0
        world = world.T  # [N, 2]
        bright_pos = world

        bright_other = bright[:, 2:]

        bright_other2 = bright_other.unsqueeze(-3).repeat(
            *([1] * (dim - 3) + [num_steps] + [N] + [1, 1]))  # *c, t, N, M, 2

        bright_pos2 = self.get_relative_quantity(position, bright_pos)
        # bright_pos2 = torch.nan_to_num(bright, na)
        if dim==3:
            return torch.cat((bright_pos2, bright_other2), dim=-1)
        else:
            bright_other2 = bright_other2.repeat(bright_pos2.shape[0], 1, 1, 1, 1)  # 补成 (32, ...)
            # print(bright_pos2.shape)
            # print(bright_other2.shape)
            return torch.cat((bright_pos2, bright_other2), dim=-1)

    def get_ave_bright_area(self, position, bright, head_direction,top_k, sight_angle, bin_size):
        """
        positon: *c, t, N, 2
        brights: *c, t, M, 3
        head_direction: *c, t,N, 2

        """
        dim = position.dim()
        N = position.shape[-2]

        num_steps = position.shape[-3]
        bright = bright.unsqueeze(-3).repeat(
            *([1] * (dim - 3) + [num_steps] + [1, 1]))  # *c, t, num, 3


        num_bin = sight_angle *2 // bin_size
        eps = 1e-6
        relative_pos = self.get_relative_quantity(position, bright[..., :2])#*c t,N,M,2
        dx, dy = relative_pos[..., 0], relative_pos[..., 1]
        angle_to_obj = torch.atan2(dy, dx)#*c t,N,M
        angle_to_obj = angle_to_obj * 180/ math.pi

        heading_angle = torch.atan2(head_direction[..., 1], head_direction[..., 0])
        heading_angle = heading_angle.unsqueeze(-1)

        # 相对角度，范围 [-180, 180]
        relative_angle = (angle_to_obj - heading_angle + 180) % 360 - 180

        #仅保留视野内的目标
        mask_in_fov = (relative_angle >= - sight_angle) & (relative_angle < sight_angle)

        #将角度平移，使得【-90,90】映射到【0,180】，在除以bin_size得到bin_index
        relative_angle_clipped = torch.where(mask_in_fov, relative_angle + sight_angle, torch.full_like(relative_angle, -999))
        bin_index = (relative_angle_clipped // bin_size).long()

        #扩展bright到和relative_angle相同维度
        """
        dim = heading_direction.dim()
        heading_direction = heading_direction.unsqueeze(-2).repeat(
            *[[1] * (dim - 1) + [relative_pos.shape[-2]] + [1]])"""
        bb = bright[..., -1]
        bb = bb.unsqueeze(-2).repeat( *[[1] * (dim - 2) + [position.shape[-2]] + [1]] )
        # not_nan_mask = ~torch.isnan(bb)
        avg = torch.zeros(*relative_angle.shape[:-1], num_bin, device=position.device)
        counts = torch.zeros_like(avg)

        for i in range(num_bin):
            mask = mask_in_fov & (bin_index == i)
            avg[..., i] = torch.where(mask, bb, torch.zeros_like(bb)).sum(dim=-1)
            counts[..., i] = mask.sum(dim=-1) + eps

        return avg / counts



    def get_filtered_features(self, features, nearby_idx, nearby_dist, dist_threshold):
        """
        features: (*c, t, N, M, dim)
        nearby_idx: (*c, t, N, k)
        nearby_dist: (*c, t, N, k)
        """
        dim = nearby_idx.dim()
        nearby_idx = nearby_idx.unsqueeze(-1).repeat(*([1] * dim + [features.shape[-1]]))  # t,n,k,dim
        features = torch.gather(features, -2, nearby_idx)  # t,n,k,dim

        dist_filter = torch.ones(features.shape, device=features.device)
        nearby_dist = nearby_dist.unsqueeze(-1).repeat(*([1] * dim + [features.shape[-1]]))
        dist_filter[nearby_dist > dist_threshold] = 0
        features[dist_filter == 0] = 0  # nearest neighbor less than k --> zero padding

        return features, dist_filter[..., 0]
    def get_filtered_features2(self, features, nearby_idx, nearby_dist, dist_threshold):
        """
        features: (*c, t, N, M, dim)
        nearby_idx: (*c, t, N, k)
        nearby_dist: (*c, t, N, k)
        """
        dim = nearby_idx.dim()
        nearby_idx = nearby_idx.unsqueeze(-1).repeat(*([1] * dim + [features.shape[-1]]))  # t,n,k,dim
        features = torch.gather(features, -2, nearby_idx)  # t,n,k,dim

        dist_filter = torch.ones(features.shape, device=features.device)
        nearby_dist = nearby_dist.unsqueeze(-1).repeat(*([1] * dim + [features.shape[-1]]))
        dist_filter[nearby_dist > dist_threshold] = 0
        features[dist_filter == 0] = 0  # nearest neighbor less than k --> zero padding

        return features, dist_filter[..., 0]
    def get_crowd_feature_within_radius(self, position, velocity, acceleration, perception_radius=1.0):
        """
        Args:
            position: (..., T, N, 2)
            velocity: same as position
            acceleration: same as position
        Returns:
            crowd_feature: (..., T, N, 5)
            人群的平均速度，平均加速度，平均距离
        """
        orig_shape = position.shape  # e.g. (B, T, N, 2) or (T, N, 2)
        *prefix_dims, T, N, _ = position.shape  # 抽出前缀

        # reshape to (-1, T, N, 2)
        flat_batch = int(torch.prod(torch.tensor(prefix_dims))) if prefix_dims else 1
        position = position.reshape(flat_batch, T, N, 2)
        velocity = velocity.reshape(flat_batch, T, N, 2)
        acceleration = acceleration.reshape(flat_batch, T, N, 2)

        # 1. relative dist (B_flat, T, N, N, 2)
        dist = self.get_relative_quantity(position, position)
        distance = torch.norm(dist, dim=-1)  # (B_flat, T, N, N)

        # 2. mask self
        eye_mask = torch.eye(N, device=position.device).bool().unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        pairwise_dist = distance.masked_fill(eye_mask, float('inf'))  # (B_flat, T, N, N)

        # 3. radius mask
        in_radius_mask = pairwise_dist < perception_radius  # (B_flat, T, N, N)
        num_neighbors = in_radius_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B_flat, T, N, 1)

        # 4. mean vel
        vel_j = velocity.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B_flat, T, N, N, 2)
        masked_vel = vel_j * in_radius_mask.unsqueeze(-1)  # (B_flat, T, N, N, 2)
        mean_vel = masked_vel.sum(dim=3) / num_neighbors  # (B_flat, T, N, 2)

        # 5. mean acc
        acc_j = acceleration.unsqueeze(2).expand(-1, -1, N, -1, -1)
        masked_acc = acc_j * in_radius_mask.unsqueeze(-1)
        mean_acc = masked_acc.sum(dim=3) / num_neighbors  # (B_flat, T, N, 2)

        # 6. mean dist
        masked_dist = pairwise_dist * in_radius_mask  # (B_flat, T, N, N)
        masked_dist = torch.nan_to_num(masked_dist, nan=0.0)
        mean_dist = masked_dist.sum(dim=3) / num_neighbors.squeeze(-1)  # (B_flat, T, N)

        # 7. concat
        crowd_feature = torch.cat([mean_vel, mean_acc, mean_dist.unsqueeze(-1)], dim=-1)  # (B_flat, T, N, 5)

        # reshape back
        if prefix_dims:
            out_shape = (*prefix_dims, T, N, 5)
        else:
            out_shape = (T, N, 5)
        return crowd_feature.reshape(out_shape)

    def get_interests_features(self, position, velocity, interests, ped, radius):
        num_steps = position.shape[-3]
        #shops N,2
        interests_features = torch.tensor([[] for _ in range(num_steps)], device=ped.device)
        if len(interests) >0:
            dim = interests.dim()
            interests = interests.unsqueeze(-3).repeat(
                *([1] * (dim - 2) + [num_steps] + [1, 1]))  # (T, N, 2) → (T, t, N, 2)

        return interests_features



    def get_bright_features(self, position, velocity, acceleration,heading_direction,ped, brights, args):
        dim = brights.shape[-1] #3
        N = position.shape[-2]

        num_steps = position.shape[-3]
        brights = brights.unsqueeze(-3).repeat(
            *([1] * (dim - 3) + [num_steps] + [1, 1]))  # *c, t, num, 3
        avg_bright = self.get_ave_bright_area(
            position, brights, heading_direction, args.topk_brights, args.sight_angle_brights, args.bin)  # top k=100
        brights_pos = brights[...,:2]
        # bb = brights[...,2].unsqueeze(-1).unsqueeze(-3).repeat(*([1]+ [N] + [1, 1]))
        brig = torch.cat((brights_pos, torch.zeros(brights_pos.shape, device=brights.device),
                          torch.zeros(brights_pos.shape, device=brights.device)), dim=-1)
        brights_pos_features = self.get_relative_quantity(ped, brig)  #这是位置信息 t N M dim
        brights_features = torch.cat((brights_pos_features, avg_bright), dim=-1)
        # brights_features, neigh_bright_mask = self.get_filtered_features2(
        #     brights_features, near_brights_idx, near_brights_dist, args.dist_threshold_brights)




        return brights_features


    def get_relative_features(
            self, data, position, velocity, acceleration, destination, obstacles, shops, brights,
            topk_ped, sight_angle_ped, dist_threshold_ped, topk_obs,
            sight_angle_obs, dist_threshold_obs, args
    ):
        """
            position: *c, t, N, 2
            obstacles: *c, N, 2
            destination: *c, t, N, 2
        Return:
            dest_features: *c, t, N, 2
        Notice:
            c is channel size
        """

        acceleration[acceleration.isnan()] = 0
        velocity[velocity.isnan()] = 0

        num_steps = position.shape[-3]
        heading_direction = self.get_heading_direction(velocity)

        near_ped_dist, near_ped_idx = self.get_nearby_obj_in_sight(
            position, position, heading_direction, topk_ped, sight_angle_ped)  # top_k=6
        ped = torch.cat((position, velocity, acceleration), dim=-1)
        ped_features = self.get_relative_quantity(ped, ped)  # *c t N N dim
        ped_features, neigh_ped_mask = self.get_filtered_features(
            ped_features, near_ped_idx, near_ped_dist, dist_threshold_ped)

        dest_features = destination - position
        dest_features[dest_features.isnan()] = 0.


        obs_features = torch.tensor([[] for _ in range(num_steps)], device=obstacles.device)
        if len(obstacles) > 0:
            dim = obstacles.dim()
            obstacles = obstacles.unsqueeze(-3).repeat(
                *([1] * (dim - 2) + [num_steps] + [1, 1]))  # *c, t, N, 2
            near_obstacle_dist, near_obstacle_idx = self.get_nearby_obj_in_sight(
                position, obstacles, heading_direction, topk_obs, sight_angle_obs)  # top k=100
            obs = torch.cat((obstacles, torch.zeros(obstacles.shape, device=obstacles.device),
                             torch.zeros(obstacles.shape, device=obstacles.device)), dim=-1)
            obs_features = self.get_relative_quantity(ped, obs)  # t N M dim
            obs_features, neigh_obs_mask = self.get_filtered_features(
                obs_features, near_obstacle_idx, near_obstacle_dist, dist_threshold_obs)
        # obs_features= torch.tensor([[0, 0], [1,  1]], dtype=torch.float, device=args.device)
        # near_obstacle_idx= torch.tensor([[0, 0], [1,  1]], dtype=torch.float, device=args.device)
        # neigh_obs_mask =torch.tensor([[0, 0], [1,  1]], dtype=torch.float, device=args.device)


        crowd_features = self.get_crowd_feature_within_radius(position, velocity,acceleration, args.perception_radius)
        shops_features = self.get_interests_features(position,velocity,shops, ped, args.interests_radius)
        # brights_features= self.get_ave_bright_area(position, brights, heading_direction, args.topk_brights, args.sight_angle_brights, args.bin)
        # brights_features = self.get_bright(position, brights, args)
        brights_features = brights.unsqueeze(0).unsqueeze(0).expand(position.shape[-3],position.shape[-2],-1,-1) #8320

        # env_emb = self.get_env_features(args,data)
        return ped_features, obs_features, dest_features, crowd_features, shops_features, brights_features,\
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask

    @staticmethod
    def calculate_collision_label(ped_features):
        """

        Args:
            ped_features: ...,k,6: (p,v,a)

        Returns:
            collisions: ...,k
        """

        with torch.no_grad():
            time = torch.arange(10, device=ped_features.device) * 0.1
            time = time.resize(*([1] * (ped_features.dim() - 1)), 10, 1)
            collisions = ped_features[..., :2].unsqueeze(-2) + ped_features[..., 2:4].unsqueeze(-2) * time
            collisions = torch.norm(collisions, p=2, dim=-1)  # c,t,n,k,10
            collisions[collisions >= 0.5] = 0
            collisions[(collisions < 0.5) & (collisions != 0)] = 1
            collisions = torch.sum(collisions, dim=-1)  # c,t,n,k
            collisions[collisions > 0] = 1

        return collisions

    @staticmethod
    def collision_detection(position, threshold, real_position=None):
        """
        Args:
            position: t,n,2 / c,t,n,2


        Returns:

        """
        # position = position.clone()
        relative_pos = Pedestrians.get_relative_quantity(position, position)  # c,n,n,2
        rel_distance = torch.norm(relative_pos, p=2, dim=-1)  # c,n,n
        collisions = rel_distance.clone()
        collisions[rel_distance < threshold] = 1
        collisions[rel_distance >= threshold] = 0

        # delete self loop
        identical_matrix = torch.eye(collisions.shape[-1], device=collisions.device)
        if collisions.dim() == 3:
            identical_matrix = identical_matrix.unsqueeze(0).repeat(collisions.shape[0], 1, 1)
        elif collisions.dim() == 4:
            identical_matrix = identical_matrix.reshape(1, 1, identical_matrix.shape[-1], -1)
            identical_matrix = identical_matrix.repeat(collisions.shape[0], collisions.shape[1], 1, 1)
        collisions = collisions - identical_matrix

        collisions[collisions.isnan()] = 0  # t,n,n

        # valid_steps = position[..., 0].clone()  # c,t,n
        # valid_steps[~valid_steps.isnan()] = 1
        # valid_steps[valid_steps.isnan()] = 0
        # valid_steps = torch.sum(valid_steps, dim=-2, keepdim=True)  # c,1,n
        if real_position is not None:

            assert real_position.dim() == 3, 'Value Error: real_position only supports 3 dimensional inputs (t,n,2)'
            relative_pos = Pedestrians.get_relative_quantity(real_position, real_position)  # c,n,n,2
            rel_distance = torch.norm(relative_pos, p=2, dim=-1)  # c,n,n
            real_collisions = rel_distance.clone()
            real_collisions[rel_distance < threshold] = 1
            real_collisions[rel_distance >= threshold] = 0
            real_collisions[real_collisions.isnan()] = 0  # t,n,n
            friends = torch.sum(real_collisions, dim=0)  # n,n
            friends[friends <= 25] = 1
            friends[friends > 25] = 0
            friends = friends.unsqueeze(0)
        else:
            if collisions.dim() == 3:
                friends = torch.sum(collisions, dim=0)  # n,n
                friends[friends <= 25] = 1
                friends[friends > 25] = 0
                friends = friends.unsqueeze(0)
            elif collisions.dim() == 4:
                friends = collisions[:, :4]
                friends = torch.sum(friends, dim=1)
                friends[friends > 0] = 1
                friends = 1 - friends
                friends = friends.unsqueeze(1)
        collisions *= friends

        return collisions


class TimeIndexedPedData(Dataset, Pedestrians):
    """
    Attributes:
        ped_features: t * N * k1 * dim(6): relative position, velocity, acceleration
        obs_features: t * N * k2 * dim(6)
        self_features: t * N * dim(6): dest_features, cur_acc, desired_speed
        labels: t * N * 2
    """

    def __init__(
            self,
            ped_features=torch.tensor([]),
            obs_features=torch.tensor([]),
            self_features=torch.tensor([]),
            desired_speed=torch.tensor([]),
            labels=torch.tensor([]),
            mask_p=torch.tensor([]),
            mask_v=torch.tensor([]),
            shops = torch.tensor([]),
            # env_emb = torch.tensor([]),
            brights = torch.tensor([]),
            mask_a=torch.tensor([])):
        super(TimeIndexedPedData, self).__init__()
        self.ped_features = ped_features
        self.obs_features = obs_features
        self.self_features = self_features
        self.labels = labels
        self.desired_speed = desired_speed
        self.shops = shops
        self.brights = brights
        self.mask_p = mask_p
        self.mask_v = mask_v
        # self.env_emb = env_emb
        self.mask_a = mask_a
        self.num_frames = ped_features.shape[0]
        self.dataset_len = self.num_frames
        if self.dataset_len:
            self.num_pedestrians = ped_features.shape[1]
            self.ped_feature_dim = ped_features.shape[-1]
            self.obs_feature_dim = obs_features.shape[-1]
            self.self_feature_dim = self_features.shape[-1]
        else:
            self.num_pedestrians = 0
            self.ped_feature_dim = 0
            self.obs_feature_dim = 0
            self.self_feature_dim = 0

        self.mask_p_pred = None
        self.mask_v_pred = None
        self.mask_a_pred = None
        self.meta_data = None

    def __len__(self):

        return self.num_frames

    def __getitem__(self, index):

        if self.num_frames > 0:
            item = [self.ped_features[index], self.obs_features[index],
                    self.self_features[index], self.labels[index], self.self_hist_features[index],
                    self.mask_p_pred[index], self.mask_v_pred[index], self.mask_a_pred[index]]
        else:
            raise ValueError("Haven't load any data yet!")

        return item

    def to_pointwise_data(self):
        tmp_data = PointwisePedData()
        tmp_data.load_from_time_indexed_peddata(self)
        return tmp_data

    def to_channeled_time_index_data(self, stride=25, mode='slice'):
        tmp_data = ChanneledTimeIndexedPedData()
        tmp_data.load_from_time_indexed_peddata(self, stride, mode)
        return tmp_data

    def move_index_matrix(self, idx_matrix, direction='forward', n_steps=1, dim=0):
        """
       ** [[0,1,1,1],[1,1,0,0]] -> [[0,0,1,1], [0,1,0,0]]

        Args:
            idx_matrix: 0-1 index matrix
            direction: 'forward' or 'backward'
            n_steps: number of steps
            dim: moving dimension

        Returns:
            mask: results
        """
        mask = idx_matrix.clone()
        moving_shape = list(mask.shape)
        moving_shape[dim] = n_steps
        if direction == 'backward':
            mask = mask.index_select(dim, torch.arange(mask.shape[dim] - n_steps, device=idx_matrix.device))
            mask = torch.cat((torch.zeros(moving_shape, device=idx_matrix.device), mask), dim=dim)
        elif direction == 'forward':
            mask = mask.index_select(dim, torch.arange(n_steps, mask.shape[dim], device=idx_matrix.device))
            mask = torch.cat((mask, torch.zeros(moving_shape, device=idx_matrix.device)), dim=dim)
        mask *= idx_matrix
        return mask

    @staticmethod
    def turn_detection(data: RawData):
        """
        Args:
            data:

        Returns:
            non_abnormal: t,n

        """
        position = data.position.clone()
        velocity = data.velocity.clone()
        T, N, _ = position.shape
        position[position.isnan()] = 1e4

        starts = torch.zeros((N, 2), device=position.device) + 1e4
        v_starts = torch.zeros((N, 2), device=position.device) + 1e4
        ends = torch.zeros((N, 2), device=position.device) + 1e4
        for i in range(T):
            v_starts[starts >= 1e4] = velocity[i, starts >= 1e4]
            starts[starts >= 1e4] = position[i, starts >= 1e4]
            ends[ends >= 1e4] = position[T - i - 1, ends >= 1e4]
        dist = torch.norm(ends - starts, p=2, dim=-1) + 1e-6
        norm_v = torch.norm(v_starts, p=2, dim=-1) + 1e-6

        cos_theta = torch.sum((ends - starts) * v_starts, dim=-1) / dist / norm_v
        cos_theta[cos_theta < np.cos(3.1415 * 20 / 180)] = 0
        cos_theta[cos_theta > 0] = 1

        non_abnormal = cos_theta

        mean_velocity = torch.norm(velocity, p=2, dim=-1)  # t,n
        mean_velocity = torch.sum(mean_velocity, dim=0) / torch.sum(data.mask_v, dim=0)

        non_abnormal[mean_velocity < 1.3 * 0.3] = 0

        return non_abnormal

    def make_dataset(self, args, raw_data: RawData):
        """
        Transform a RawData object into a TimeIndexedPedData object
        Inputs:
            raw_data: RawData object
        Outputs:
            ped_features: t * N * k1 * dim(6): relative position, velocity, acceleration
            obs_features: t * N * k2 * dim(6)
            self_features: t * N * dim(2 + 2 + 1): dest_features, cur_acc, desired_speed
            self_hist_features:  t*N*k*dim(6)
                historical velocity: vx0, vy0, vx1, vy1, ..., vxn, vyn
            labels: t * N * 6 Position, speed, acceleration at the t+1 time step
            mask_a: t * N

        """

        # raw_data.to(args.device)

        ped_features, obs_features, dest_features, crowd_features, shops_features, brights_features,\
            near_ped_idx, neigh_ped_mask, near_obstacle_idx, neigh_obs_mask = self.get_relative_features(raw_data,
            raw_data.position, raw_data.velocity,
            raw_data.acceleration, raw_data.destination,
            raw_data.obstacles, raw_data.shops, raw_data.brights, args.topk_ped, args.sight_angle_ped,
            args.dist_threshold_ped, args.topk_obs,
            args.sight_angle_obs, args.dist_threshold_obs,args)

        raw_data.to(args.device)
        self.args = args
        ped_features = ped_features.to(args.device)
        crowd_features = crowd_features.to(args.device)
        shops_features = shops_features.to(args.device)
        # ped = ped.to(args.device)
        brights_features = brights_features.to(args.device)
        obs_features = obs_features.to(args.device)
        dest_features = dest_features.to(args.device)
        near_ped_idx = near_ped_idx.to(args.device)
        # neigh_bright_idx = neigh_bright_idx.to(args.device)
        neigh_ped_mask = neigh_ped_mask.to(args.device)
        near_obstacle_idx = near_obstacle_idx.to(args.device)
        neigh_obs_mask = neigh_obs_mask.to(args.device)
        # neigh_bright_mask = neigh_bright_mask.to(args.device)

        self.near_ped_idx = near_ped_idx
        # self.neigh_bright_idx = neigh_bright_idx
        self.neigh_ped_mask = neigh_ped_mask
        self.near_obstacle_idx = near_obstacle_idx
        self.neigh_obs_mask = neigh_obs_mask
        # self.neigh_bright_mask = neigh_bright_mask
        self.crowd_features = crowd_features
        self.shops_features = shops_features
        # self.env_emb = env_emb
        self.brights_features = brights_features

        self.abnormal_mask = self.turn_detection(raw_data)

        self.ped_features = ped_features
        if len(obs_features) > 0:
            self.obs_features = obs_features
        else:
            self.obs_features = torch.tensor([[] for _ in range(ped_features.shape[0])], device=ped_features.device)

        # get hist_features
        num_frames = ped_features.shape[0]
        num_peds = ped_features.shape[1]
        hist_features = torch.zeros(list(raw_data.velocity.shape[:-1]) + [6], device=ped_features.device)  # t, N, 6
        hist_features = hist_features.unsqueeze(2).repeat(1, 1, args.num_history_feature, 1)  # t, N, k, 6
        for i in range(args.num_history_feature):
            tmp_frame = args.num_history_feature - i - 1
            hist_features[tmp_frame:, :, i, 0:2] = raw_data.position[:num_frames - tmp_frame, :, :]
            hist_features[tmp_frame:, :, i, 2:4] = raw_data.velocity[:num_frames - tmp_frame, :, :]
            hist_features[tmp_frame:, :, i, 4:6] = raw_data.acceleration[:num_frames - tmp_frame, :, :]
        # mask = hist_features!=hist_features
        # hist_features = hist_features.masked_fill(mask,0) # aborted: do zero_padding during training/sampling

        # hist_velocity = hist_velocity.reshape(num_frames, num_peds, -1)  # t, N, k*2
        crowd_hist_features = torch.zeros_like(crowd_features, device=crowd_features.device)  # t, N, 6
        crowd_hist_features = crowd_hist_features.unsqueeze(2).repeat(1, 1, args.num_crowd_feature, 1)
        for i in range(args.num_crowd_feature):
            tmp_frame = args.num_crowd_feature -i -1
            crowd_hist_features[tmp_frame:, :, i, :] = crowd_features[:num_frames - tmp_frame, :,:]
        self.crowd_hist_features = crowd_hist_features
        # get hist_velocity features
        num_frames = ped_features.shape[0]
        num_peds = ped_features.shape[1]
        hist_velocity = torch.zeros(raw_data.velocity.shape, device=ped_features.device)  # t, N, 2
        hist_velocity = hist_velocity.unsqueeze(2).repeat(1, 1, args.num_history_velocity, 1)  # t, N, k, 2
        for i in range(args.num_history_velocity):
            tmp_frame = args.num_history_velocity - i - 1
            hist_velocity[tmp_frame:, :, i, :] = raw_data.velocity[:num_frames - tmp_frame, :, :]
        hist_velocity = hist_velocity.reshape(num_frames, num_peds, -1)  # t, N, k*2

        # calculate desired_speed
        skip_frames = args.skip_frames
        desired_speed = torch.zeros(num_peds, device=ped_features.device)  # N
        for i in range(num_peds):
            start_idx = 0
            for j in range(num_frames):
                if torch.norm(raw_data.velocity[j, i, :]) > 0:
                    start_idx = j
                    break
            desired_speed[i] = torch.mean(
                torch.norm(raw_data.velocity[start_idx:start_idx + skip_frames, i, :], p=2, dim=-1))
        desired_speed = desired_speed.unsqueeze(0).unsqueeze(-1)
        desired_speed = desired_speed.repeat(num_frames, 1, 1)
        self.self_hist_features = hist_features
        self.self_features = torch.cat((dest_features, hist_velocity, raw_data.acceleration, desired_speed),
                                       dim=-1)  # MN**f_des

        self.labels = torch.cat((raw_data.position, raw_data.velocity, raw_data.acceleration), dim=-1)

        collision_labels = self.calculate_collision_label(self.ped_features)
        self.labels = torch.cat((self.labels, collision_labels), dim=-1)

        # update time steps that are useless for validation
        self.mask_a_pred = self.move_index_matrix(raw_data.mask_a, 'backward', skip_frames - 1, dim=0)
        self.mask_v_pred = self.move_index_matrix(raw_data.mask_v, 'backward', skip_frames - 1, dim=0)
        self.mask_p_pred = self.move_index_matrix(raw_data.mask_p, 'backward', skip_frames - 1, dim=0)

        # the last time step cannot be used for prediction
        self.mask_a_pred = self.move_index_matrix(self.mask_a_pred, 'forward', 1, dim=0)
        # self.mask_v_pred = self.move_index_matrix(self.mask_v_pred, 'forward', 1, dim=0)
        # self.mask_p_pred = self.move_index_matrix(self.mask_p_pred, 'forward', 1, dim=0)

        self.meta_data = raw_data.meta_data
        self.topk_obs = args.topk_obs
        self.num_frames = self.dataset_len = num_frames
        self.num_pedestrians = self.ped_features.shape[1]
        self.ped_feature_dim = self.ped_features.shape[-1]
        self.obs_feature_dim = self.obs_features.shape[-1]
        self.self_feature_dim = self.self_features.shape[-1]

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')

    def set_dataset_info(self, dataset, raw_data, slice_idx):
        # pdb.set_trace()
        self.meta_data = raw_data.meta_data
        self.time_unit = raw_data.time_unit
        self.num_frames = self.dataset_len = dataset.num_frames
        self.position = raw_data.position[slice_idx, :, :]
        # self.brights = raw_data.brights[slice_idx, :, :]
        self.velocity = raw_data.velocity[slice_idx, :, :]
        self.acceleration = raw_data.acceleration[slice_idx, :, :]
        self.position_desc = raw_data.position_desc
        self.velocity_desc = raw_data.velocity_desc
        self.acceleration_desc = raw_data.acceleration_desc
        self.obstacles = raw_data.obstacles
        self.brights = raw_data.brights
        # self.image = raw_data.image
        self.id = raw_data.id
        self.description = raw_data.description
        self.info = raw_data.info
        self.box = raw_data.box
        self.in_box = raw_data.in_box
        # self.env_emb = raw_data.env_emb
        self.text_obs_emb = raw_data.text_obs_emb
        self.text_in_emb = raw_data.text_in_emb
        self.image_obs_emb = raw_data.image_obs_emb
        self.image_in_emb = raw_data.image_in_emb
        self.image_big = raw_data.image_big
        self.text_big = raw_data.text_big
        self.destination = raw_data.destination[slice_idx, :, :]
        self.dest_idx = raw_data.dest_idx[slice_idx, :]
        self.waypoints = raw_data.waypoints
        self.dest_num = raw_data.dest_num

        self.mask_p = raw_data.mask_p[slice_idx, :]
        self.mask_a = raw_data.mask_a[slice_idx, :]
        self.mask_v = raw_data.mask_v[slice_idx, :]

        self.mask_p_pred = dataset.mask_p_pred[slice_idx, :]
        self.mask_v_pred = dataset.mask_v_pred[slice_idx, :]
        self.mask_a_pred = dataset.mask_a_pred[slice_idx, :]
        self.self_feature_dim = dataset.self_feature_dim
        self.ped_feature_dim = dataset.ped_feature_dim
        self.obs_feature_dim = dataset.obs_feature_dim
        self.abnormal_mask = dataset.abnormal_mask


class TimeIndexedPedDataPolarCoor(TimeIndexedPedData):
    """docstring for TimeIndexedPedDataPolarCoor"""

    def __init__(self):
        super(TimeIndexedPedDataPolarCoor, self).__init__()

    @staticmethod
    def cart_to_polar(points, base):
        """
        Args:
            points: c, t, n, 2
            base: c, t, n, 2 )

        Returns:
            polar_coor: c, t, n, 2
        """
        volume = torch.norm(points, p=2, dim=-1, keepdim=True)
        volume_ = volume.clone()
        volume_[volume_ == 0] += 0.1  # to avoid zero devision

        p = points / volume_
        cos_p = p[..., 0]
        sin_p = p[..., 1]
        cos_b = base[..., 0]
        sin_b = base[..., 1]
        sign_sin_pb = torch.sign(sin_p * cos_b - cos_p * sin_b)
        sign_sin_pb = sign_sin_pb.unsqueeze(-1)

        theta = torch.sum(points * base, dim=-1, keepdim=True) / volume_
        theta = torch.clamp(theta, -1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(theta) * sign_sin_pb

        return torch.cat((volume, theta), dim=-1)

    @staticmethod
    def polar_to_cart(points, base):
        """
        Args:
            points: c, t, n, 2
            base: c, t, n, 2 )

        Returns:
            cart_coor: c, t, n, 2
        """
        cart_base = torch.zeros(base.shape, device=points.device)
        cart_base[..., 0] = 1.  # base is (1, 0)
        polar_base = TimeIndexedPedDataPolarCoor.cart_to_polar(base, cart_base)
        polar_base[..., 0] = 0
        points = points + polar_base
        x = points[..., 0] * torch.cos(points[..., 1])
        y = points[..., 0] * torch.sin(points[..., 1])
        return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1)), dim=-1)

    def to_polar_system(self):
        """
       ped_features,obs_features, labels, dest features
        Args:
            points: ..., N, 2
            base: ..., N, 2
            ped_features: t * N * k1 * dim(6): relative position, velocity, acceleration in polar coordinates
            obs_features: t * N * k2 * dim(6):
            self_features: t * N * dim(2 + 2*k + 2 + 1): dest_features, hist_velocity, cur_acc, desired_speed
            labels: t * N * 6 Position, speed, acceleration, acc_polar at the t+1 time step
        """

        velocity = self.self_features[..., -5:-3]  # c, t, n, 2
        n_direction = self.get_heading_direction(velocity)  # c, t, N, 2

        # labels
        # acc_polar = self.cart_to_polar(self.labels[..., -2:], n_direction)
        # self.labels = torch.cat((self.labels, acc_polar), dim=-1)

        n_directions_ = n_direction.unsqueeze(-2)
        n_directions_ = n_directions_.repeat(*[1] * (n_direction.dim() - 1), self.ped_features.shape[-2], 1)
        self.ped_features = torch.cat(
            (self.cart_to_polar(self.ped_features[..., 0:2], n_directions_),
             self.cart_to_polar(self.ped_features[..., 2:4], n_directions_),
             self.cart_to_polar(self.ped_features[..., 4:6], n_directions_)), dim=-1
        )
        n_directions_ = n_direction.unsqueeze(-2)
        n_directions_ = n_directions_.repeat(*[1] * (n_direction.dim() - 1), self.obs_features.shape[-2], 1)
        if self.obs_feature_dim > 0:
            self.obs_features = torch.cat(
                (self.cart_to_polar(self.obs_features[..., 0:2], n_directions_),
                 self.cart_to_polar(self.obs_features[..., 2:4], n_directions_),
                 self.cart_to_polar(self.obs_features[..., 4:6], n_directions_)), dim=-1
            )


class PointwisePedData(Dataset):
    """
    Attributes:
        ped_features: (N * t) * k1 * dim(6)
        obs_features: (N * t) * k2 * dim(6)
        self_features: (N * t) * dim(6)
        labels: (N * t) * 6
    """

    def __init__(
            self,
            ped_features=torch.tensor([]),
            obs_features=torch.tensor([]),
            self_features=torch.tensor([]),
            labels=torch.tensor([]),
            self_hist_features=torch.tensor([]),
            crowd_hist_features = torch.tensor([])):
        super(PointwisePedData, self).__init__()
        self.ped_features = ped_features
        self.obs_features = obs_features
        self.crowd_hist_features = crowd_hist_features
        self.self_features = self_features
        self.self_hist_features = self_hist_features
        self.labels = labels
        self.dataset_len = labels.shape[0]
        self.self_feature_dim = self.self_features.shape[-1]
        self.ped_feature_dim = self.ped_features.shape[-1]
        self.obs_feature_dim = self.obs_features.shape[-1]

    def __len__(self):

        return self.dataset_len

    def __getitem__(self, idx):
        item = [self.ped_features[idx], self.obs_features[idx],
                self.self_features[idx], self.labels[idx], self.self_hist_features[idx]]
        return item

    def add(self, other):
        assert (self.time_unit == other.time_unit), "PointwisePedData with different time_unit cannot be merged"
        assert (self.ped_features.shape[-1] == other.ped_features.shape[
            -1]), "PointwisePedData with different feature shape cannot be merged"

        self.ped_features = torch.cat((self.ped_features, other.ped_features), dim=0)
        self.obs_features = torch.cat((self.obs_features, other.obs_features), dim=0)
        self.self_features = torch.cat((self.self_features, other.self_features), dim=0)
        self.labels = torch.cat((self.labels, other.labels), dim=0)
        self.self_hist_features = torch.cat((self.self_hist_features, other.self_hist_features), dim=0)
        self.dataset_len = self.dataset_len + other.dataset_len

    def make_dataset(self, args, raw_data):
        pass

    def load_from_time_indexed_peddata(self, data: TimeIndexedPedData, slice_idx=None):
        if slice_idx:
            ped_features, obs_features, self_features, labels, self_hist_features = data[slice_idx]
            mask_a_pred = data.mask_a_pred[slice_idx]
        else:
            ped_features, obs_features, self_features, labels, self_hist_features = data[:]
            mask_a_pred = data.mask_a_pred

        filter_idx = mask_a_pred.reshape(-1)
        # move one step forward to calibrate with pairwise training
        labels[:-1, :, :] = labels[1:, :, :].clone()
        labels[-1, :, :] = 0
        labels = labels.reshape(filter_idx.shape[0], -1)
        self.labels = labels[filter_idx > 0]
        self.dataset_len = self.labels.shape[0]

        ped_features = ped_features.reshape(*[[-1] + list(ped_features.shape[2:])])
        self.ped_features = ped_features[filter_idx > 0]
        self.ped_feature_dim = self.ped_features.shape[-1]

        self_features = self_features.reshape(*[[-1] + list(self_features.shape[2:])])
        self.self_features = self_features[filter_idx > 0]
        self.self_feature_dim = self.self_features.shape[-1]

        self_hist_features = self_hist_features.reshape(*[[-1] + list(self_hist_features.shape[2:])])
        self.self_hist_features = self_hist_features[filter_idx > 0]
        self.self_hist_feature_dim = self.self_hist_features.shape[-1]

        if obs_features.shape[-1]:
            obs_features = obs_features.reshape(*[[-1] + list(obs_features.shape[2:])])
            self.obs_features = obs_features[filter_idx > 0]
            self.obs_feature_dim = self.obs_features.shape[-1]
        else:
            self.obs_features = torch.zeros((self.ped_features.shape[0], data.topk_obs, self.ped_feature_dim),
                                            device=ped_features.device)

        self.time_unit = data.time_unit

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')


class ChanneledTimeIndexedPedData(Dataset):
    """
    Attributes:
        ped_features: *c, t, N, k1, dim(6)
        obs_features: *c, t, N, k2, dim(6)
        self_features: *c, t, N, dim(6): dest_features, hist_velocity, cur_acc, desired_speed
        labels: *c, t, N, 2
    """

    def __init__(self):
        super(ChanneledTimeIndexedPedData, self).__init__()
        self.num_frames = 0

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):

        if self.num_frames > 0:
            item = [self.ped_features[index, ...], self.obs_features[index, ...],
                    self.self_features[index, ...], self.labels[index, ...]]
        else:
            raise ValueError("Haven't load any data yet!")

        return item

    def transform(self, matrix: torch.tensor, stride, mode='slice'):
        """
        t, ... --> *c, stride, ...
        """
        num_frames = matrix.shape[0]
        if mode == 'slice':
            dim = matrix.dim()
            matrix = matrix.unsqueeze(0).repeat(stride, *([1] * (dim)))
            for i in range(1, stride):
                matrix[i, :num_frames - i, ...] = matrix[i, i:, ...].clone()  # t, *c, ...
            matrix = matrix[:, :num_frames - stride, ...]
            permute_idx = list(range(matrix.dim()))
            permute_idx[0], permute_idx[1] = permute_idx[1], permute_idx[0]
            matrix = matrix.permute(*permute_idx)
        elif mode == 'split':
            step = int(num_frames / stride)
            matrix = matrix[:stride * step, ...]
            matrix = matrix.reshape(step, stride, *matrix.shape[1:])
        else:
            raise NotImplementedError
        return matrix

    def load_from_time_indexed_peddata(self, data: TimeIndexedPedData, stride=25, mode='slice'):
        """
        mode:
            'slice' -
            'split' -

        """
        assert (data.num_frames > stride), "ValueError: stride < #total time steps"
        length = data.num_frames - 10
        sample = int(length - length * 0.10)
        # sample = 0
        self.ped_features = self.transform(data.ped_features, stride, mode)[sample:]
        self.obs_features = self.transform(data.obs_features, stride, mode)[sample:]
        self.shops_features = self.transform(data.shops_features, stride, mode)[sample:]
        self.brights_features = self.transform(data.brights_features, stride, mode)[sample:]
        self.crowd_features = self.transform(data.crowd_features, stride, mode)[sample:]
        self.crowd_hist_features = self.transform(data.crowd_hist_features, stride, mode)[sample:]
        self.self_features = self.transform(data.self_features, stride, mode)[sample:]
        self.self_hist_features = self.transform(data.self_hist_features, stride, mode)[sample:]
        self.near_ped_idx = self.transform(data.near_ped_idx, stride, mode)[sample:]
        self.neigh_ped_mask = self.transform(data.neigh_ped_mask, stride, mode)[sample:]
        self.near_obstacle_idx = self.transform(data.near_obstacle_idx, stride, mode)[sample:]
        self.neigh_obs_mask = self.transform(data.neigh_obs_mask, stride, mode)[sample:]
        self.labels = self.transform(data.labels, stride, mode)[sample:]
        self.mask_p = self.transform(data.mask_p, stride, mode)[sample:]
        self.mask_v = self.transform(data.mask_v, stride, mode)[sample:]
        self.mask_a = self.transform(data.mask_a, stride, mode)[sample:]
        self.mask_a_pred = self.transform(data.mask_a_pred, stride, mode)[sample:]
        self.mask_v_pred = self.transform(data.mask_v_pred, stride, mode)[sample:]
        self.mask_p_pred = self.transform(data.mask_p_pred, stride, mode)[sample:]
        self.position = self.transform(data.position, stride, mode)[sample:]  # *c, t, n, 2
        self.velocity = self.transform(data.velocity, stride, mode)[sample:]
        self.acceleration = self.transform(data.acceleration, stride, mode)[sample:]
        self.destination = self.transform(data.destination, stride, mode)[sample:]
        self.dest_idx = self.transform(data.dest_idx, stride, mode)[sample:]

        self.waypoints = data.waypoints.unsqueeze(0).repeat(self.position.shape[0], *[1] * 3)  # *c, d, n, 2

        self.num_frames = stride
        self.dataset_len = self.ped_features.shape[0]

        self.set_static_info_like(data)

    @staticmethod
    def slice(data, slice_idx):
        out = ChanneledTimeIndexedPedData()
        out.ped_features = data.ped_features[slice_idx, ...]
        out.obs_features = data.obs_features[slice_idx, ...]
        out.shops_features = data.shops_features[slice_idx, ...]
        # out.env_emb = data.env_emb
        out.brights_features = data.brights_features[slice_idx, ...]
        out.crowd_hist_features = data.crowd_hist_features[slice_idx, ...]
        out.crowd_features = data.crowd_features[slice_idx, ...]
        out.self_features = data.self_features[slice_idx, ...]
        out.self_hist_features = data.self_hist_features[slice_idx, ...]
        out.near_ped_idx = data.near_ped_idx[slice_idx, ...]
        out.neigh_ped_mask = data.neigh_ped_mask[slice_idx, ...]
        out.near_obstacle_idx = data.near_obstacle_idx[slice_idx, ...]
        out.neigh_obs_mask = data.neigh_obs_mask[slice_idx, ...]
        out.labels = data.labels[slice_idx, ...]
        out.mask_p = data.mask_p[slice_idx, ...]
        out.mask_v = data.mask_v[slice_idx, ...]
        out.mask_a = data.mask_a[slice_idx, ...]
        out.mask_a_pred = data.mask_a_pred[slice_idx, ...]
        out.mask_v_pred = data.mask_v_pred[slice_idx, ...]
        out.mask_p_pred = data.mask_p_pred[slice_idx, ...]
        out.position = data.position[slice_idx, ...]
        out.velocity = data.velocity[slice_idx, ...]
        out.acceleration = data.acceleration[slice_idx, ...]
        out.destination = data.destination[slice_idx, ...]
        out.dest_idx = data.dest_idx[slice_idx, ...]
        out.waypoints = data.waypoints[slice_idx, ...]
        out.num_frames = data.num_frames
        out.dataset_len = data.dataset_len
        out.set_static_info_like(data)

        return out

    def set_static_info_like(self, data):
        self.obstacles = data.obstacles
        self.brights = data.brights
        self.shops = data.shops
        # self.env_emb = data.env_emb
        self.text_obs_emb = data.text_obs_emb
        self.text_in_emb = data.text_in_emb

        self.image_obs_emb = data.image_obs_emb
        self.image_in_emb = data.image_in_emb

        self.image_big = data.image_big
        self.text_big = data.text_big
        self.id = data.id
        self.description = data.description
        self.info = data.info
        self.box = data.box
        self.in_box = data.in_box
        # self.image = data.image

        self.dest_num = data.dest_num  # n
        self.topk_obs = data.topk_obs
        self.meta_data = data.meta_data
        self.time_unit = data.time_unit
        self.num_pedestrians = data.num_pedestrians
        self.ped_feature_dim = data.ped_feature_dim
        self.obs_feature_dim = data.obs_feature_dim
        self.self_feature_dim = data.self_feature_dim
        self.abnormal_mask = data.abnormal_mask


class SocialForceData(RawData):
    '''
        Dataset for models.socialforce.simulator use

    Attributes:
        - num_steps: int. The number of frames
        - num_pedestrians: int. The number of pedestrians
        - position: (t, N, 2).
        - velocity: (t, N, 2).
        - acceleration: (t, N, 2).
        - destination: (D, N, 2).
        - arrived: (t, N).
        - tau: (N). Reflects the magnitude of the force exerted by the desire to reach the destination, i.e. F = (desired_speed * unit_vector(destination - position) - velocity) / tau.
        - desired_speed: (N).
        - obstacles: (M, 2).
        - avaliable: (t, N). Set to 0 before the pedestrian appears or after the pedestrian leaves.
    '''

    def __init__(self,
                 position=torch.tensor([]),
                 velocity=torch.tensor([]),
                 acceleration=torch.tensor([]),
                 destination=torch.tensor([]),
                 obstacles=torch.tensor([]),
                 mask_p=torch.tensor([]),
                 mask_v=torch.tensor([]),
                 mask_a=torch.tensor([]),
                 desired_speed=torch.tensor([]),
                 meta_data=dict(),
                 default_tau=0.5,
                 use_mask_preq=False,
                 dynamic_desired_speed=False):
        """
        Input:
            ...(same as super())
            "desired_speed": (N),
            "mask_preq": (t, N),
            "duration": (N),
        """
        super().__init__(position=position, velocity=velocity,
                         acceleration=acceleration, waypoints=destination,
                         obstacles=obstacles, mask_p=mask_p, mask_v=mask_v,
                         mask_a=mask_a, meta_data=meta_data)
        if (not self.velocity.numel()):
            self.velocity = torch.zeros_like(self.position)
        if (not self.acceleration.numel()):
            self.acceleration = torch.zeros_like(self.position)
        if (not self.mask_p.numel()):
            self.mask_p = torch.ones((self.num_steps, self.num_pedestrians))
        if (not self.mask_v.numel()):
            self.mask_v = torch.ones((self.num_steps, self.num_pedestrians))
        if (not self.mask_a.numel()):
            self.mask_a = torch.ones((self.num_steps, self.num_pedestrians))

        self.destination = torch.zeros((self.num_steps, self.num_pedestrians, 2),
                                       device=self.position.device) + torch.tensor(float('nan'),
                                                                                   device=self.position.device)
        self.destination[self.num_steps - 1, :, :] = self.waypoints[0, :, :]
        self.destination_flag = torch.zeros(self.num_pedestrians, dtype=int, device=self.position.device)
        self.dynamic_desired_speed = dynamic_desired_speed
        self.desired_speed = desired_speed
        self.default_tau = default_tau
        self.tau = default_tau * torch.ones(self.num_pedestrians, device=self.position.device)
        if use_mask_preq:
            self.mask_preq = torch.ones((self.num_steps, self.num_pedestrians), device=self.position.device)
        else:
            self.mask_preq = None
        self.duration = torch.zeros(self.num_pedestrians, device=self.position.device)

    def get_frame(self, f: int) -> dict:
        """Get f-th frame in dict format."""
        frame = super().get_frame(f)
        frame["desired_speed"] = self.desired_speed
        frame["tau"] = self.tau
        if self.mask_preq is not None:
            frame["mask_preq"] = self.mask_preq[f, :]
        frame["duration"] = self.duration

        return frame

    def get_current_frame(self):
        """Get last frame in dict format."""
        return self.get_frame(self.num_steps - 1)

    def add_frame(self, frame: dict):
        """Add a frame to dataset."""
        super().add_frame(frame)
        if (self.mask_preq != None):
            self.mask_preq = torch.cat((self.mask_preq, frame['mask_preq'].unsqueeze(0)), dim=0)
        if (self.dynamic_desired_speed):
            self.desired_speed = frame['desired_speed']
        self.duration[self.mask_p[-1, :] == 1] += 1

    def add_pedestrians(self, add_num_pedestrians, add_desired_speed, add_tau=torch.tensor([]), **kwargs):
        super().add_pedestrians(add_num_pedestrians, **kwargs)
        self.desired_speed = torch.cat((self.desired_speed, add_desired_speed), dim=0)
        self.tau = torch.cat(
            (self.tau, self.default_tau * torch.ones(add_num_pedestrians, device=self.position.device)), dim=0)
        if (self.mask_preq != None):
            self.mask_preq = torch.cat(
                (self.mask_preq, torch.zeros((self.num_steps - 1, add_num_pedestrians), device=self.position.device)),
                dim=1)
        if (add_tau.numel() > 0):
            self.tau[-add_num_pedestrians:] = add_tau
        self.duration = torch.cat((self.duration, torch.zeros(add_num_pedestrians, device=self.position.device)), dim=0)

    def save_data(self, data_path: str):
        super().save_data(data_path)

    def to(self, device):
        for u, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                exec('self.' + u + '=' + 'self.' + u + '.to(device)')