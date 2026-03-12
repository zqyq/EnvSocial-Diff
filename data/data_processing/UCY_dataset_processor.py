import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../../')
from data.dataset import RawData
from utils.visualization import state_animation
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='UCY dataset processor')
    parser.add_argument('-i', '--input', type=str, default="../UCY/",
                        help='input file path')
    parser.add_argument("--static", type=str, default="../UCY/")
    parser.add_argument('-o', '--output', type=str, default='../../data_origin/UCY_dataset',
                        help='output file path')
    parser.add_argument('-d', '--duration', type=float, default='60',
                        help='length of time snippet to save')
    parser.add_argument('-t', '--time', type=float, default='162',
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
    time_unit = 1.0/12.5
    savename = args.output + f"UCY_zara02_Dataset_time{time_range[0]}-{time_range[1]}_timeunit{time_unit:.2f}"
    to_begin = frame_range[0] / 25 / time_unit
    meta_data = {
        "time_unit": time_unit, 
        "version": "v2.2",
        "begin_time": time_range[0],
        "source": "UCY dataset"
    }
    ref_img_path = os.path.join("/mnt/d/SPD_2025/data/UCY/students03", "reference.png")
    if os.path.exists(ref_img_path):
        ref_img = cv2.imread(ref_img_path)
        h_img, w_img = ref_img.shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0
    else:
        # 如果没有 reference 图，你需要手动指定中心偏移（从你知道的分辨率）
        # 举例：cx, cy = 360, 288
        raise FileNotFoundError(f"reference image not found at {ref_img_path}; need it to get image center")

    M = np.array([[-2.595651699999999840e-02, -5.157280400000000145e-18, 7.838868099999996453e+00],
                  [-1.095387399999999886e-03, 2.166433000000000247e-02, 5.566045600000004256e+00],
                  [1.954012500000000506e-20, 4.217141000000002596e-19, 1.000000000000000444e+00]])
# ])

    # Uncomment to show the transformed scene as a picture
    # import cv2
    # image = cv2.imread('./cache/UCY/students_003.jpg')
    # post1 = np.float32([[166, 115], [561, 130], [132, 440], [602, 445]])
    # post2 = np.float32([[0, length],[width, length], [0, 0], [width, 0]])
    # M = cv2.getPerspectiveTransform(post1, post2 * 30)
    # cv2.imshow("image", cv2.warpPerspective(image, M, (width*30,length*30)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ss = 0
    trajectories = []
    print('Processing...')
    with open(args.input + "students03/annotation.vsp") as f:
        num_pedestrians = int(f.readline().split(' ')[0])
        trajs = []
        for i in tqdm(range(num_pedestrians)):
            S = int(f.readline().split(' ')[0])
            print(S)
            traj = np.zeros([S, 3])
            for j in range(S):
                traj[j, :] = np.array(f.readline().split(' ')[0:3], dtype=float)

            # Coordinate transformation
            image_coordination = np.concatenate((traj[:, 0:2], np.ones((traj.shape[0], 1))), axis=1)

            image_coordination = np.concatenate((traj[:, 0:2] + np.array([cx, cy]), np.ones((traj.shape[0], 1))),
                                                axis=1)

            world_coordination = np.einsum('ij,nj->ni', M, image_coordination)
            traj[:, 0] = world_coordination[:, 0] / world_coordination[:, 2]
            traj[:, 1] = world_coordination[:, 1] / world_coordination[:, 2]

            # Interpolate
            begin_frame, end_frame = int(traj[0, 2]), int(traj[-1, 2])
            sample_frame = np.arange(begin_frame, end_frame + 1, time_unit * 25)
            traj_ = np.zeros([len(sample_frame), 3])
            traj_[:, 2] = sample_frame
            try:
                traj_[:, 0] = interp1d(traj[:, 2], traj[:, 0], kind='cubic')(traj_[:, 2])
                traj_[:, 1] = interp1d(traj[:, 2], traj[:, 1], kind='cubic')(traj_[:, 2])
            except (ValueError): # traj_.shape[0] is too less to do high order interpolate
                traj_[:, 0] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 0])
                traj_[:, 1] = np.interp(traj_[:, 2], traj[:, 2], traj[:, 1])    
            # traj = [(x,y,int(f / time_unit / 25)) for x,y,f in traj_ if ((f >= frame_range[0]) and (f <= frame_range[1]))]
            traj = [(x, y, int(f / time_unit / 25 - to_begin)) for x, y, f in traj_ if
                    ((f >= frame_range[0]) and (f <= frame_range[1]))]
            if(traj):
                trajectories.append(traj)
                ss = ss+1


    destination = []
    for traj in trajectories:
        destination.append([(traj[-1][0], traj[-1][1], traj[-1][2])])

    data = np.array((meta_data, trajectories, destination, []), dtype=object)
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
