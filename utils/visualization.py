"""Utility functions for plots and animations."""
from matplotlib import patches

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data.dataset import RawData
import torch
from matplotlib.patches import Circle
num_pedestrians=107
num_steps=676
time_unit=0.08

def init_animation(ax, data, circle: dict={}, number: dict={}) -> dict:
    """Init entities needed in animation and return as a dict.
        - "title": Title of the figure
        - {ped_id}: The entities corresponding to pedestrian {ped_id}
    """
    actors = {}
    for ped_id in range(data.num_pedestrians):
        actors[ped_id] = {
            "circle": plt.Circle((0, 0), **circle, visible=False),
            "number": ax.text(0, 0, str(ped_id), **number, size="xx-small", visible=False, verticalalignment="center", horizontalalignment="center", color=(0, 0, 0, 1)),
            "legend": ax.text(0.1, 0.9 - 0.08 * ped_id, '', transform=ax.transAxes, visible=False),
        }
        ax.add_patch(actors[ped_id]["circle"])

    actors["title"] = plt.title('')
    # if(data.obstacles.numel()):
    #     plt.plot(data.obstacles[:, 0], data.obstacles[:, 1], "-k")
    actors["ax"] = ax
    # actors[ped]["circle"] = None
    return actors

def update_animation(frame_num: int, data: RawData, actors: dict, show_speed=False, color=None) -> list:
    frame = data.get_frame(frame_num)
    actors_list = []

    for ped_id in range(frame["num_pedestrians"]):
        if ped_id == 8 or ped_id == 9 or ped_id == 79:
            continue
        if frame["mask_p"][ped_id] == 0:
            actors[ped_id]["circle"].set_visible(False)
            actors[ped_id]["number"].set_visible(False)
            if show_speed:
                actors[ped_id]["legend"].set_visible(False)
            continue

        speed = torch.norm(frame["velocity"][ped_id]).item()
        acc = torch.norm(frame["acceleration"][ped_id]).item()
        pos = frame["position"][ped_id].cpu().numpy()
        radius = 5

        if color:
            color_ = color(frame)
        else:
            color_ = (1.0, 0.55, 0.0, 1.0)

        # 更新空心圆的位置和颜色
        circle = actors[ped_id]["circle"]
        circle.set_center(pos)
        circle.set_radius(radius)
        circle.set_edgecolor(color_)
        circle.set_facecolor(color_)
        circle.set_visible(True)

        actors[ped_id]["number"].set(position=pos, visible=True)

        if show_speed:
            actors[ped_id]["legend"].set(text=f'$v_{{{ped_id}}} = {speed:.2f}m/s, a_{{{ped_id}}} = {acc:.2f}m/s^2$', visible=True)

        actors_list.extend([
            actors[ped_id]["circle"],
            actors[ped_id]["number"],
        ])
        if show_speed:
            actors_list.append(actors[ped_id]["legend"])

    # 更新标题
    title_text = f'Frame {frame_num} / {frame_num * data.meta_data["time_unit"]:.2f}s'
    actors["title"].set(text=title_text)
    actors_list.append(actors["title"])

    return actors_list




def state_animation(ax, data:RawData, *, movie_file=None, writer=None, show_speed=False):
    """Generate animation for {data}."""
    if(movie_file): print(f"Saving animation to '{movie_file}'...")
    actors = init_animation(ax, data)

    def update(i):
        progress = round(i / data.num_steps * 100)
        print("\r", end="")
        print("Animation progress: {}%: ".format(progress), end="")
        sys.stdout.flush()
        return update_animation(i, data, actors, show_speed)

    ani = animation.FuncAnimation(
        ax.get_figure(), update,
        frames=data.num_steps,
        # frames=10,
        interval=data.meta_data["time_unit"] * 1000.0, blit=True)
    if movie_file:
        ani.save(movie_file, writer=writer, dpi=200)
    return ani


def state_animation_compare(ax, data1:RawData, data2:RawData, *, movie_file=None, writer=None, show_speed=False):
    """Generate animation to compare {data1} and {data2}.
        - data1: Data to compare, draw in colorful disks.
        - data2: Data as base, draw in black and white circle.

        Note: data1 and data2 should have same time unit.
    """
    if(movie_file): print(f"Saving compare animation to '{movie_file}'...")
    actors1 = init_animation(ax, data1, circle={"zorder":9}, number={"zorder":10})
    actors2 = init_animation(ax, data2, circle={"zorder":7}, number={"zorder":8, "alpha":0.2})

    def update(i):
        progress = round(i / data2.num_steps * 100)
        print("\r", end="")
        print("Animation progress: {}%: ".format(progress), end="")
        sys.stdout.flush()
        return update_animation(i, data1, actors1, show_speed) \
        + update_animation(i, data2, actors2, show_speed, color=lambda x: (1.0, 0.55, 0.0, 1.0))

    ani = animation.FuncAnimation(
        ax.get_figure(), update,
        frames=data2.num_steps,
        interval=data2.meta_data["time_unit"] * 1000.0, blit=True)
    if movie_file:
        ani.save(movie_file, writer=writer, dpi=200)
    return ani


def plot_trajectory(p_pred, p_label, name):
    #**
    p_pred = p_pred.cpu()
    p_label = p_label.cpu()
    plt.figure()
    #** p_pred, p_label**,**
    for t in range(p_pred.shape[0]):
        for n in range(p_pred.shape[1]):
            if not np.isnan(p_pred[t, n, 0]):
                if t>=25 and not np.isnan(p_pred[t-25, n, 0]):
                    plt.scatter(p_pred[t, n, 0], p_pred[t, n, 1], c='r', marker='o')
                else:
                    plt.scatter(p_pred[t, n, 0], p_pred[t, n, 1], c='r', marker='x')
                if t>0 and not np.isnan(p_pred[t-1, n, 0]):
                    plt.plot(p_pred[t-1:t+1, n, 0], p_pred[t-1:t+1, n, 1], c='r', linestyle='--')
            if not np.isnan(p_label[t, n, 0]):
                if t>=25 and not np.isnan(p_label[t-25, n, 0]):
                    plt.scatter(p_label[t, n, 0], p_label[t, n, 1], c='b', marker='o')
                else:
                    plt.scatter(p_label[t, n, 0], p_label[t, n, 1], c='b', marker='x')
                if t>0 and not np.isnan(p_label[t-1, n, 0]):
                    plt.plot(p_label[t-1:t+1, n, 0], p_label[t-1:t+1, n, 1], c='b', linestyle='--')
    f = plt.gcf()  
    f.savefig('visualize_data/'+name+'.jpg')
    f.clear()  
