import numpy as np
from cmd_line import parse_args
from src.utils.other import *#load_data, load_model, make_model, get_tb_path, get_checkpoint_path, get_args_path, get_experiments_path
import torch
from src.dataset.Datasets import *
import matplotlib
matplotlib.use("Agg") # necessary on some systems to render plots in background
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import subprocess
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
from functools import partial
import subprocess

def process_frame(x_frame, z, timestep, frames_folder):
    z_plot = z[:timestep+1]
    z_plot_alpha = [0.2]*timestep + [1.0]
    z1_min = np.min(z[:,0])
    z1_max = np.max(z[:,0])
    z2_min = np.min(z[:,1])
    z2_max = np.max(z[:,1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    axes[0].imshow(x_frame.reshape((51, 51)), cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[0].set_title("x")
    axes[1].scatter(z_plot[:,0], z_plot[:,1], alpha=z_plot_alpha)
    axes[1].set_title("z_true")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("theta")
    axes[1].set_xlim(z1_min, z1_max)
    axes[1].set_ylim(z2_min, z2_max)

    # Save the figure
    fig.savefig(frames_folder + f't{timestep}.png', bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()

    cp_path, cp_folder = get_checkpoint_path(args)
    
    frames_folder = cp_folder + "/frames_z_true/"

    if not os.path.isdir(frames_folder):
        os.system("mkdir -p " + frames_folder)

    _, data_paths = get_data_paths()

    test_set = np.load(data_paths[2], allow_pickle=True).item()

    sample_index = 1

    x = test_set['x'].reshape(-1, args.timesteps, args.u_dim)[sample_index]
    z = test_set['z'].reshape(-1, args.timesteps, args.z_dim)[sample_index]

    with ThreadPoolExecutor(max_workers=32) as executor:
        process_frame_with_folder = partial(process_frame, frames_folder=frames_folder)

        list(tqdm(executor.map(
            process_frame_with_folder, 
            x,
            np.tile(z, (args.timesteps, 1, 1)),
            range(args.timesteps)
        ), total=args.timesteps))

    subprocess.call([
        'ffmpeg', '-y', '-framerate', '60', '-i', frames_folder + 't%d.png', '-r', '60', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        cp_folder + 'sample_z_true.mp4'
    ])

if __name__ == "__main__":
    main()