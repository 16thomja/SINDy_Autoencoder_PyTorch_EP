import numpy as np
from cmd_line import parse_args
from src.utils.other import *#load_data, load_model, make_model, get_tb_path, get_checkpoint_path, get_args_path, get_experiments_path
import torch
from src.models.SINDyAE_o2 import Net
from src.dataset.Datasets import *
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import subprocess
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
from functools import partial
import subprocess

def process_frame(x_frame, x_recon_frame, timestep, frames_folder):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    axes[0].imshow(x_frame.reshape((51, 51)), cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[0].axis('off')
    axes[0].set_title("x")
    axes[1].imshow(x_recon_frame.reshape((51, 51)), cmap='gray', origin='lower', vmin=0, vmax=1)
    axes[1].axis('off')
    axes[1].set_title("x_recon")

    # Save the figure
    fig.savefig(frames_folder + f't{timestep}.png')
    plt.close(fig)

def main():
    args = parse_args()

    cp_path, cp_folder = get_checkpoint_path(args)
    
    frames_folder = cp_folder + "/frames/"

    if not os.path.isdir(frames_folder):
        os.system("mkdir -p " + frames_folder)

    train_set, val_set, test_set = load_data(args)

    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    net = Net(args)
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)

    x = test_set.x[0].view(-1, net.u_dim).type(torch.FloatTensor).to(device)

    with torch.no_grad():
        x_recon = net.decoder(net.encoder(x))

    x = x.detach().cpu().numpy()
    x_recon = x_recon.detach().cpu().numpy()

    x_recon_flat = x_recon.flatten().reshape(-1, 1)

    x_recon = MinMaxScaler(feature_range=(0, 1)).fit_transform(x_recon_flat).reshape(x_recon.shape)

    with ThreadPoolExecutor(max_workers=32) as executor:
        process_frame_with_folder = partial(process_frame, frames_folder=frames_folder)

        list(tqdm(executor.map(
            process_frame_with_folder, 
            x,
            x_recon, 
            range(args.timesteps)
        ), total=args.timesteps))

    subprocess.call([
        'ffmpeg', '-y', '-framerate', '60', '-i', frames_folder + 't%d.png', '-r', '60', '-pix_fmt', 'yuv420p',
        cp_folder + 'sample_autoencoder_output.mp4'
    ])

if __name__ == "__main__":
    main()