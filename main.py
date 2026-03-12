import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
import subprocess
import atexit
from pathlib import Path
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from cmd_line import parse_args
from src.trainer.baseline import train, test
from src.utils.other import *#load_data, load_model, make_model, get_tb_path, get_checkpoint_path, get_args_path, get_experiments_path
from src.utils.model_utils import init_weights


def main():
    # get and save args
    args = parse_args()

    # train and val data (will refer to the val data as test data)
    train_set, test_set, _ = load_data(args)

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    train_name, test_name = get_tb_path(args)
    tb_root = str(Path(train_name).parent)
    os.makedirs(tb_root, exist_ok=True)

    tb_proc = None
    if args.launch_tensorboard == 1:
        try:
            cmd = [
                "tensorboard",
                "--logdir",
                tb_root,
                "--host",
                args.tensorboard_host,
                "--port",
                str(args.tensorboard_port),
            ]
            tb_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            atexit.register(lambda: tb_proc.terminate() if tb_proc.poll() is None else None)
            print(f"TensorBoard running (logdir={tb_root})")
            print(f"  URL: http://{args.tensorboard_host}:{args.tensorboard_port}")
        except FileNotFoundError:
            print("WARNING: `tensorboard` not found on PATH; skipping auto-launch.")

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # checkpoint, args, experiments path
    cp_path, cp_folder = get_checkpoint_path(args)
    args_path, args_folder = get_args_path(args)
    exp_folder = get_experiments_path(args)
    if not os.path.isdir(cp_folder):
        os.system("mkdir -p " + cp_folder)
    if not os.path.isdir(exp_folder):
        os.system("mkdir -p " + exp_folder)
    if args.print_folder == 1:
        print("Checkpoints saved at:        ", cp_folder)
        print("Experiment results saved at: ", exp_folder)
        print("Tensorboard logs saved at:   ", train_name[:-5])

    # save args
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # create model, optim, scheduler, initial epoch
    net = make_model(args).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.adam_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    initial_e = 0
    
    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        net, optim, scheduler, initial_e = load_model(net, cp_path, device, optim, scheduler)
    else:  # init network
        net.apply(init_weights)

    # boards
    train_board = SummaryWriter(train_name, purge_step=initial_e)
    test_board = SummaryWriter(test_name, purge_step=initial_e)

    # lambdas for loss function
    lambdas = args.lambda_ddx, args.lambda_ddz, args.lambda_reg

    # for each epoch
    for epoch in tqdm(range(args.epochs), desc="Epoch", total=args.epochs, dynamic_ncols=True):
        # train
        train(net, train_loader, train_board, optim, epoch + initial_e, args.clip, lambdas)

        # test
        if (epoch + 1) % args.test_interval == 0:
            test(net, test_loader, test_board, epoch + initial_e, args.timesteps, lambdas)
        
        # step on learning rate scheduler
        scheduler.step()
    
        # save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {'epoch': epoch + initial_e,
                          'model': net.state_dict(),
                          'optimizer': optim.state_dict(),
                          'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, cp_path)
        

if __name__ == "__main__":
    main()
