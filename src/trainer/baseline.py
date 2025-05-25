import torch
from tqdm import tqdm
import numpy as np

def normalize(img):
    img = torch.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    mn = img.min()
    mx = img.max()
    if mx > mn:
        # Normal case: spread into [0,1]
        img = (img - mn) / (mx - mn)
    else:
        # All pixels identical: just make a zero image
        img = torch.zeros_like(img)
    return img

def train(net, train_loader, train_board, optim, epoch, clip, lambdas):
    net.train()

    # losses
    epoch_l_recon = 0
    epoch_l_ddx = 0
    epoch_l_ddz = 0
    epoch_l_reg = 0

    # for each batch
    for x, dx, ddx, dz in tqdm(train_loader, desc="Training", total=len(train_loader), dynamic_ncols=True):
        l_recon, l_ddx, l_ddz, l_reg = net(x, dx, ddx, lambdas)
        epoch_l_recon += l_recon.item()
        epoch_l_ddx += l_ddx.item()
        epoch_l_ddz += l_ddz.item()
        epoch_l_reg += l_reg.item()

        # backprop
        batch_loss = l_recon + l_ddx + l_ddz + l_reg
        optim.zero_grad()
        batch_loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        optim.step()

        # update the mask
        net.threshold_mask[torch.abs(net.sindy_coefficients) < net.sequential_threshold] = 0
    
    # average
    num_batches = len(train_loader)
    epoch_l_recon /= num_batches
    epoch_l_ddx /= num_batches
    epoch_l_ddz /= num_batches
    epoch_l_reg /= num_batches

    # tensorboard
    train_board.add_scalar('L recon', epoch_l_recon, epoch)
    train_board.add_scalar('L ddx', epoch_l_ddx, epoch)
    train_board.add_scalar('L ddz', epoch_l_ddz, epoch)
    train_board.add_scalar('L regularization', epoch_l_reg, epoch)


def test(net, test_loader, test_board, epoch, timesteps, lambdas):
    net.eval()
    total_recon, total_ddx, total_ddz, total_reg = 0, 0, 0, 0
    for batch_idx, (x, dx, ddx, dz) in enumerate(tqdm(test_loader, desc="Testing", total=len(test_loader), dynamic_ncols=True)):
        l_recon, l_ddx, l_ddz, l_reg = net(x, dx, ddx, lambdas)
        total_recon += l_recon.item()
        total_ddx += l_ddx.item()
        total_ddz += l_ddz.item()
        total_reg += l_reg.item()

        # log a visual sample from first batch to TensorBoard
        if batch_idx == 0:
            batch_size, T, _ = x.shape
            device = torch.cuda.current_device()
            
            # reshape to (b * t) x u
            x = x.view(-1, net.u_dim).float().to(device)
            
            with torch.no_grad():
                reconstructed = net.decoder(net.encoder(x))

            input_sample = x[0].view(1, net.u_w, net.u_w)
            reconstructed_sample = reconstructed[0].view(1, net.u_w, net.u_w)

            #input_sample = normalize(input_sample)
            #reconstructed_sample = normalize(reconstructed_sample)

            test_board.add_image('Input Sample', input_sample, epoch, dataformats='CHW')
            test_board.add_image('Reconstructed Sample', reconstructed_sample, epoch, dataformats='CHW')
            test_board.add_histogram('Reconstructed Sample Values', reconstructed_sample, epoch)
    
    num_batches = len(test_loader)
    test_board.add_scalar('L recon', total_recon / num_batches, epoch)
    test_board.add_scalar('L ddx', total_ddx / num_batches, epoch)
    test_board.add_scalar('L ddz', total_ddz / num_batches, epoch)
    test_board.add_scalar('L regularization', total_reg / num_batches, epoch)
