import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.model_utils import library_size, sindy_library


class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()

        self.z_dim = args.z_dim
        self.u_dim = args.u_dim
        self.use_inverse = args.use_inverse
        self.use_sine = args.use_sine
        self.use_cosine = args.use_cosine
        self.poly_order = args.poly_order
        self.include_constant = args.include_constant
        self.library_dim = library_size(
            self.z_dim, 
            self.poly_order, 
            use_inverse=self.use_inverse,
            use_sine=self.use_sine,
            use_cosine=self.use_cosine,
            include_constant=self.include_constant
        )
        self.mse = nn.MSELoss(reduction='mean')

        self.u_w = int(np.sqrt(self.u_dim))

        self.encoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(1, self.u_w, self.u_w)),
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),   # 51 -> 26
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 26 -> 13
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 13 -> 7
            nn.ELU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 7 * 7, self.z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 64 * 7 * 7),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.Flatten(start_dim=1)
        )

        self.sindy_coefficients = nn.Parameter(torch.zeros(self.library_dim, self.z_dim, requires_grad=True))
        nn.init.normal_(self.sindy_coefficients, mean=0.0, std=1e-4)
        self.sequential_threshold = args.sequential_threshold
        self.threshold_mask = nn.Parameter(torch.ones_like(self.sindy_coefficients), requires_grad=False)


    def forward(self, x, dx, ddx, lambdas):
        device = self.sindy_coefficients.device

        # get latent state and reconstructed full state
        z = self.encoder(x)
        x_recon = self.decoder(z)

        # propagate full state derivatives through encoder to get latent derivatives
        dz, ddz = self.get_derivative_order2(x, dx, ddx, self.encoder)

        # build the SINDy library using the latent state + derivative
        theta = sindy_library(
            z, 
            dz, 
            self.poly_order, 
            device, 
            self.use_inverse, 
            self.use_sine, 
            self.use_cosine, 
            self.include_constant
        )

        # predict the second derivative of z using the library
        ddz_pred = self.predict(theta)

        # propagate predicted latent second derivative through decoder to get SINDy prediction of full space dynamics
        _, ddx_pred = self.get_derivative_order2(z, dz, ddz_pred, self.decoder)

        # calculate loss
        loss = self.loss_func(x, x_recon, ddx_pred, ddz_pred, ddx, ddz, lambdas)

        """
        zero = torch.tensor(0., device=device)

        alpha = 5.0
        mask = 1.0 + alpha * x
        l_recon = ((mask * (x_recon - x))**2).mean()

        loss = (l_recon, zero, zero, zero)
        """

        return loss
    

    def predict(self, theta):
        # sindy_coefficients: library_dim X z_dim
        theta = theta.unsqueeze(1) # (b * T) x L  --->   (b * T) x 1 x L
        masked_coeffs = self.sindy_coefficients * self.threshold_mask
        
        return torch.matmul(theta, masked_coeffs).squeeze() # (b x T) x z
    

    def get_derivative_order2(self, x, dx, ddx, net):
        v, dv, ddv = x, dx, ddx
        for layer in net:
            if isinstance(layer, (nn.Flatten, nn.Unflatten)):
                v, dv, ddv = layer(v), layer(dv), layer(ddv)

            elif isinstance(layer, nn.Linear):
                v = layer(v)
                dv = F.linear(dv, layer.weight, bias=None)
                ddv = F.linear(ddv, layer.weight, bias=None)
            
            elif isinstance(layer, nn.Conv2d):
                v = layer(v)
                dv = F.conv2d(dv, layer.weight, bias=None,
                              stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups)
                ddv = F.conv2d(ddv, layer.weight, bias=None,
                              stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups)
                
            elif isinstance(layer, nn.ConvTranspose2d):
                v = layer(v)
                dv = F.conv_transpose2d(dv, layer.weight, bias=None,
                              stride=layer.stride, padding=layer.padding,
                              output_padding=layer.output_padding,
                              dilation=layer.dilation, groups=layer.groups)
                ddv = F.conv_transpose2d(ddv, layer.weight, bias=None,
                              stride=layer.stride, padding=layer.padding,
                              output_padding=layer.output_padding,
                              dilation=layer.dilation, groups=layer.groups)

            elif isinstance(layer, nn.ELU):
                phi = F.elu(v, alpha=layer.alpha)
                dphi = torch.where(v > 0, torch.ones_like(v), layer.alpha * torch.exp(v))
                ddphi = torch.where(v > 0, torch.zeros_like(v), layer.alpha * torch.exp(v))

                # note order: ddv uses dv from prev layer before dv is updated
                ddv = ddphi * (dv ** 2) + dphi * ddv
                dv = dphi * dv
                v = phi

            else:
                raise TypeError(f"Unsupported layer type: {type(layer)}")
            
        return dv, ddv


    def loss_func(self, x, x_recon, ddx_pred, ddz_pred, ddx, ddz, lambdas):
        # reconstruction loss
        l_recon = self.mse(x_recon, x)
        
        # SINDy loss in ddx
        l_ddx = lambdas[0] * self.mse(ddx_pred, ddx)
        
        # SINDy loss in ddz
        l_ddz = lambdas[1] * self.mse(ddz_pred, ddz)
        
        # SINDy regularization
        l_reg = lambdas[2] * torch.abs(self.sindy_coefficients).mean()
        
        return l_recon, l_ddx, l_ddz, l_reg