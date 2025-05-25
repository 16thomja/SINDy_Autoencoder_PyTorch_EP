import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp
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

        # encoder layers
        self.enc1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2) # 51 -> 26
        self.enc2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2) # 26 -> 13
        self.enc3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) # 13 -> 7
        self.fc_enc = nn.Linear(64 * 7 * 7, self.z_dim) # flatten -> fc -> z

        # decoder layers - mirror of encoder
        self.fc_dec = nn.Linear(self.z_dim, 64 * 7 * 7)
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=0)
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=0)

        self.sindy_coefficients = nn.Parameter(torch.zeros(self.library_dim, self.z_dim, requires_grad=True))
        nn.init.xavier_normal_(self.sindy_coefficients)
        self.sequential_threshold = args.sequential_threshold
        self.threshold_mask = nn.Parameter(torch.ones_like(self.sindy_coefficients), requires_grad=False)


    def encoder(self, x):
        # (b * T, u_dim) -> (b * T, 1, u_w, u_w)
        x = x.view(-1, 1, self.u_w, self.u_w)
        x = F.elu(self.enc1(x))
        x = F.elu(self.enc2(x))
        x = F.elu(self.enc3(x))
        x = x.view(x.size(0), -1) # flatten
        z = self.fc_enc(x)

        return z


    def decoder(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 64, 7, 7) # unflatten
        x = F.elu(self.dec1(x))
        x = F.elu(self.dec2(x))
        #x = torch.sigmoid(self.dec3(x))
        x = self.dec3(x)
        x = x.view(-1, self.u_dim)

        return x

    def forward(self, x, dx, ddx, lambdas):
        device = self.sindy_coefficients.device

        # reshape data to be (b * T) x u
        x = x.view(-1, self.u_dim).float().to(device)
        dx = dx.view(-1, self.u_dim).float().to(device)
        ddx = ddx.view(-1, self.u_dim).float().to(device)

        """
        # propogate state + derivatives through encoder to get latent state + derivatives
        z, dz, ddz = self.get_derivative_order2(x, dx, ddx, self.encoder)

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

        # propogate predicted latent second derivative through decoder to predict full space dynamics
        x_recon, _, ddx_pred = self.get_derivative_order2(z, dz, ddz_pred, self.decoder)

        # calculate loss
        loss = self.loss_func(x, x_recon, ddx_pred, ddz_pred, ddx, ddz, lambdas)
        """

        x_recon = self.decoder(self.encoder(x))
        zero = torch.tensor(0., device=device)

        alpha = 5.0
        mask = 1.0 + alpha * x
        l_recon = ((mask * (x_recon - x))**2).mean()

        loss = (l_recon, zero, zero, zero)

        return loss
    

    def predict(self, theta):
        # sindy_coefficients: library_dim X z_dim
        theta = theta.unsqueeze(1) # (b * T) x L  --->   (b * T) x 1 x L
        masked_coeffs = self.sindy_coefficients * self.threshold_mask
        
        return torch.matmul(theta, masked_coeffs).squeeze() # (b x T) x z
    

    def get_derivative_order2(self, x, dx, ddx, fn):
        # x:         (N, D_in)
        # dx:        (N, D_in)  = ∂x/∂t
        # ddx:       (N, D_in)  = ∂²x/∂t²
        # fn:        Callable that maps x -> y (N, D_out)
        x = x.requires_grad_(True) # track gradient wrt x
        y, dy = jvp(fn, (x,), (dx,), create_graph=True) # apply gradient to x, dx
        _, ddy = jvp(fn, (x,), (ddx,), create_graph=True) # apply gradient to ddx
        
        return y, dy, ddy


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