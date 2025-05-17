import torch
import torch.nn as nn
from scipy.special import binom


def library_size(n, poly_order, use_sine=False, include_constant=True):
    """
    Compute number of terms in SINDy library for second-order system with 2*n inputs
    """
    # concat state and first derivative
    dim = 2 * n
    # polynomials
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(dim + k - 1, k))
    if use_sine:
        l += dim
    if not include_constant:
        l -= 1

    return l

def sindy_library(z, dz, poly_order, device, use_sine=False, include_constant=True):
    """
    Build SINDy library Theta for second order systems: [z, dz]
    z, dz: tensors of shape (batch, n)
    returns Theta of shape (batch, L)
    """
    # batch size, latent dim
    m, n = z.shape
    # combined state + derivative
    X = torch.cat((z, dz), dim=1) # shape (m, 2n)
    # library size
    L = library_size(n, poly_order, use_sine, include_constant)
    Theta = torch.ones((m, L), device=device)
    idx = 1 # skip constant at col 0

    # linear terms
    for i in range(2 * n):
        Theta[:, idx] = X[:, i]
        idx += 1
    
    # higher-order polynomials
    if poly_order >= 2:
        for i in range(2 * n):
            for j in range(i, 2 * n):
                Theta[:, idx] = X[:, i] * X[:, j]
                idx += 1
    if poly_order >= 3:
        for i in range(2 * n):
            for j in range(i, 2 * n):
                for k in range(j, 2 * n):
                    Theta[:, idx] = X[:, i] * X[:, j] * X[:, k]
                    idx += 1

    # sine
    if use_sine:
        for i in range(2 * n):
            Theta[:, idx] = torch.sin(X[:, i])
            idx += 1
    
    return Theta


def build_equation_labels(z_dim, poly_order, use_sine=False, include_constant=True):
    """
    Generate human readable term labels for SINDy library
    """
    labels = []
    dim = 2 * z_dim
    names = ['X', 'Y', 'Xdot', 'Ydot']

    if include_constant:
        labels.append('1')
    for i in range(dim):
        labels.append(names[i])
    if poly_order >= 2:
        for i in range(dim):
            for j in range(i, dim):
                labels.append(f"{names[i]}*{names[j]}")
    if poly_order >= 3:
        for i in range(dim):
            for j in range(i, dim):
                for k in range(j, dim):
                    labels.append(f"{names[i]}*{names[j]}*{names[k]}")
    if use_sine:
        for name in names:
            labels.append(f"sin({name})")
    
    return labels


def get_equation(labels, coefs, lhs=''):
    """
    Build human readable equation string from labels and corresponding coefficients
    lhs: variable name, e.g. "Xddot"
    """
    terms = []
    for label, coef in zip(labels, coefs):
        if abs(coef) > 1e-8:
            terms.append(f"{coef:.4g})*{label}")
    rhs = ' + '.join(terms) if terms else '0'
    if lhs:
        return f"{lhs} = {rhs}"
    
    return rhs


def init_weights(m):
    """
    Xavier init for linear layers
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)