import torch
import torch.nn as nn
from scipy.special import binom


def library_size(n, poly_order, use_inverse=False, use_sine=False, use_cosine=False, include_constant=True):
    """
    Compute number of terms in SINDy library for second-order system with 2*n inputs
    """
    # concat state and first derivative
    dim = 2 * n
    # optional inverse of state
    if use_inverse:
        dim += n
    # optional sine of state
    if use_sine:
        dim += n
    # optional cosine of state
    if use_cosine:
        dim += n
    
    # combinatorial polynomials on all terms above
    l = 0
    for k in range(poly_order + 1):
        l += int(binom(dim + k - 1, k))
    if not include_constant:
        l -= 1

    return l

def sindy_library(z, dz, poly_order, device, use_inverse=False, use_sine=False, use_cosine=False, include_constant=True):
    """
    Build SINDy library Theta for second order systems: [z, dz]
    z, dz: tensors of shape (batch, n)
    returns Theta of shape (batch, L)
    """
    # batch size, latent dim
    m, n = z.shape

    # start with just state and derivative
    X_parts = [z, dz]
    # optionally add inverses for state
    if use_inverse:
        X_parts.append(1.0 / (z + 1e-6))
    # optionally add sine for state
    if use_sine:
        X_parts.append(torch.sin(z))
    # optionally add cosine for state
    if use_cosine:
        X_parts.append(torch.cos(z))

    X = torch.cat(X_parts, dim=1)
    # number of terms to be combined in polynomial builder loop
    D = X.shape[1]

    # library size
    L = library_size(n, poly_order, use_inverse, use_sine, use_cosine, include_constant)
    Theta = torch.zeros((m, L), device=device)
    idx = 0

    # constant
    if include_constant:
        Theta[:, idx] = 1
        idx += 1

    # 1st order
    for i in range(D):
        Theta[:, idx] = X[:, i]
        idx += 1
    
    # higher order
    for order in range(2, poly_order+1):
        def rec_build(start, curr_term, depth):
            nonlocal idx
            if depth == 0:
                Theta[:, idx] = curr_term
                idx += 1
                return
            for j in range(start, D):
                rec_build(j, curr_term * X[:, j], depth-1)
        
        rec_build(0, torch.ones(m, device=device), order)
    
    return Theta


def build_equation_labels(z_dim, poly_order,
                          use_inverse=False, use_sine=False, use_cosine=False,
                          include_constant=True):
    """
    Generate human-readable labels for the recursive SINDy library builder.
    Matches the order of terms produced by sindy_library rec_build.
    """
    state_names = ['X', 'Y'][:z_dim]
    deriv_names = [name + 'dot' for name in state_names]

    names = state_names + deriv_names

    if use_inverse:
        inv_names = [f"1/{name}" for name in state_names]
        names += inv_names
    if use_sine:
        sin_names = [f"sin({name})" for name in state_names]
        names += sin_names
    if use_cosine:
        cos_names = [f"cos({name})" for name in state_names]
        names += cos_names

    D = len(names)
    labels = []
    idx = 0

    # constant
    if include_constant:
        labels.append('1')
        idx += 1

    # first order
    for i in range(D):
        labels.append(names[i])
        idx += 1

    # higher order
    def rec_label(start, curr_label, depth):
        nonlocal idx
        if depth == 0:
            labels.append(curr_label)
            idx += 1
            return
        for j in range(start, D):
            new_label = curr_label + '*' + names[j] if curr_label else names[j]
            rec_label(j, new_label, depth - 1)

    for order in range(2, poly_order + 1):
        rec_label(0, '', order)

    return labels


def get_equation(labels, coefs, lhs=''):
    """
    Build human readable equation string from labels and corresponding coefficients
    lhs: variable name, e.g. "Xddot"
    """
    terms = []
    for label, coef in zip(labels, coefs):
        if abs(coef) > 1e-8:
            terms.append(f"({coef:.4g})*{label}")
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