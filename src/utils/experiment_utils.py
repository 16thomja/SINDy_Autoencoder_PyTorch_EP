from src.utils.model_utils import build_equation_labels, get_equation

def print_gov_eqs(net):
    labels = build_equation_labels(
        net.z_dim,
        net.poly_order,
        use_inverse=net.use_inverse,
        use_sine=net.use_sine,
        use_cosine=net.use_cosine,
        include_constant=net.include_constant
    )
    coefs = (net.threshold_mask * net.sindy_coefficients).detach().cpu().numpy()
    print(get_equation(labels, coefs[:,0], lhs="Xddot"))
    print(get_equation(labels, coefs[:,1], lhs="Yddot"))