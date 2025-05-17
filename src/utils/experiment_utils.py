from src.utils.model_utils import build_equation_labels, get_equation

def print_gov_eqs(net):
    labels = build_equation_labels(
        net.z_dim,
        net.poly_order,
        use_sine=True
    )
    coefs = (net.threshold_mask * net.sindy_coefficients).detach().cpu().numpy()
    print(get_equation(labels, coefs, lhs="Xddot"))
    print(get_equation(labels, coefs, lhs="Yddot"))