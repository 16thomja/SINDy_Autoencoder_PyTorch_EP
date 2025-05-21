import numpy as np
from scipy.integrate import solve_ivp


def get_elastic_pendulum_data(n_ics, timesteps, k, m, L, g):
    t,x,dx,ddx,z = generate_elastic_pendulum_data(n_ics, timesteps, k, m, L, g)
    data = {}
    data['t'] = t
    data['x'] = x.reshape((n_ics*t.size, -1))
    data['dx'] = dx.reshape((n_ics*t.size, -1))
    data['ddx'] = ddx.reshape((n_ics*t.size, -1))
    data['z'] = z.reshape((n_ics*t.size, -1))[:, 0:2]
    data['dz'] = z.reshape((n_ics*t.size, -1))[:, 2:4]

    return data


def generate_elastic_pendulum_data(n_ics, timesteps, k, m, L, g):
    def f(t, z):
        r, theta, r_dot, theta_dot = z
        r_ddot = r * theta_dot**2 + (k / m) * (L - r) + g * np.cos(theta)
        theta_ddot = -(g / r) * np.sin(theta) - (2 * r_dot * theta_dot) / r
        return [r_dot, theta_dot, r_ddot, theta_ddot]
    
    sigma = 0.05 # pendulum bob Gaussian blob decay
    quadrant_length = 3.0 # side length of each quadrant in movie

    dt = 0.02
    duration = timesteps * dt
    t_span = (0, duration)
    t_eval = np.arange(0, duration, dt)

    z = np.zeros((n_ics, t_eval.size, 4))
    dz = np.zeros_like(z)

    r_range = ([0.5, 2.0])
    theta_range = np.array([-np.pi, np.pi])
    r_dot_range = np.array([-1.0, 1.0])
    theta_dot_range = np.array([-2.0, 2.0])

    i = 0
    while i < n_ics:
        z0 = np.array([
            (r_range[1] - r_range[0]) * np.random.rand() + r_range[0],
            (theta_range[1] - theta_range[0]) * np.random.rand() + theta_range[0],
            (r_dot_range[1] - r_dot_range[0]) * np.random.rand() + r_dot_range[0],
            (theta_dot_range[1] - theta_dot_range[0]) * np.random.rand() + theta_dot_range[0]
        ])

        solution = solve_ivp(
            f, t_span, z0, method='Radau', t_eval=t_eval, args=()
        )
        z_trajectory = solution.y.T

        # physical constraints checks
        # throw away entire run if any one of these is violated
        if (
            np.any(np.isnan(z_trajectory)) or # no nans
            np.any(z_trajectory[:, 0] <= 0) or # positive spring length
            np.any(np.abs(z_trajectory[:, 0] > quadrant_length - np.sqrt(sigma))) or # keep bob in frame
            np.any(np.abs(z_trajectory[:, 1]) >= np.pi) # no flipovers
        ):
            continue

        z[i] = z_trajectory
        dz[i] = np.array([f(t_eval[j], z_trajectory[j]) for j in range(len(t_eval))])
        i += 1
        
    x,dx,ddx = elastic_pendulum_to_movie(z, dz, quadrant_length, sigma)

    return t_eval,x,dx,ddx,z


def elastic_pendulum_to_movie(z, dz, quadrant_length, sigma):
    n_ics = z.shape[0]
    n_samples = z.shape[1]
    n = 51
    y1, y2 = np.meshgrid(np.linspace(-quadrant_length, quadrant_length, n), np.linspace(quadrant_length, -quadrant_length, n))

    create_image = lambda r, theta: np.exp(-1 / sigma * ((y2 - r * np.cos(theta))**2 + (y1 - r * np.sin(theta))**2))

    argument_derivative_r = lambda r, theta, dr: -2 / sigma * ((y1 - r * np.cos(theta)) * np.cos(theta) * dr +
                                                              (y2 - r * np.sin(theta)) * np.sin(theta) * dr)

    argument_derivative_theta = lambda r, theta, dtheta: -2 / sigma * ((y1 - r * np.cos(theta)) * (-r * np.sin(theta)) * dtheta +
                                                                      (y2 - r * np.sin(theta)) * (r * np.cos(theta)) * dtheta)

    argument_derivative2_r = lambda r, theta, dr, ddr: -2 / sigma * ((np.cos(theta))**2 * dr**2 +
                                                                    (y1 - r * np.cos(theta)) * np.cos(theta) * ddr +
                                                                    (np.sin(theta))**2 * dr**2 +
                                                                    (y2 - r * np.sin(theta)) * np.sin(theta) * ddr)

    argument_derivative2_theta = lambda r, theta, dtheta, ddtheta: -2 / sigma * ((-r * np.sin(theta)) * (-r * np.sin(theta)) * dtheta**2 +
                                                                               (y1 - r * np.cos(theta)) * (-r * np.cos(theta)) * dtheta**2 +
                                                                               (y1 - r * np.cos(theta)) * (-r * np.sin(theta)) * ddtheta +
                                                                               (r * np.cos(theta)) * (r * np.cos(theta)) * dtheta**2 +
                                                                               (y2 - r * np.sin(theta)) * (r * np.sin(theta)) * dtheta**2 +
                                                                               (y2 - r * np.sin(theta)) * (r * np.cos(theta)) * ddtheta)

    # Initialize arrays for images and derivatives
    x = np.zeros((n_ics, n_samples, n, n))  # Images
    dx = np.zeros((n_ics, n_samples, n, n))  # First derivatives
    ddx = np.zeros((n_ics, n_samples, n, n))  # Second derivatives

    for i in range(n_ics):
        for j in range(n_samples):
            # State variables: r, theta
            r = z[i, j, 0]
            dr = dz[i, j, 0]
            ddr = dz[i, j, 2]

            theta = z[i, j, 1]
            theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # Wrap theta to [-pi, pi]
            dtheta = dz[i, j, 1]
            ddtheta = dz[i, j, 3]

            # Generate the image and its derivatives
            x[i, j] = create_image(r, theta)
            dx[i, j] = x[i, j] * (argument_derivative_r(r, theta, dr) + argument_derivative_theta(r, theta, dtheta))
            ddx[i, j] = x[i, j] * ((argument_derivative_r(r, theta, dr) + argument_derivative_theta(r, theta, dtheta))**2 +
                                   argument_derivative2_r(r, theta, dr, ddr) +
                                   argument_derivative2_theta(r, theta, dtheta, ddtheta))

    return x, dx, ddx
