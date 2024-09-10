import numpy as np
from beam import Beam, Prob_Solv_Modes, In_Cond
from par import Parameters, get_params
from scipy import integrate
import matplotlib.pyplot as plt


def obtain_analytical_free(par: Parameters, my_beam: Beam, w0: float, tf: float,
                           n_time: int):

    prob = Prob_Solv_Modes(my_beam)
    gamma_max = 5/1000 # gamma_max must be increased, because spatial eigenfrequencies increase, since the beam is very short

    prob.pass_g_max(gamma_max)
    eig_gam = prob.find_eig()

    my_beam.gamma = np.array(eig_gam)
    my_beam.update_freq()

    omega_1 = my_beam.omega[0]
    t_ad = 2*np.pi/omega_1

    # Just one parameter independent for gamma (order of the system reduced)
    F = prob.find_all_F(my_beam)
    prob.update_gamma(my_beam)
    phi = prob.return_modemat(F)
    my_beam.update_phi(phi)
    my_In_Cond = In_Cond(my_beam)

    w0 = w0*my_beam.phi[:, 0]
    wdot_0 = np.zeros(len(w0))

    my_In_Cond.pass_init_cond(w0, wdot_0)
    A, B = my_In_Cond.compute_coeff()

    t_lin = np.linspace(0, tf, n_time)

    my_beam.calculate_solution_free(A, B, t_lin)
    w = my_beam.w

    V0 = calculate_init_en(my_beam)

    return t_ad, w, V0


def calculate_init_en(my_beam: Beam) -> float:
    x = my_beam.xi

    EJ = my_beam.E*my_beam.J
    w = my_beam.w[:,0]

    dw_dxx = df_num(x, df_num(x, w))

    V0 = 1/2*EJ*integrate.simpson(y=dw_dxx**2, x=x)

    return V0


def obtain_max_stress(my_beam: Beam, w: np.ndarray):
    w0 = w[:,0]
    x = my_beam.xi
    E = my_beam.E
    eps_max = -my_beam.H/2 * np.max(df_num(x, df_num(x, w0)))
    sigma_max = E*eps_max

    return sigma_max


def df_num(x: np.ndarray, y: np.ndarray):
    dx = np.diff(x)
    dy = np.diff(y)

    derivative = np.zeros_like(y)

    # Forward difference for the first point
    derivative[0] = dy[0] / dx[0]

    # Central difference for the middle points
    for i in range(1, len(x) - 1):
        dx_avg = (x[i+1] - x[i-1]) / 2
        dy_avg = (y[i+1] - y[i-1]) / 2
        derivative[i] = dy_avg / dx_avg

    # Backward difference for the last point
    derivative[-1] = dy[-1] / dx[-1]

    return derivative
