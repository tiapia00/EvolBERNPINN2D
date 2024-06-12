import numpy as np
from beam import Beam, Prob_Solv_Modes, In_Cond
from par import Parameters, get_params
from scipy import integrate


def obtain_analytical_free(par: Parameters, my_beam: Beam, w0: float, t_ad_f: float, n: int):
    prob = Prob_Solv_Modes(my_beam)
    gamma_max = 5  # gamma_max must be increased, because spatial eigenfrequencies increase, since the beam is very short

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

    t_lin = np.linspace(0, t_ad_f, n)

    my_beam.calculate_solution_free(A, B, t_lin)
    w = my_beam.w

    V0_hat = calculate_ad_init_en(my_beam, t_ad)

    return t_ad, w, V0_hat

def obtain_analytical_forced(par, my_beam: Beam, load_dist: callable, t_ad_f: float, n: int):
    my_beam.calculate_beam_mat()

    my_beam.calculate_Q(load_dist)

    t = np.linspace(0, t_ad_f, n)
    sol = my_beam.calculate_solution_forced(t)

    return sol

def calculate_ad_init_en(my_beam: Beam, t_ad) -> float:
    Lx = my_beam.xi[-1]
    x_ad = my_beam.xi/Lx

    EJ = my_beam.E*my_beam.J
    w_ad = my_beam.w[:, 0]/Lx

    dw_dxx = df_num(x_ad, df_num(x_ad, w_ad))

    V = 1/2*EJ*integrate.simpson(y=dw_dxx**2, x=x_ad)
    V_ad = V/(my_beam.rho*Lx**2/t_ad**2)

    return V_ad


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
