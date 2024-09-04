import numpy as np
from beam import Beam, Prob_Solv_Modes, In_Cond
from par import Parameters, get_params
from scipy import integrate
import matplotlib.pyplot as plt


def obtain_analytical_free(my_beam: Beam, w0: float, t: np.ndarray):
    prob = Prob_Solv_Modes(my_beam)
    gamma_max = 1 # gamma_max must be increased, because spatial eigenfrequencies increase, since the beam is very short

    prob.pass_g_max(gamma_max)
    eig_gam = prob.find_eig()

    my_beam.gamma = np.array(eig_gam)
    my_beam.update_freq()

    # Just one parameter independent for gamma (order of the system reduced)
    F = prob.find_all_F(my_beam)
    prob.update_gamma(my_beam)
    phi = prob.return_modemat(F)
    my_beam.update_phi(phi)
    my_In_Cond = In_Cond(my_beam)

    omega0 = np.min(my_beam.omega)
    T_tild = 2*np.pi/omega0

    w0 = w0*my_beam.phi[:, 0]
    wdot_0 = np.zeros(len(w0))

    my_In_Cond.pass_init_cond(w0, wdot_0)
    A, B = my_In_Cond.compute_coeff()

    my_beam.calculate_solution_free(A, B, t)
    w = my_beam.w

    ens = getesnum(my_beam, t, w)

    return w, ens, T_tild


def getesnum(my_beam: Beam, ts: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = my_beam.xi
    EJ = my_beam.E*my_beam.J
    m = my_beam.rho * my_beam.A

    V = np.zeros_like(ts)
    T = np.zeros_like(ts)
    dw_dt = np.zeros_like(w)

    for i in range(dw_dt.shape[0]):
        dw_dt[i,:] = df_num(ts, w[i,:])

    for i in range(len(ts)):
        dw_dxx = df_num(x, df_num(x, w[:,i]))
        V[i] = 1/2*EJ*integrate.simpson(y=dw_dxx**2, x=x)
        T[i] = 1/2*m*integrate.simpson(y=dw_dt[:,i]**2, x=x)
    
    ens = np.stack([V, T], axis=1)

    return ens


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
