import numpy as np
import torch
from torch import nn
from typing import Tuple
from beam import Beam, Prob_Solv_Modes, In_Cond
from par import Parameters, get_params
import os
import pytz
import datetime
from read_write import get_current_time, pass_folder
from tqdm import tqdm
from typing import Callable
import pytz
from nn import *
from scipy import integrate


def obtain_analytical_trv(par: Parameters):
    Lx, t, n, w0 = get_params(par.beam_par)
    E, rho, _, h = get_params(par.mat_par)
    my_beam = Beam(Lx, E, rho, h/1000, 40e-3, n)  # h: m

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

    w_0 = w0*my_beam.phi[:, 0]
    w_dot_0 = np.zeros(len(w_0))

    my_In_Cond.pass_init_cond(w_0, w_dot_0)
    A, B = my_In_Cond.compute_coeff()

    t_lin = np.linspace(0, t, n)
    my_beam.calculate_solution(A, B, t_lin)
    w = my_beam.w

    w_ad = w/Lx

    V0_hat = calculate_ad_init_en(my_beam, t_ad)

    return t_ad, w_ad, V0_hat


def calculate_an_init_en(my_beam: Beam, t_ad) -> float:
    Lx = my_beam.xi[-1]
    x_ad = my_beam.xi/Lx

    EJ = my_beam.E*my_beam.J
    w_ad = my_beam.w/Lx

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
