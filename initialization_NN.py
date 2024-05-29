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


def train_init_NN(par: Parameters, device: torch.device):
    Lx, t, n = get_params(par.beam_par)
    E, rho, _, h = get_params(par.mat_par)
    my_beam = Beam(Lx, E, rho, h/1000, 40e-3, n)  # h: m

    prob = Prob_Solv_Modes(my_beam)
    gamma_max = 5  # gamma_max must be increased, because spatial eigenfrequencies increase, since the beam is very short

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

    w_0 = my_beam.phi[:, 0]
    w_dot_0 = np.zeros(len(w_0))

    my_In_Cond.pass_init_cond(w_0, w_dot_0)
    A, B = my_In_Cond.compute_coeff()

    t_lin = np.linspace(0, t, n)/t
    my_beam.calculate_solution(A, B, t_lin)
    w = my_beam.w

    def adimensionalize_sol(w: np.ndarray, w_ast: float):
        return w/w_ast

    w_ad = adimensionalize_sol(w, Lx)

    return w_ad
