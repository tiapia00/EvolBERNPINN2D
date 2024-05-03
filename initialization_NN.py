import numpy as np
import torch
from torch import nn
from typing import Tuple
from beam import Beam, Prob_Solv_Modes, In_Cond
from par import Parameters, get_params
import os
import pytz
import datetime
from read_write import get_current_time
from tqdm import tqdm
from typing import Callable
import pytz
from nn import *

def train_init_NN(par: Parameters):
    Lx, t, n, num_hidden, dim_hidden, lr, epochs = get_params(par.nn_par)
    E, rho, _, h = get_params(par.mat_par)
    my_beam = Beam(Lx, E, rho, h/1000, 40e-3, n) # h: m

    prob = Prob_Solv_Modes(my_beam)
    gamma_max = 100 # gamma_max must be increased, because spatial eigenfrequencies increase, since the beam is very short

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

    w_0 = my_beam.phi[:,0]
    w_dot_0 = np.zeros(len(w_0))

    my_In_Cond.pass_init_cond(w_0, w_dot_0)
    A, B = my_In_Cond.compute_coeff()

    t_lin = np.linspace(0, 1, n)
    my_beam.calculate_solution(A, B, t_lin)
    w = my_beam.w


    def adimensionalize_sol(w: np.ndarray, w_ast: float):
        return w/w_ast

    w_ad = adimensionalize_sol(w, Lx).T


    # correct to transpose it? based on how reshape works
    # - columns of w are referred to different time points

    # # Setup NN for initial conditions

    # ## Define training dataset

    x_lin = np.linspace(0, 1, n)

    x, t = np.meshgrid(x_lin, t_lin)
    x = x.reshape(-1)
    t = t.reshape(-1)
    X = np.stack((x, t), axis=1)

    # ## Split data for validation

    y = w.reshape(-1, 1)

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # ## Prepare data for pytorch

    x_train = X_train[:, 0].reshape(-1, 1)
    t_train = X_train[:, 1].reshape(-1, 1)

    x_val = X_val[:, 0].reshape(-1, 1)
    t_val = X_val[:, 1].reshape(-1, 1)

    x = torch.tensor(x_train).to(device).float()
    t = torch.tensor(t_train).to(device).float()

    x_val = torch.tensor(x_val).to(device).float()
    t_val = torch.tensor(t_val).to(device).float()

    y = torch.tensor(y_train).to(device).float()
    y_val = torch.tensor(y_val).to(device).float()

    nn_init = NN(num_hidden, dim_hidden, dim_input = 2, dim_output = 1).to(device)

    loss_fn = Loss_NN(
        x,
        t,
        y
    )

    nn_trained, loss_values = train_model_nn(
        nn_init, loss_fn=loss_fn, learning_rate=lr, max_epochs=epochs, x_val=x_val, t_val=t_val, y_val=y_val)

    folder = 'in_model'
    time = get_current_time(fmt='%m-%d %H:%M')
    model_name = f'{get_current_time()}.pth'

    model_path = os.path.join(folder, model_name)
    os.makedirs(folder, exist_ok=True)

    torch.save(nn_trained.state_dict(), model_path)

# # To be checked
# - ~correct association of meshpoints with solution?~
