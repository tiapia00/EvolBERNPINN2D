from plots import plot_initial_conditions, plot_sol, plot_midpoint_displ
import numpy as np
import os
import torch
from torch import nn
from typing import Tuple
from beam import Beam, Prob_Solv_Modes, In_Cond
import pytz
import datetime
from read_write import get_current_time, get_last_modified_file, pass_folder, delete_old_files
from tqdm import tqdm
from typing import Callable
from nn import *
from pinn import *
from par import Parameters, get_params
from initialization_NN import train_init_NN

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retrain_PINN = True
delete_old = False

if delete_old:
    delete_old_files("model")
    delete_old_files("in_model")

par = Parameters()

E, rho, _, nu = get_params(par.mat_par)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_train, layers, dim_hidden, lr, epochs = get_params(par.pinn_par)

x_domain = np.array([0.0, Lx])/Lx
y_domain = np.array([0.0, Ly])/Lx
t_domain = np.array([0.0, T])/T

points = {
            'res_points': get_interior_points(x_domain, y_domain,
                                              t_domain, n_train, device),
            'initial_points': get_initial_points(x_domain, y_domain,
                                                 t_domain, n_train, device),
            'boundary_points': get_boundary_points(x_domain, y_domain,
                                                   t_domain, n_train, device)
        }

pinn = PINN(dim_hidden, points, act=nn.Tanh()).to(device)

if retrain_PINN:

    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    loss_fn = Loss(
        x_domain,
        y_domain,
        t_domain,
        n_train,
        return_adim(x_domain, t_domain, rho, mu, lam),
        initial_conditions,
        points
    )

    pinn_trained, loss_values = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
                                            max_epochs=epochs, path_logs=dir_logs, points=points, n_train=n_train)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, points, act=nn.Tanh()).to(device)

print(pinn_trained)

pinn_trained.eval()

x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_train, device)
t_value = 0.0
t = torch.full_like(x, t_value)
x = x.to(device)
y = y.to(device)
t = t.to(device)
z = f(pinn_trained, x, y, t)
ux0, uy0 = initial_conditions(x, y, Lx, i=1)
z0 = torch.cat((ux0, uy0), dim=1)

plot_initial_conditions(z, z0, x, y, n_train, dir_model)

x, y, t = get_interior_points(x_domain, y_domain, t_domain, n_train, device)
plot_sol(pinn_trained, x, y, t, n_train, dir_model, 'NN prediction')
plot_midpoint_displ(pinn_trained, t, n_train, t_ad, uy_mid, dir_model)
