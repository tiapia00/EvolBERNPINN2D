import numpy as np
import os
import torch
from torch import nn
from typing import Tuple
from beam import Beam, Prob_Solv_Modes, In_Cond
import pytz
import datetime
from read_write import get_current_time, get_last_modified_file, pass_folder
from tqdm import tqdm
from typing import Callable
from nn import *
from pinn import *
from par import Parameters, get_params
from initialization_NN import train_init_NN

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

retrain_init = False
retrain_PINN = False

par = Parameters()

if retrain_init:
    train_init_NN(par, device)

E, rho, _ , nu = get_params(par.mat_par)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_train, layers, dim_hidden, lr, epochs, weight_IN, weight_BOUND = get_params(par.pinn_par)

x_domain = np.array([0.0, Lx])/Lx
y_domain = np.array([0.0, Ly])/Lx
t_domain = np.array([0.0, T])/T

pinn = PINN(layers, dim_hidden, act=nn.Tanh()).to(device)

if retrain_PINN:
    
    path = pass_folder()
    
    loss_fn = Loss(
        x_domain,
        y_domain,
        t_domain,
        n_train,
        return_adim(x_domain, t_domain, rho, mu, lam),
        initial_conditions,
        weight_IN,
        weight_BOUND
    )
    
    filename_model = get_last_modified_file('in_model')
    pretrained_model_dict = torch.load(filename_model, map_location=torch.device(device))

    pretrained_model = NN(layers, dim_hidden, 2, 1)
    pretrained_model.load_state_dict(pretrained_model_dict)

    for i in np.arange(len(pinn.middle_layers)):
        pinn_layer = pinn.middle_layers[i]
        pretrained_layer = pretrained_model.middle_layers[i]
        pinn.middle_layers[i].weight.data.copy_(pretrained_model.middle_layers[i].weight)
        pinn.middle_layers[i].bias.data.copy_(pretrained_model.middle_layers[i].bias)

    pinn_trained, loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=lr, max_epochs=epochs, path=path)
    
    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(path, model_name)
    
    torch.save(pinn_trained.state_dict(), model_path)
    
else:
    pinn_trained = PINN(layers, dim_hidden, act=nn.Tanh()).to(device)
    
    filename = get_last_modified_file('model', '.pth')
    path = os.path.dirname(filename)
    
    pinn_trained.load_state_dict(torch.load(filename, map_location = device))
    print(f'{filename} loaded.\n')

    
pinn_trained.eval()


from plots import plot_initial_conditions, plot_uy

x, y, _ = get_initial_points(x_domain, y_domain, t_domain, n_train)
t_value = 0.0
t = torch.full_like(x, t_value)
x = x.to(device)
y = y.to(device)
t = t.to(device)
z = f(pinn_trained, x ,y, t)
ux0, uy0 = initial_conditions(x, y, Lx, i = 1)
z0 = torch.cat((ux0, uy0), dim=1)

plot_initial_conditions(z, z0, x, y, n_train, path)

x, y, t = get_interior_points(x_domain, y_domain, t_domain, n_train)
plot_uy(pinn_trained, x, y, t, n_train, path)


# # To be added
# - ~separate loss in more bars to see how the various loss term come to zero~
# - see if some quadrature rule has been implemented
# - scheme of weights initialization in order to automatically satisfy initial conditions
# - plots (in progress)
# - NN operators (to generalize results)
# - try to implement function that allows that satisfy initial conditions?
