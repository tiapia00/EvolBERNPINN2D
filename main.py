from plots import *
from beam import Beam
import numpy as np
import os
import torch
from utils import *
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free
from scipy.interpolate import make_interp_spline
import scipy.fft as fft

torch.set_default_dtype(torch.float32)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

load = True
train = False 

def get_step(tensors: tuple):
    a, b, c = tensors

    step_a = torch.diff(a)[0]
    step_b = torch.diff(b)[0]
    step_c = torch.diff(c)[0]

    return (step_a, step_b, step_c)

par = Parameters()

Lx, t, h, n_space_beam, n_time, w0 = get_params(par.beam_par)
E, rho, _ = get_params(par.mat_par)
my_beam = Beam(Lx, E, rho, h, h/3, n_space_beam)

t_beam, t_tild, w, V_an, Ek_an = obtain_analytical_free(my_beam, w0, t, 1000, 1)

interpdisplbeam = make_interp_spline(t_beam, w[w.shape[0]//2,:])
interpVbeam = make_interp_spline(t_beam, V_an, k=5)
interpTbeam = make_interp_spline(t_beam, Ek_an, k=5)

lam, mu = par.to_matpar_PINN()

Lx, Ly, tmax, n_space, n_time, w0, dim_hidden, n_hidden, multux, multuy, multhyperx, lr, epochs = get_params(par.pinn_par)
L_tild = Lx
x_domain = torch.linspace(0, Lx, n_space)/Lx
y_domain = torch.linspace(0, Ly, n_space)/Lx
t_domain = torch.linspace(0, tmax, n_time)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, multhyperx, y_domain, t_domain, device)
scaley = 2

points = {
    'res_points': grid.get_interior_points_train(scaley),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.generate_grid_bound(),
    'initial_points_hyper': grid.get_initial_points_hyper(),
    'all_points_eval': grid.get_all_points_eval(),
}

adim = (mu/lam, (lam+mu)/lam, rho/(lam*t_tild.item()**2)*Lx**2)
par = {"Lx": Lx,
        "w0": w0,
        "b": h/3,
        "lam": lam,
        "mu":mu,
        "rho": rho,
        "t_ast": t_tild}

inpoints = torch.cat(points["initial_points"], dim=1)
spacein = inpoints[:,:2]
cond0 = initial_conditions(spacein, w0)
condx = cond0[:,1].reshape(n_space, n_space)
condx = condx[:,0]

pinn = PINN(dim_hidden, w0, n_hidden, multux, multuy, n_space, n_time, scaley, device).to(device)
loss_fn = Loss(
        points,
        n_space,
        n_time,
        h/3,
        w0,
        steps,
        adim,
        par,
        scaley,
        device,
        interpVbeam,
        interpTbeam,
        t_tild
    )

_, V, T, _, _, _, _, _ = loss_fn.res_loss(pinn, True)

V0 = V[0].item()
T0 = 0

loss_fn.V0 = V0
loss_fn.T0 = T0

dir_model = pass_folder('model')
dir_logs = pass_folder('model/logs')
if load:
    filename = 'load/0.0001_5000_(1, 80).pth'
    dir_load = os.path.dirname(filename)
    pinn.load_state_dict(torch.load(filename, map_location=device))
if train:
    pinn_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
        max_epochs=epochs, path_logs=dir_logs, modeldir=dir_model)
    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)
    torch.save(pinn_trained.state_dict(), model_path)
else:
    pinn_trained = pinn

print(pinn_trained)

pinn_trained.eval()

tin = inpoints[:,-1].unsqueeze(1)
z = pinn_trained(spacein, tin)
v = calculate_speed(z, tin, par)
z = torch.cat([z, v], dim=1)

plot_initial_conditions(z, cond0, spacein, dir_model)

allpoints = torch.cat(points["all_points_eval"], dim=1)
space = allpoints[:,:2]
t = allpoints[:,-1].unsqueeze(1)
nsamples = (n_space, n_space) + (n_time,)
sol, V, T = obtainsolt_u(pinn_trained, space, t, nsamples, 1, par, steps, device)

sol1D = sol[sol.shape[1]//2,sol.shape[1]//2,:,1]
nfft = sol1D.shape[0]
beamdispl = interpdisplbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van = interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van *= np.max(V)/np.max(Van)
Tan = interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Tan *= np.max(T)/np.max(Tan)

dt = steps[2].item()
errV = (calculateRMS(V, dt, tmax) - calculateRMS(Van, dt, tmax))/(
        calculateRMS(Van, dt, tmax)
).item()
errT = (calculateRMS(T, dt, tmax) - calculateRMS(Tan, dt, tmax))/(
        calculateRMS(Tan, dt, tmax)
).item()

sol = sol.reshape(n_space**2, n_time, 2)
plot_sol(sol, spacein, t, dir_model)
plot_average_displ(sol, t, dir_model)

# Compute gradients with respect to the loss
loss = loss_fn(pinn_trained)[0]
loss.backward()

param_gradients = [(name, param.grad.abs().mean().item()) for name, param in pinn.named_parameters()]
# Sort by gradient magnitude to find the most determinant parameters
param_gradients.sort(key=lambda x: x[1], reverse=True)
most_determinant_params = [param_gradients[0][0], param_gradients[1][0]]

# Define perturbations for the two most determinant parameters and create a grid
perturb_range = np.linspace(-0.5, 0.5, 30)
loss_profile = np.zeros((50, 50))

# Set up grid and evaluate loss for perturbations
original_params = {name: param.clone() for name, param in pinn.named_parameters()}

for i, alpha in enumerate(perturb_range):
    for j, beta in enumerate(perturb_range):
        # Apply perturbations to the two most determinant parameters
        for idx, perturb in enumerate([alpha, beta]):
            layer_name, param_type = most_determinant_params[idx].rsplit(".", 1)
            param = getattr(getattr(pinn_trained, layer_name), param_type)
            param.data = original_params[most_determinant_params[idx]] + perturb
        
        # Forward pass and loss calculation
        loss_profile[i, j] = loss_fn(pinn_trained)[0].item()

# Reset parameters to original values
for name, param in original_params.items():
    layer_name, param_type = name.rsplit(".", 1)
    original_param = getattr(getattr(pinn_trained, layer_name), param_type)
    original_param.data = param

# Plot the loss profile
plt.figure(figsize=(10, 6))
plt.contourf(perturb_range, perturb_range, loss_profile, levels=50, cmap='viridis')
plt.colorbar(label='Loss')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.tight_layout()
plt.savefig(f'{dir_model}/loss_map.png')

import os
import shutil

def create_zip(file_paths, zip_name):
    shutil.make_archive(zip_name, 'zip', file_paths)

timenow = get_current_time(fmt='%m-%d %H:%M')

create_zip(dir_model, f'model_FF-{timenow}')
create_zip(dir_logs, f'logs_FF-{timenow}')