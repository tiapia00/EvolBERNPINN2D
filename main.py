from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder, delete_old_files, get_current_time
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free

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

retrain_PINN =  True
delete_old = False

if delete_old:
    delete_old_files("model")
    delete_old_files("in_model")

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

t_tild, w_ad, V0 = obtain_analytical_free(my_beam, w0, t, n_time, 2)
print(t_tild)
print(V0)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, dim_hidden, n_hidden, lr, epochs = get_params(par.pinn_par)

L_tild = Lx
x_domain = torch.linspace(0, Lx, n_space)/Lx
y_domain = torch.linspace(0, Ly, n_space)/Lx
t_domain = torch.linspace(0, T, n_time)

omegas = my_beam.omega * t_tild
gammas = my_beam.gamma * Lx

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.get_boundary_points(),
    'all_points': grid.get_all_points()
}

adim = (mu/lam, (lam+mu)/lam, rho/(lam*t_tild.item()**2)*Lx**2)
par = {"Lx": Lx,
        "w0": w0,
        "lam": lam,
        "mu":mu,
        "rho": rho,
        "t_ast": t_tild}
pinn = PINN(dim_hidden, w0, n_hidden).to(device)

#En0 = calc_initial_energy(pinn, n_space, points, device)

adap_in = np.array([1., 2.])
loss_fn = Loss(
        points,
        n_space,
        n_time,
        h/3,
        w0,
        steps,
        adim,
        par,
        10,
        adap_in,
        device
    )


if retrain_PINN:
    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    pinn_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs, modeldir=dir_model)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, w0, n_hidden).to(device)
    filename = get_last_modified_file('model', '.pth')

    dir_model = os.path.dirname(filename)
    print(f'Target for outputs: {dir_model}\n')

    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

print(pinn_trained)

pinn_trained.eval()

inpoints = torch.cat(points["initial_points"], dim=1)
spacein = inpoints[:,:2]
tin = inpoints[:,-1].unsqueeze(1)
z = pinn_trained(spacein, tin)
v = calculate_speed(z, tin, par)
z = torch.cat([z, v], dim=1)

cond0 = initial_conditions(spacein, w0)

plot_initial_conditions(z, cond0, spacein, dir_model)

allpoints = torch.cat(points["all_points"], dim=1)
space = allpoints[:,:2]
t = allpoints[:,-1].unsqueeze(1)
nsamples = (n_space, n_space) + (n_time,)
sol = obtainsolt_u(pinn_trained, space, t, nsamples)
plot_sol(par['w0']*sol, spacein, t, dir_model)
plot_average_displ(par['w0']*sol, t, dir_model)

import shutil

def create_zip(file_paths, zip_name):
    shutil.make_archive(zip_name, 'zip', file_paths)

timenow = get_current_time(fmt='%m-%d %H:%M')

create_zip(dir_model, f'model_W-{timenow}')
create_zip(dir_logs, f'logs_W-{timenow}')