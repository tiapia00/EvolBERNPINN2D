from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder, delete_old_files
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free, obtain_analytical_forced

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

retrain_PINN = False
delete_old = False

if delete_old:
    delete_old_files("model")
    delete_old_files("in_model")

par = Parameters()

Lx, t, n, w0 = get_params(par.beam_par)
E, rho, _, h = get_params(par.mat_par)
my_beam = Beam(Lx, E, rho, h/1000, 40e-3, n)

t_tild, w_ad, en0 = obtain_analytical_free(par, my_beam, w0, t, n)

load_dist = np.sin
sol = obtain_analytical_forced(par, my_beam, load_dist, t, n)

E, rho, _, nu = get_params(par.mat_par)
lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_train, w0, dim_hidden, lr, epochs = get_params(par.pinn_par)

L_tild = Lx
x_domain = torch.linspace(0, Lx, n_train)/L_tild
y_domain = torch.linspace(-Ly/2, Ly/2, n_train)/Ly
t_domain = torch.linspace(0, T, n_train)/t_tild

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.get_boundary_points()
}

pinn = PINN(dim_hidden, points, w0, initial_conditions).to(device)

loss_fn = Loss(
        return_adim(L_tild, t_tild, rho, mu, lam),
        initial_conditions,
        points,
        w0
    )
if retrain_PINN:
    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    pinn_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs, points=points, n_train=n_train)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, points, w0, initial_conditions).to(device)
    filename = get_last_modified_file('model', '.pth')

    dir_model = os.path.dirname(filename)
    print(f'Target for outputs: {dir_model}\n')

    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

print(pinn_trained)

pinn_trained.eval()

x, y, t = points['initial_points']
x = x.to(device)
y = y.to(device)
t = t.to(device)
z = f(pinn_trained, x, y, t)
cond0 = initial_conditions(points['initial_points'], w0)

plot_initial_conditions(z, cond0, x, y, n_train, dir_model)

x, y, t = grid.get_all_points()

plot_sol(pinn_trained, x, y, t, n_train, dir_model, device)

plot_compliance(pinn_trained, x, y, t, n_train, w_ad, dir_model, device)
plot_sol_comparison(pinn_trained, x, y, t, w_ad, n_train,
                    dir_model, device)

t, en_k, en_p, en = calc_energy(pinn_trained, loss_fn, n_train, device)
plot_energy(t, en_k, en_p, en, en0, dir_model)
