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

retrain_PINN = True
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

Lx, t, h, n, w0 = get_params(par.beam_par)
E, rho, _ = get_params(par.mat_par)
my_beam = Beam(Lx, E, rho, h, 4e-3, n)

t_tild, w_ad, en0 = obtain_analytical_free(par, my_beam, w0, t, n)

load_dist = (np.sin, np.sin)
#t_points, sol = obtain_analytical_forced(par, my_beam, load_dist, t_tild, n)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_train, w0, dim_hidden, n_hid_space, lr, epochs = get_params(par.pinn_par)

L_tild = Lx
x_domain = torch.linspace(0, Lx, n_train)/Lx
y_domain = torch.linspace(-Ly/2, Ly/2, n_train)/Ly
t_domain = torch.linspace(0, T, n_train)/t_tild

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.get_boundary_points(),
    'all_points': grid.get_all_points()
}


prop = {'E': E, 'J': my_beam.J, 'm': rho * my_beam.A}
pinn = PINN(dim_hidden, n_hid_space, points, w0, prop, initial_conditions, device).to(device)

En0 = calc_initial_energy(pinn, n_train, points, device)
print(En0)

loss_fn = Loss(
        return_adim(L_tild, t_tild, rho, mu, lam),
        initial_conditions,
        points,
        n_train,
        w0,
        En0,
        steps
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
    pinn_trained = PINN(dim_hidden, n_hid_space, points, w0, initial_conditions, device).to(device)
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
v = calculate_speed(pinn_trained, (x, y, t))
z = torch.cat([z, v], dim=1)

cond0 = initial_conditions(points['initial_points'], w0)

plot_initial_conditions(z, cond0, x, y, n_train, dir_model)

x, y, t = grid.get_all_points()

plot_sol(pinn_trained, x, y, t, n_train, dir_model, device)

plot_compliance(pinn_trained, x, y, t, n_train, w_ad, dir_model, device)
plot_sol_comparison(pinn_trained, x, y, t, w_ad, n_train,
                    dir_model, device)

t, en, en_p, en_k = calc_energy(pinn_trained, points, n_train, device)
plot_energy(t, en_k, en_p, en, En0, dir_model)
