from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder, delete_old_files
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free
from eigenNN import MLNet, denormalizematr

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

dir_model = pass_folder('model')
dir_logs = pass_folder('model/logs')

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

Lx, t, h, nx, n_time, w0 = get_params(par.beam_par)
E, rho, _ = get_params(par.mat_par)
my_beam = Beam(Lx, E, rho, h, 4e-3, nx)

w, en0 = obtain_analytical_free(my_beam, w0, t, n_time)

#t_points, sol = obtain_analytical_forced(par, my_beam, load_dist, t_tild, n)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, dim_hidden, n_hid_space, dim_mult, lr, epochs = get_params(par.pinn_par)

x_domain = torch.linspace(0, Lx, n_space[0])
y_domain = torch.linspace(0, Ly, n_space[1])
t_domain = torch.linspace(0, T, n_time)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.grid_bound,
    'all_points': grid.get_all_points()
}

prop = {'E': E, 'J': my_beam.J, 'm': rho * my_beam.A, 'A': my_beam.A}
m_par = (lam, mu, rho)
nsamples = n_space + (n_time,)

calculate = Calculate(
        initial_conditions,
        m_par,
        points,
        nsamples,
        steps,
        w0,
        device
    )

eigenNN = MLNet(4, 60, 9)
eigenNN.load_state_dict(torch.load('data//eigenestmodel.pth'))
input_eigen = torch.tensor([E, rho, Lx, Ly]).reshape(1,-1)
eigen = eigenNN(input_eigen)
ef_range = torch.load('data//ef_range.pt')
eigen = denormalizematr(eigen, ef_range)

omega_trans = eigen.squeeze(0)[:1]
omega_ax = eigen.squeeze(0)[:1]

nninbcs = NNinbc(20, 3).to(device)
nninbcs_trained = train_inbcs(nninbcs, calculate, 1000, 1e-3)

x, y, t = points['initial_points']
x_in = x.to(device)
y_in = y.to(device)
t_in = t.to(device)
in_points = torch.cat([x_in, y_in, t_in], dim=1)

#nndist = NNd(20, 3).to(device)
#nndist_trained = train_dist(nndist, calculate, 5000, 1e-3)
#output = nndist_trained(in_points)
#plot_distance0(output, in_points[:,:2], dir_model)

pinn = PINN(n_hid_space, dim_mult, nninbcs_trained,
        omega_ax, omega_trans, prop).to(device)

Psi_0, K_0 = calculate.gete0(pinn)

if retrain_PINN:
    pinn_trained, indicators = train_model(pinn, calc=calculate, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained, indicators = PINN(dim_hidden, n_hid_space, dim_mult).to(device)
    filename = get_last_modified_file('model', '.pth')

    dir_model = os.path.dirname(filename)
    print(f'Target for outputs: {dir_model}\n')

    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

print(pinn_trained)

pinn_trained.eval()

space = torch.cat([x, y], dim=1)
z = pinn_trained(space, t)
v = calculate_speed(pinn_trained, (x, y, t), device)
z = torch.cat([z, v], dim=1)

cond0 = initial_conditions(x_in, w0)

plot_initial_conditions(z, cond0, x_in, y_in, n_space, dir_model)

x, y, _ = grid.get_initial_points()
_, _, t = grid.get_all_points()
x = x.to(device)
y = y.to(device)
t = t.to(device)
space_in = torch.cat([x, y], dim=1)
sol = obtainsolt(pinn_trained, space_in, t, nsamples, device)
plot_sol(sol, space_in, t, dir_model)

plot_sol_comparison(sol, space_in, t, w, dir_model)
plot_indicators(indicators, t, dir_model)