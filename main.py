from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder, delete_old_files
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free, obtain_max_stress

torch.set_default_dtype(torch.float32)

def calculate_speed(output: torch.Tensor, t: torch.Tensor):
    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=device),
            create_graph=False, retain_graph=True)[0].detach()
    vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=device),
            create_graph=False, retain_graph=True)[0].detach()
    
    v = torch.cat([vx, vy], dim=1)

    return v

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
my_beam = Beam(Lx, E, rho, h, 1, n_space_beam)

t_tild, w, en0 = obtain_analytical_free(par, my_beam, w0, t, n_time)
# en0 [J]
sig_max = obtain_max_stress(my_beam, w)
#sig_max [Pa]
lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, dim_hidden, n_hidden, lr, epochs = get_params(par.pinn_par)

x_domain = torch.linspace(0, Lx, n_space[0])/Lx
y_domain = torch.linspace(0, Ly, n_space[1])/Lx
t_domain = torch.linspace(0, T, n_time)/t_tild

adim = ((t_tild**2/(rho*w0)*sig_max/Lx).item(), (lam+2+mu)/Lx*w0, lam/Lx*w0, mu/Lx*w0, w0)
adim_NN = (w0, sig_max)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.generate_grid_bound(),
    'all_points': grid.get_all_points()
}

prop = {'E': E, 'J': my_beam.J, 'm': rho * my_beam.A}
pinn = PINN(dim_hidden, n_hidden, adim_NN).to(device)

in_penalty = np.array([1, 2])
loss_fn = Loss(
        points,
        n_space,
        n_time,
        w0,
        steps,
        in_penalty,
        adim
    )


if retrain_PINN:
    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    pinn_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, n_hidden).to(device)
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
t_in = t.to(device)
space_in = torch.cat([x, y], dim=1)
z = pinn_trained(space_in, t_in)[:,:2]
v = calculate_speed(z, t_in)
z = torch.cat([z, v], dim=1)

cond0 = initial_conditions(points['initial_points'], w0)

plot_initial_conditions(z, cond0, x, y, dir_model)
x, y, t = grid.get_all_points()
nsamples = n_space + (n_time,)
sol = obtainsolt(pinn_trained, space_in, t, nsamples, device)
plot_sol(sol, space_in, t, dir_model)

plot_compliance(pinn_trained, x, y, t, w_ad, dir_model, device)
#plot_sol_comparison(pinn_trained, x, y, t, w_ad, n_space,
#                    n_time, n_space_beam, dir_model, device)

t, en, en_p, en_k = calc_energy(pinn_trained, points, n_space, n_time, steps[0], steps[1])
plot_energy(t, en_k, en_p, en, En0, dir_model)

