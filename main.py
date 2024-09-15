from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder, delete_old_files
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free, obtain_max_stress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
#en0 [?]
sig_max = obtain_max_stress(my_beam, w)
#sig_max [MPa]
lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, dim_hidden, n_hidden, lr, epochs = get_params(par.pinn_par)

x_domain = torch.linspace(0, Lx, n_space[0])/Lx
y_domain = torch.linspace(0, Ly, n_space[1])/Lx
t_domain = torch.linspace(0, T, n_time)/t_tild

adim = (((t_tild**2/(rho*w0)*sig_max/Lx)**(-1)).item(), (sig_max*Lx/(w0*lam)).item(), mu/lam, w0, sig_max)
adim_NN = (w0, sig_max)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.generate_grid_bound(),
    'all_points': grid.get_all_points()
}

nn_inbcs = NN(100, 4).to(device)

x = points['all_points'][0].detach().cpu().numpy()
y = points['all_points'][1].detach().cpu().numpy()
t = points['all_points'][2].detach()

t0idx = torch.nonzero(t.squeeze() == 0).squeeze()
in_penalty = np.array([1, 2])

loss_fn = Loss(
        points,
        n_space,
        n_time,
        w0,
        steps,
        in_penalty,
        adim,
        t0idx,
        device
    )

distances = get_D(points['all_points'], 1, torch.max(y_domain)).to(device)
## Plot distances for the domain ##
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(x, y, t.detach().cpu().numpy(), c=distances.detach().cpu().numpy())
cb = plt.colorbar(sc)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
#plt.show()

nn_inbcs = train_inbcs(nn_inbcs, loss_fn, 50000, 1e-3)

x, y, t_in = points['initial_points']
x = x.to(device)
y = y.to(device)
t_in = t_in.to(device)
space_in = torch.cat([x, y], dim=1)

output = nn_inbcs(space_in, t_in)
v = calculate_speed(output[:,:2], t_in).detach().cpu().numpy()
output = output.detach().cpu().numpy()
fig, axs = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [2, 1]})
ax_big = plt.subplot2grid((2, 2), (0, 0), colspan=2)

ax_big.plot(x.squeeze().detach().cpu().numpy() + output[:,0], y.squeeze().detach().cpu().numpy() + output[:,1])
ax_big.set_title('Displacement')
ax_big.set_xlabel(r'$\hat{x}$')
ax_big.set_ylabel(r'$\hat{y}$')

scattervx = axs[1, 0].scatter(x.squeeze().detach().cpu().numpy() + output[:,0], y.squeeze().detach().cpu().numpy() + output[:,1],
        c=v[:,0], cmap='viridis')
axs[1, 0].set_xlabel(r'$\hat{x}$')
axs[1, 0].set_ylabel(r'$\hat{y}$')
cbar1 = fig.colorbar(scattervx, ax=axs[1,0])
cbar1.set_label(r'$v_x$')
axs[1, 1].scatter(x.squeeze().detach().cpu().numpy() + output[:,0], y.squeeze().detach().cpu().numpy() + output[:,1],
        c=v[:,0], cmap='viridis')
axs[1, 1].set_xlabel(r'$\hat{x}$')
axs[1, 1].set_ylabel(r'$\hat{y}$')
cbar2 = fig.colorbar(scattervx, ax=axs[1,1])
cbar2.set_label(r'$v_y$')
#plt.show()

pinn = PINN(dim_hidden, n_hidden, adim_NN, distances).to(device)

if retrain_PINN:
    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    pinn_trained = train_model(pinn, nn_inbcs, loss_fn=loss_fn, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, n_hidden, adim_NN,  nn_inbcs, distances).to(device)
    filename = get_last_modified_file('model', '.pth')

    dir_model = os.path.dirname(filename)
    print(f'Target for outputs: {dir_model}\n')

    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

print(pinn_trained)

pinn_trained.eval()

allpoints = torch.cat(points['all_points'], dim=1)
space = allpoints[:,:2]
t = allpoints[:,-1].unsqueeze(1)
out = getout(pinn_trained, nn_inbcs, space, t)
outin = out[t0idx,:2]
v = calculate_speed(out[:,:2], t)[t0idx]
z = torch.cat([outin, v], dim=1)

cond0 = initial_conditions(space_in, w0)

plot_initial_conditions(z, cond0, space_in[:,0], space_in[:,1], dir_model)
plot_init_stresses(z, space[t0idx], t[t0idx], dir_model)

x, y, t = grid.get_all_points()
nsamples = n_space + (n_time,)
sol, space_in = obtainsolt_u(pinn_trained, nn_inbcs, space, t, nsamples, device)
"""
plt.figure()
plt.plot(space_in[:,0].detach().cpu().numpy() + sol[:,0,0], space_in[:,1].detach().cpu().numpy() + sol[:,0,1])
plt.show()
"""
plot_sol(sol, space_in, t, dir_model)

#plot_sol_comparison(pinn_trained, x, y, t, w_ad, n_space,
#                    n_time, n_space_beam, dir_model, device)