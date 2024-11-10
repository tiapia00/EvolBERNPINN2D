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
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator

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

load = False
train = True
plotloss = False
getzip = True
plots = False
plot_comp = True

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

t_beam, t_tild, w, V_an, v, Ek_an = obtain_analytical_free(my_beam, w0, t, 3000, 2)
"""
plt.figure()
plt.plot(t_beam, V_an, label='Potential')
plt.plot(t_beam, Ek_an, label='Kinetic')
plt.plot(t_beam, V_an + Ek_an, label='Tot')
plt.legend()
plt.show()
"""

interpdispbeam = RegularGridInterpolator((my_beam.xi, t_beam), w)
interpvbeam = RegularGridInterpolator((my_beam.xi, t_beam), v)
interpVbeam = make_interp_spline(t_beam, V_an, k=5)
interpTbeam = make_interp_spline(t_beam, Ek_an, k=5)

lam, mu = par.to_matpar_PINN()

Lx, Ly, tmax, n_space, n_time, w0, dim_hidden, n_hidden, multux, multuy, multhyperx, modesx, modesy, lr, epochs = get_params(par.pinn_par)
L_tild = Lx
x_domain = torch.linspace(0, Lx, n_space)/Lx
y_domain = torch.linspace(0, Ly, n_space)/Lx
t_domain = torch.linspace(0, tmax, n_time)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, multhyperx, y_domain, t_domain, device)
scaley = 2 
scale_interp = 1 
x_interp = torch.linspace(0, Lx, n_space // scale_interp)/Lx
y_interp = torch.linspace(0, Ly, n_space // scale_interp)/Lx
t_interp = torch.linspace(0, tmax, n_time // scale_interp)

x_grid, y_grid, t_grid = torch.meshgrid(x_interp[1:-1], y_interp[1:-1], t_interp[1:], indexing='ij')
x_grid = x_grid.reshape(-1,1).to(device)
y_grid = y_grid.reshape(-1,1).to(device)
t_grid = t_grid.reshape(-1,1).to(device)
interp_points = (x_grid, y_grid, t_grid)

points = {
    'res_points': grid.get_interior_points_train(scaley),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.generate_grid_bound(),
    'initial_points_hyper': grid.get_initial_points_hyper(),
    'all_points': grid.get_all_points(),
    'interp_points': interp_points
}

adim = (mu/lam, (lam+mu)/lam, rho/(lam*t_tild.item()**2)*Lx**2)
par = {"Lx": Lx,
        "w0": w0,
        "b": h/3,
        "lam": lam,
        "mu":mu,
        "rho": rho,
        "t_ast": t_tild}

inpoints = torch.cat(points["initial_points_hyper"], dim=1)
spacein = inpoints[:,:2]
cond0 = initial_conditions(spacein, w0)
condx = cond0[:,1].reshape(n_space * multhyperx, n_space // scaley)
condx = condx[:,0]

x_res = x_interp[1:-1].detach().cpu().numpy()
t_res = t_interp.detach().cpu().numpy()
points_interp = np.array(np.meshgrid(x_res * Lx, t_res * t_tild)).T.reshape(-1,2)
labelled = interpdispbeam(points_interp)
labelled = labelled.reshape(n_space // scale_interp - 2, n_time // scale_interp)
labelled = np.expand_dims(labelled, axis=1)
labelled_no_noise = np.repeat(labelled, repeats=labelled.shape[0], axis=1)
sigma = 0.01
noise = np.random.normal(0, sigma, labelled.shape)
labelled_noise = labelled_no_noise + noise
labelled = torch.tensor(labelled_noise, device=device, dtype=torch.float32)

x_res = x_interp.detach().cpu().numpy()
t_res = t_interp.detach().cpu().numpy()
points_interp = np.array(np.meshgrid(x_res * Lx, t_res * t_tild)).T.reshape(-1,2)
labelled_speed = interpvbeam(points_interp)
labelled_speed = labelled_speed.reshape(n_space // scale_interp, n_time // scale_interp)
labelled_speed = np.expand_dims(labelled_speed, axis=1)
noise_speed = np.random.normal(0, sigma, labelled_speed.shape)
labelled_speed_noise = labelled_speed + noise_speed

if plots:
    plt.figure()
    plt.plot(x_res, labelled_noise[:,0,0])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$w(x, t_0)$')
    plt.savefig('displ_noise.png')
labelled = labelled[...,1:]


pinn = PINN(dim_hidden, w0, n_hidden, n_space, scaley, n_time, multux, multuy, modesx, modesy, multhyperx, scale_interp, device).to(device)
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
        multhyperx,
        device,
        interpVbeam,
        interpTbeam,
        t_tild,
        labelled,
        scale_interp
    )

_, V, T, _, _, _, _, _ = loss_fn.res_loss(pinn, True)

V0 = V[0].item()
T0 = 0

loss_fn.V0 = V0
loss_fn.T0 = T0

dir_model = pass_folder('model')
dir_logs = pass_folder('model/logs')
if load:
    filename = 'model/11-10/1508/0.001_3000_(1, 50).pth'
    dir_load = os.path.dirname(filename)
    state_dict = torch.load(filename, map_location=device)
    if 'in_penalties' in state_dict:
        del state_dict['in_penalties']
    if 'data_penalties' in state_dict:
        del state_dict['data_penalties']
    pinn.load_state_dict(state_dict, strict=False)
if train:
    pinn_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
        max_epochs=epochs, path_logs=dir_logs)
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
in_points_no_hyp = torch.cat(points['initial_points'], dim=1)
spacein_no_hyp = in_points_no_hyp[:,:2]

plot_initial_conditions(z, cond0, labelled_speed_noise, spacein, spacein_no_hyp, multhyperx * n_space, dir_model, justplotdisp=True)

allpoints = torch.cat(points["all_points"], dim=1)
space = allpoints[:,:2]
t = allpoints[:,-1].unsqueeze(1)
nsamples = (n_space, n_space) + (n_time,)
sol, V, T, v = obtainsolt_u(pinn_trained, space, t, nsamples, 1, par, steps, device)

Van = interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van *= np.max(V)/np.max(Van)
Tan = interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Tan *= np.max(T)/np.max(Tan)

plt.figure()
plt.plot(torch.unique(t).detach().cpu().numpy(), V, label=r'$\hat{V}$')
plt.plot(torch.unique(t).detach().cpu().numpy(), Van, label=r'$V$')
plt.xlabel(r'$\hat{t}$')
plt.legend()
plt.savefig(f'{dir_model}/anhatencomp.png')

sol1D = sol[sol.shape[1]//2,sol.shape[1]//2,:,1]
nfft = sol1D.shape[0]

dt = steps[2].item()
errV = (calculateRMS(V, dt, tmax) - calculateRMS(Van, dt, tmax))/(
        calculateRMS(Van, dt, tmax)
).item()
errT = (calculateRMS(T, dt, tmax) - calculateRMS(Tan, dt, tmax))/(
        calculateRMS(Tan, dt, tmax)
).item()

sol = sol.reshape(n_space**2, n_time, 2)
inpoints = torch.cat(points['initial_points'], dim=1)
spacein = inpoints[:,:2]
plot_sol(sol, spacein, t, dir_model)
plot_average_displ(sol, t, dir_model)

labelled = labelled_no_noise
"""
fig, ax = plt.subplots()
line, = ax.plot(x_domain[1:-1].detach().cpu().numpy(), labelled[:,0,0])
ax.legend()

def update(frame):
    line.set_ydata(labelled[:,0,frame])
    ax.set_title(f'$\\hat{{t}} = {frame * steps[2].item():.2f}$')
    return line, 

ani = animation.FuncAnimation(fig=fig, func=update, frames=labelled.shape[2], interval=100)
"""
if plot_comp:
    space_in = spacein.detach().cpu().numpy()
    plt.figure()
    n1 = 5 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_res, labelled_speed[:,0,n1], label='Analytical', color='red')
    ax1.plot(x_res, labelled_speed_noise[:,0,n1], label='Analytical + Noise')
    ax1.plot(x_domain, v[:,0,n1,1], label='NN')
    ax1.set_xlabel(r'$\hat{x}$')
    ax1.set_ylabel(r'$v_y$')
    ax1.set_title(f'$\\hat{{t}} = {n1 * steps[2].item():.2f}$')

    n2 = 35
    ax2.plot(x_res, labelled_speed[:,0,n2], label='Analytical', color='red')
    ax2.plot(x_res, labelled_speed_noise[:,0,n2], label='Analytical + Noise')
    ax2.plot(x_domain, v[:,0,n2,1], label='NN')
    ax2.set_xlabel(r'$\hat{x}$')
    ax2.set_ylabel(r'$v_y$')
    ax2.set_title(f'$\\hat{{t}} = {n2 * steps[2].item():.2f}$')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{dir_model}/disp_comp.png')

if plotloss:
    grad_accumulation = {name: 0.0 for name, param in pinn_trained.named_parameters()}

    num_epochs = 30

    pbar = tqdm(total=num_epochs, desc="", position=0)
    for epoch in range(num_epochs):
        loss = loss_fn(pinn_trained)[0]        # Compute loss
        pinn_trained.zero_grad()                   # Zero gradients before backprop
        loss.backward()                     # Backpropagation

        pbar.update(1)

        # Accumulate gradient magnitudes
        for name, param in pinn_trained.named_parameters():
            if param.grad is not None:
                grad_accumulation[name] += param.grad.abs().mean().item()

    pbar.close()
    for name in grad_accumulation:
        grad_accumulation[name] /= num_epochs

    sorted_params = sorted(grad_accumulation.items(), key=lambda x: x[1], reverse=True)
    most_influential_params = [sorted_params[0][0], sorted_params[1][0]]

    perturb_range = np.linspace(-1e-3, 1e-3, 30)
    loss_profile = np.zeros((30, 30))

    original_params = {name: param.clone() for name, param in pinn.named_parameters()}

    for i, alpha in enumerate(perturb_range):
        for j, beta in enumerate(perturb_range):
            for idx, perturb in enumerate([alpha, beta]):
                layer_name, param_type = most_influential_params[idx].rsplit(".", 1)
                param = getattr(getattr(pinn_trained, layer_name), param_type)
                param.data = original_params[most_influential_params[idx]] + perturb
            
            loss_profile[i, j] = loss_fn(pinn_trained)[0].item()

    plt.figure(figsize=(10, 6))
    X, Y = np.meshgrid(perturb_range, perturb_range)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, loss_profile, cmap='viridis', edgecolor='none')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
    plt.locator_params(axis='y', nbins=7)
    plt.locator_params(axis='x', nbins=7)
    plt.locator_params(axis='z', nbins=7)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.tight_layout()
    plt.savefig(f'{dir_model}/loss_map.png')

if getzip:
    import os
    import shutil

    def create_zip(file_paths, zip_name):
        shutil.make_archive(zip_name, 'zip', file_paths)

    timenow = get_current_time(fmt='%m-%d %H:%M')

    create_zip(dir_model, f'model_FF-{timenow}')
    create_zip(dir_logs, f'logs_FF-{timenow}')