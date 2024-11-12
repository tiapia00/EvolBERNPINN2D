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
from scipy.interpolate import RegularGridInterpolator
import matplotlib.animation as animation
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
plotloss = False
getzip = False
plot_comp = False
plot_mid = True

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

interpdispbeam = RegularGridInterpolator((my_beam.xi, t_beam), w)
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
    filename = 'load/0.0001_1000_(1, 40).pth'
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
sol = sol.reshape(n_space * n_space, n_time, 2)

points_interp = np.array(np.meshgrid(x_domain.detach().cpu().numpy() * Lx, t_domain.detach().cpu().numpy() * t_tild)).T.reshape(-1,2)
labelled = interpdispbeam(points_interp)
labelled = labelled.reshape(n_space, n_time)
labelled = np.expand_dims(labelled, axis=1)
labelled = np.repeat(labelled, repeats=n_space, axis=1)
Van = interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van *= np.max(V)/np.max(Van)
Tan = interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Tan *= np.max(T)/np.max(Tan)

"""
fig, ax = plt.subplots()
line, = ax.plot(x_domain, labelled[:,0,0])
ax.legend()

def update(frame):
    line.set_ydata(labelled[:,0,frame])
    ax.set_title(f'$\\hat{{t}} = {frame * steps[2].item()}$')
    return line, 

ani = animation.FuncAnimation(fig=fig, func=update, frames=labelled.shape[2], interval=100)
"""
if plot_comp:
    space_in = spacein.detach().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    n1 = 10
    ax1.plot(x_domain, labelled[:,0,n1], label='Analytical', color='red')
    ax1.scatter(space_in[:,0] + sol[:,n1,0], space_in[:,1] + sol[:,n1,1], label='NN')
    ax1.set_xlabel(r'$\hat{x}$')
    ax1.set_title(f'$\\hat{{t}} = {n1 * steps[2].item():.2f}$')

    n2 = 55
    ax2.plot(x_domain, labelled[:,0,n2], label='Analytical', color='red')
    ax2.scatter(space_in[:,0] + sol[:,n2,0], space_in[:,1] + sol[:,n2,1], label='NN')
    ax2.set_xlabel(r'$\hat{x}$')
    ax2.set_title(f'$\\hat{{t}} = {n2 * steps[2].item():.2f}$')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{dir_model}/disp_comp.png')

def getFRF(omega, n):
    mode_sum = 0
    for i in range(n):
        mode_sum += (np.sin(np.pi * (i+1) * xk/L))**2/(-omega**2*m*L/2 + (((i+1)*np.pi)/L)**4*E*J*L/2)
    return mode_sum

if plot_mid:
    sol = sol.reshape(n_space, n_space, n_time, 2)
    t_end = n_time // 2
    solmid = np.mean(sol, axis=1)
    solmid = solmid[n_space//2, :t_end, 1]
    t_plot = torch.unique(t).detach().cpu().numpy()[:t_end]
    Fk0 = -1e-4
    n = 20 
    Omega = np.pi * 1/t_tild
    xk = Lx/2 
    L = Lx
    E = E
    J = my_beam.J 
    m = rho * my_beam.A
    t_lin = np.linspace(0, 1, 1000)
    mode_sum = getFRF(Omega, n)
    w = Fk0 * np.sin(Omega * t_lin) * mode_sum
    omegaFRF = np.linspace(0, 400, 1000)
    G = np.abs(getFRF(omegaFRF, n))
    plt.figure()
    plt.plot(omegaFRF, G)
    plt.xlabel(r'$\Omega$')
    plt.ylabel(r'$G$')
    plt.savefig(f'{dir_model}/FRF.png')
    w_interp = make_interp_spline(x=t_lin, y=w, k=5)
    w_eval = w_interp(t_plot * t_tild)
    plt.figure()
    plt.plot(t_plot, w_eval, label='Analytical')
    plt.plot(t_plot, np.max(np.abs(w_eval))/np.max(np.abs(solmid)) * solmid, label='NN')
    plt.xlabel(r'$\hat{t}$')
    plt.ylabel(r'$w_\text{mid}$')
    plt.legend()
    plt.savefig(f'{dir_model}/mid_comp.png')


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

    perturb_range = np.linspace(-1e-1, 1e-1, 30)
    loss_profile = np.zeros((30, 30))

    original_params = {name: param.clone() for name, param in pinn_trained.named_parameters()}

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