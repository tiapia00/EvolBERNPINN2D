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
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

torch.set_default_dtype(torch.float32)

def read_multi_section_table(file_path):
    # Read the entire file into a list of lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    current_header = None
    sections = {}  # Dictionary to store data by section
    
    # Process each line
    for line in lines:
        line = line.strip()  # Remove any leading/trailing whitespace
        # Check if the line is a section header (e.g., contains 'X' and a label)
        if "X" in line:
            # Extract only the section name (last part of the header)
            current_header = line.split()[-1]  # Take the last word as the section label, e.g., 'AVGA'
            sections[current_header] = []  # Initialize a new list for this section
        elif current_header:
            # Split line into columns and add to the current section list
            sections[current_header].append(line.split())
    
    # Create a DataFrame for each section
    dataframes = {}
    for header, rows in sections.items():
        # Define column names
        columns = ['X', header]  # Use 'X' and the section name as column headers
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert data to numeric types and drop rows with NaN values
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # Store in the dictionary of DataFrames
        dataframes[header] = df
    
    return dataframes

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
getzip = False 
plots = False
plot_comp = True
plot_mid = False
import_abq = False

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

t_beam, t_tild, w, V_an, Ek_an = obtain_analytical_free(my_beam, w0, t, 1000, 2)
if plots:
    plt.figure()
    plt.plot(t_beam, V_an, label='Potential Energy')
    plt.plot(t_beam, Ek_an, label='Kinetic Energy')
    plt.xlim((0, 0.05))
    plt.legend()

    plt.show()
    fig, ax = plt.subplots()
    line, = ax.plot(my_beam.xi, w[:,0])
    ax.legend()

    def update(frame):
        line.set_ydata(w[:,frame])
        return line, 

    ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=100)
    plt.show()

if import_abq:
    path_abq = 'load/ABQres.rpt'
    data_abq = read_multi_section_table(path_abq)
    keys = list(data_abq)
    interps = dict.fromkeys(keys, None)
    for i, data in enumerate(data_abq.items()):
        key = keys[i]
        data_abq[key] = data_abq[key].drop_duplicates(subset='X')
        interpi = make_interp_spline(data_abq[key]['X'] - 1, data_abq[key][key], k=5)
        interps[key] = interpi


interpdispbeam = RegularGridInterpolator((my_beam.xi, t_beam), w)
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

points = {
    'res_points': grid.get_interior_points_train(scaley),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.generate_grid_bound(),
    'initial_points_hyper': grid.get_initial_points_hyper(),
    'all_points': grid.get_all_points_eval(),
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

points_interp = np.array(np.meshgrid(x_domain.detach().cpu().numpy() * Lx, t_domain.detach().cpu().numpy() * t_tild)).T.reshape(-1,2)
labelled = interpdispbeam(points_interp)
labelled = labelled.reshape(n_space, n_time)
labelled = np.expand_dims(labelled, axis=1)
labelled = np.repeat(labelled, repeats=n_space, axis=1)
labelled = torch.tensor(labelled, device=device, dtype=torch.float32)

pinn = PINN(dim_hidden, w0, n_hidden, n_space, scaley, n_time, multux, multuy, modesx, modesy, multhyperx, device).to(device)
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
        labelled
    )

_, V, T, _, _, _, _, _ = loss_fn.res_loss(pinn, True)

V0 = V[0].item()
T0 = 0

loss_fn.V0 = V0
loss_fn.T0 = T0

dir_model = pass_folder('model')
dir_logs = pass_folder('model/logs')
if load:
    filename = 'load/1e-05_10000_(1, 60).pth'
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
torch.cuda.empty_cache()

tin = inpoints[:,-1].unsqueeze(1)
z = pinn_trained(spacein, tin)
v = calculate_speed(z, tin, par)
z = torch.cat([z, v], dim=1)

plot_initial_conditions(z, cond0, spacein, dir_model)

allpoints = torch.cat(points["all_points"], dim=1)
space = allpoints[:,:2]
t = allpoints[:,-1].unsqueeze(1)
nsamples = (n_space, n_space) + (n_time,)
sol, V, T, vmid, amid = obtainsolt_u(pinn_trained, space, t, nsamples, 1, par, steps, device)
Vmax = np.max(V)
Tmax = np.max(T)
vmidmax = np.max(vmid)
amidmax = np.max(amid)

maxscale = [Tmax, Vmax, vmidmax, amidmax]

Van = interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van *= Vmax/np.max(Van)
Tan = interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Tan *= Tmax/np.max(Tan)

plt.figure()
plt.plot(torch.unique(t).detach().cpu().numpy(), V, label=r'$\hat{V}$')
plt.plot(torch.unique(t).detach().cpu().numpy(), Van, label=r'$V$')
plt.xlabel(r'$\hat{t}$')
plt.legend()
plt.savefig(f'{dir_model}/anhatencomp.png')

sol1D = sol[sol.shape[0] // 2,sol.shape [1] //2,:,1]
nfft = sol1D.shape[0]
middispl = labelled[labelled.shape[0] // 2, labelled.shape[1] // 2, :]
plt.figure()
plt.plot(t_domain.detach().cpu().numpy(), middispl.detach().cpu().numpy(), label=r'$w$')
plt.plot(t_domain.detach().cpu().numpy(), sol1D, label=r'$\hat{w}$')
plt.xlabel(r'$\hat{t}$')
plt.legend()
plt.savefig(f'{dir_model}/wmidcomp.png')

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

def getFRF(omega, n, xk, xj):
    mode_sum = 0
    for i in range(n):
        mode_sum += (np.sin(np.pi * (i+1) * xk/L)*(np.sin(np.pi * (i+1) * xj/L)))/(-omega**2*m*L/2 + (((i+1)*np.pi)/L)**4*E*J*L/2)
    return mode_sum

if plot_comp:
    space_in = spacein.detach().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x_domain, labelled[:,0,23].detach().cpu().numpy(), label='Analytical', color='red')
    ax1.scatter(space_in[:,0] + sol[:,23,0], space_in[:,1] + sol[:,23,1], label='NN')
    ax1.set_xlabel(r'$\hat{x}$')
    ax1.set_title(r'$\hat{t} = 0.115$')

    ax2.plot(x_domain, labelled[:,0,43].detach().cpu().numpy(), label='Analytical', color='red')
    ax2.scatter(space_in[:,0] + sol[:,43,0], space_in[:,1] + sol[:,43,1], label='NN')
    ax2.set_xlabel(r'$\hat{x}$')
    ax2.set_title(r'$\hat{t} = 0.215$')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{dir_model}/disp_comp.png')

if plot_mid:
    sol = sol.reshape(n_space, n_space, n_time, 2)
    t_end = n_time
    solmid = np.mean(sol, axis=1)
    idx = n_space // 2
    solmid = solmid[idx, :, 1]
    t_plot = torch.unique(t).detach().cpu().numpy()
    Fk1 = -1e-4
    Fk2 = 5e-5
    n = 20 
    Omega_1 = np.pi * 1/t_tild
    Omega_2 = np.pi * 11/t_tild
    xk = Lx/2 
    xj = Lx/2
    L = Lx
    E = E
    J = my_beam.J 
    m = rho * my_beam.A
    t_lin = np.linspace(0, 1, 1000)
    mode_sum_1 = getFRF(Omega_1, n, xk, xj)
    mode_sum_2 = getFRF(Omega_2, n, xk, xj)
    w_1 = Fk1 * np.sin(Omega_1 * t_lin) * mode_sum_1
    #w_2 = Fk2 * np.sin(Omega_2 * t_lin) * mode_sum_2
    w = w_1
    omegaFRF = np.linspace(0, 400, 1000)
    w_interp = make_interp_spline(x=t_lin, y=w, k=5)
    w_eval = w_interp(t_plot * t_tild)
    plt.figure()
    plt.plot(t_plot, w_eval, label='Analytical')
    plt.plot(t_plot, np.max(np.abs(w_eval))/np.max(np.abs(solmid)) * solmid, label='NN')
    err = np.mean((np.max(np.abs(w_eval))/np.max(np.abs(solmid)) * solmid - w_eval)**2)
    print(err)
    plt.xlabel(r'$\hat{t}$')
    plt.ylabel(r'$w_\text{mid}$')
    plt.legend()
    plt.savefig(f'{dir_model}/mid_comp.png')
    xj = np.linspace(0, L, 1000)
    mode_sum_1 = getFRF(Omega_1, n, xk, xj)
    mode_sum_2 = getFRF(Omega_2, n, xk, xj)
    w_1 = Fk1 * np.sin(Omega_1 * t_lin) * np.repeat(np.expand_dims(mode_sum_1, axis=1), t_lin.shape[0], axis=1)
    w_2 = Fk2 * np.sin(Omega_2 * t_lin) * np.repeat(np.expand_dims(mode_sum_2, axis=1), t_lin.shape[0], axis=1)
    w = w_1 + w_2
    w_interp = RegularGridInterpolator((xj, t_lin), w)
    points_interp = np.array(np.meshgrid(x_domain.detach().cpu().numpy() * Lx, t_domain.detach().cpu().numpy() * t_tild)).T.reshape(-1,2)
    w_ad = w_interp(points_interp).reshape(x_domain.shape[0], t_domain.shape[0])
    w_ad *= np.max(np.abs(sol))/np.max(np.abs(w_ad))

    plt.close()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    n1 = 10
    ax1.plot(x_domain, w_ad[:,n1], label='Analytical', color='red')
    ax1.plot(x_domain, np.mean(sol[:,:,n1, 1], axis=1), label='NN')
    ax1.set_xlabel(r'$\hat{x}$')
    ax1.set_ylabel(r'$w$')
    ax1.set_title(f'$\\hat{{t}} = {n1 * steps[2].item():.2f}$')

    n2 = 45
    ax2.plot(x_domain, w_ad[:,n2], label='Analytical', color='red')
    ax2.plot(x_domain, np.mean(sol[:,:,n2, 1], axis=1), label='NN')
    ax2.set_xlabel(r'$\hat{x}$')
    ax2.set_title(f'$\\hat{{t}} = {n2 * steps[2].item():.2f}$')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{dir_model}/displ_snap.png')


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

ev_interp = {}
if import_abq:
    t = torch.unique(t).detach().cpu().numpy()
    for key, interpolator in interps.items():
       ev_interp[key] = interpolator(t * t_tild) 
    scale_keys = ['Kinetic', 'Strain', 'AVGV', 'AVGA']
    for i, key in enumerate(scale_keys):
        ev_interp[key] *= maxscale[i]/np.max(ev_interp[key])

    plt.figure()
    plt.plot(t, ev_interp['AVGU'], label='FEM')
    plt.plot(t, sol1D, label='NN')
    plt.xlabel(r'$\hat{t}$')
    plt.legend()
    plt.savefig(f'{dir_model}/FEMNNu.png')

    plt.figure()
    plt.plot(t, ev_interp['AVGV'], label='FEM')
    plt.plot(t, vmid, label='NN')
    plt.xlabel(r'$\hat{t}$')
    plt.legend()
    plt.savefig(f'{dir_model}/FEMNNv.png')

    plt.figure()
    plt.plot(t, ev_interp['AVGA'], label='FEM')
    plt.plot(t, amid, label='NN')
    plt.xlabel(r'$\hat{t}$')
    plt.legend()
    plt.savefig(f'{dir_model}/FEMNNa.png')

    plt.figure()
    plt.plot(t, ev_interp['Kinetic'], label='FEM')
    plt.plot(t, T, label='NN')
    plt.legend()
    plt.xlabel(r'$\hat{t}$')
    plt.savefig(f'{dir_model}/FEMNNT.png')

    plt.figure()
    plt.plot(t, ev_interp['Strain'], label='FEM')
    plt.plot(t, V, label='NN')
    plt.legend()
    plt.xlabel(r'$\hat{t}$')
    plt.savefig(f'{dir_model}/FEMNNPot.png')

if getzip:
    import os
    import shutil

    def create_zip(file_paths, zip_name):
        shutil.make_archive(zip_name, 'zip', file_paths)

    timenow = get_current_time(fmt='%m-%d %H:%M')

    create_zip(dir_model, f'model_FF-{timenow}')
    create_zip(dir_logs, f'logs_FF-{timenow}')