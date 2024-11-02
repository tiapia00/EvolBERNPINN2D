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

load = True
train = True
plotloss = False
getzip = True
plots = False
import_abq = True

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
    print(data_abq)

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
    filename = 'load/0.001_10000_(1, 60).pth'
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
sol, V, T = obtainsolt_u(pinn_trained, space, t, nsamples, 1, par, steps, device)

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