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

t_beam, t_tild, w, V_an, Ek_an = obtain_analytical_free(my_beam, w0, t, 2000, 1)

interpdisplbeam = make_interp_spline(t_beam, w[w.shape[0]//2,:])
interpVbeam = make_interp_spline(t_beam, V_an, k=5)
interpTbeam = make_interp_spline(t_beam, Ek_an, k=5)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, dim_hidden, n_hidden, multux, multuy, multhyperx, lr, epochs = get_params(par.pinn_par)
b = h/3

L_tild = Lx
x_domain = torch.linspace(0, Lx, n_space)/Lx
y_domain = torch.linspace(0, Ly, n_space)/Lx
t_domain = torch.linspace(0, T, n_time)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, multhyperx, y_domain, t_domain, device)
hypert = 1 

points = {
    'res_points': grid.get_interior_points_train(),
    'initial_points': grid.get_initial_points(),
    'boundary_points': grid.generate_grid_bound(),
    'initial_points_hyper': grid.get_initial_points_hyper(),
    'all_points_eval': grid.get_all_points_eval(hypert),
}

adim = (mu/lam, (lam+mu)/lam, rho/(lam*t_tild.item()**2)*Lx**2)
par = {"Lx": Lx,
        "w0": w0,
        "lam": lam,
        "mu":mu,
        "rho": rho,
        "b": b,
        "t_ast": t_tild}

inpoints = torch.cat(points["initial_points"], dim=1)
spacein = inpoints[:,:2]
cond0 = initial_conditions(spacein, w0)
condx = cond0[:,1].reshape(n_space, n_space)
condx = condx[:,0]

pinn = PINN(dim_hidden, w0, n_hidden, multux, multuy, device).to(device)

in_penalty = torch.tensor([1., 1., 1., 1.])
in_penalty.requires_grad_(False)
loss_fn = Loss(
        points,
        n_space,
        n_time,
        b,
        w0,
        steps,
        adim,
        par,
        in_penalty,
        device,
        interpVbeam,
        interpTbeam,
        t_tild,
        lr
    )

dir_model = pass_folder('model')
dir_logs = pass_folder('model/logs')
if load:
    filename = 'load/0.0001_3000_(1, 60).pth'
    dir_load = os.path.dirname(filename)
    pinn.load_state_dict(torch.load(filename, map_location=device))
    with np.load(f'{dir_load}/data.npz') as data:
        loss_fn.gamma = -0.3819
        
if train:
    pinn_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
        max_epochs=epochs, path_logs=dir_logs, path_model=dir_model)
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
tmax = torch.max(t).item()
nsamples = (n_space, n_space) + (n_time,)
sol, V, T = obtainsolt_u(pinn_trained, space, t, nsamples, hypert, par, steps, device)
plot_energy(torch.unique(t, sorted=True).detach().cpu().numpy(), V, T, dir_model)

sol1D = sol[sol.shape[1]//2,sol.shape[1]//2,:,1]
nfft = sol1D.shape[0]
window = np.hanning(nfft)
beamdispl = interpdisplbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van = interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van *= np.max(V)/np.max(Van)
Tan = interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Tan *= np.max(T)/np.max(Tan)

errV = (calculateRMS(V, steps[2], tmax) - calculateRMS(Van, steps[2], tmax))/(
        calculateRMS(Van, steps[2], tmax)
).item()
errT = (calculateRMS(T, steps[2], tmax) - calculateRMS(Tan, steps[2], tmax))/(
        calculateRMS(Tan, steps[2], tmax)
).item()

freqsfft = np.fft.rfftfreq(nfft, steps[2].item())
fftpredicted = np.fft.rfft(window * sol1D)
fftan = np.fft.rfft(window * beamdispl)
freqmaxpred = np.argmax(fftpredicted)
freqmaxan = np.argmax(fftan)

errfreq = (freqsfft[freqmaxan] - freqsfft[freqmaxpred])/freqsfft[freqmaxan]

with open(f'{dir_model}/freqerr.txt', 'w') as file:
    file.write(f"errfreq = {errfreq}\n"
               f"errV = {-errV}\n"
               f"errT = {-errT}\n")

sol = sol.reshape(n_space**2, n_time * hypert, 2)
plot_sol(sol, spacein, t, dir_model)
plot_average_displ(sol, t, dir_model)

data = {
    'hatw_mid': sol1D,
    'anw_mid': beamdispl, 
    'hatT': T,
    'hatV': V,
    'anT': interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild),
    'anV': interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild),
    'gamma': loss_fn.gamma
}

np.savez(f'{dir_model}/data.npz', **data)


if plotloss:
    grad_accumulation = {name: 0.0 for name, param in pinn_trained.named_parameters()}

    num_epochs = 20 

    pbar = tqdm(total=num_epochs, desc="", position=0)
    for epoch in range(num_epochs):
        loss = loss_fn(pinn_trained, update=False)[0]        # Compute loss
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

    npoints = 30
    perturb_range = np.linspace(-0.5, 0.5, npoints)
    loss_profile = np.zeros((npoints, npoints))

    original_params = {name: param.clone() for name, param in pinn_trained.named_parameters()}

    for i, alpha in enumerate(perturb_range):
        for j, beta in enumerate(perturb_range):
            # Apply perturbations
            for idx, perturb in enumerate([alpha, beta]):
                param_name = most_influential_params[idx]
                # Split the parameter name into layer_name and param_type
                layer_name, param_type = param_name.rsplit('.', 1)  # Split from the right on the last dot
                
                # Access the layer using the name and index
                if '.' in layer_name:  # Handle if layer_name includes an index (e.g., 'hid_space_layers_x.0')
                    layer_base_name, layer_index = layer_name.rsplit('.', 1)  # Separate base name and index
                    layer = getattr(pinn_trained, layer_base_name)[int(layer_index)]  # Access the layer by index
                else:
                    layer = getattr(pinn_trained, layer_name)  # Get the layer directly

                param = getattr(layer, param_type)  # Get the parameter (weight/bias)

                # Apply the perturbation
                param.data = original_params[param_name].data + perturb
            
            # Calculate the loss with the perturbed parameters
            loss_profile[i, j] = loss_fn(pinn_trained, update=False)[0].item()

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

def create_zip(file_paths, zip_name):
    shutil.make_archive(zip_name, 'zip', file_paths)

if getzip:
    import os
    import shutil
    timenow = get_current_time(fmt='%m-%d %H:%M')
    create_zip(dir_model, f'model_FF-{timenow}')
    create_zip(dir_logs, f'logs_FF-{timenow}')