from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder, delete_old_files, get_current_time
from pinn import *
from par import Parameters, get_params
from analytical import obtain_analytical_free
import matplotlib.animation as animation

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
my_beam = Beam(Lx, E, rho, h, h/3, n_space_beam)
modes = 2

t_tild, w, V0 = obtain_analytical_free(my_beam, w0, t, 500, 1)

"""
fig = plt.figure()
ax = plt.axes()

def drawframe(i):
    ax.clear()

    plt.title("Solution")
    ax.set_ylim(np.min(w), np.max(w))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$w$')
    ax.plot(my_beam.xi, w[:, i], color='blue')
    return ax

ani = animation.FuncAnimation(
    fig, drawframe, frames=w.shape[1], blit=False, repeat=True)

plt.show()
"""

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, dim_hidden, n_hidden, multux, multuy, multhyperx, lr, epochs = get_params(par.pinn_par)

L_tild = Lx
x_domain = torch.linspace(0, Lx, n_space)/Lx
y_domain = torch.linspace(0, Ly, n_space)/Lx
t_domain = torch.linspace(0, T, n_time)

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, multhyperx, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'initial_points_hyper': grid.get_initial_points_hyper(),
    'boundary_points': grid.get_boundary_points(),
    'all_points': grid.get_all_points()
}

adim = (mu/lam, (lam+mu)/lam, rho/(lam*t_tild.item()**2)*Lx**2)
par = {"Lx": Lx,
        "w0": w0,
        "lam": lam,
        "mu":mu,
        "rho": rho,
        "t_ast": t_tild}

inpoints = torch.cat(points["initial_points"], dim=1)
spacein = inpoints[:,:2]
cond0 = initial_conditions(spacein, w0)
condx = cond0[:,1].reshape(n_space, n_space)
condx = condx[:,0]
yf, freq = calculate_fft(condx.detach().cpu().numpy(), steps[0].item(), x_domain.cpu().numpy())

def extractcompfft(yf: np.ndarray, freq: np.ndarray):
    lastpos = np.where(freq > 0)[0][-1]
    freqpos = freq[:lastpos]
    magnpos = np.abs(yf[:lastpos])
    magnpos[1:-1] *= 2

    return magnpos, freqpos

magnpos, freqpos = extractcompfft(yf, freq)
magnpos *= 1./np.max(magnpos)

modesx = [0]
modesy = [1, 2]

pinn = (PINN(dim_hidden, w0, n_hidden, multux, multuy, modesx, modesy, device).to(device))

#En0 = calc_initial_energy(pinn, n_space, points, device)

in_penalty = torch.tensor([1., 1., 1., 1., 1.])
loss_fn = Loss(
        points,
        n_space,
        n_time,
        h/3,
        w0,
        steps,
        adim,
        par,
        in_penalty,
        device
    )


if retrain_PINN:
    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    pinns_trained = train_model(pinn, loss_fn=loss_fn, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs, modeldir=dir_model)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    #torch.save(pinns_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, w0, n_hidden, multux, multuy, magnpos, device).to(device)
    filename = get_last_modified_file('model', '.pth')

    dir_model = os.path.dirname(filename)
    print(f'Target for outputs: {dir_model}\n')

    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

for pinn_trained in pinns_trained:
    pinn_trained.eval()

tin = inpoints[:,-1].unsqueeze(1)
z = pinn(spacein, tin)
scaling = w0/torch.max(z).item()
v = calculate_speed(z, tin, par)
z = torch.cat([par['w0'] * z, v], dim=1)

plot_initial_conditions(scaling * z, cond0, spacein, dir_model)

allpoints = torch.cat(points["all_points"], dim=1)
space = allpoints[:,:2]
t = allpoints[:,-1].unsqueeze(1)
nsamples = (n_space, n_space) + (n_time,)
sol = obtainsolt_u(pinns_trained, space, t, nsamples)
sol *= scaling
plot_sol(sol.reshape(n_space*n_space, n_time, 2), spacein, t, dir_model)
plot_rms_space_mid(sol, t, steps, dir_model)

import os
import shutil

def create_zip(file_paths, zip_name):
    shutil.make_archive(zip_name, 'zip', file_paths)

timenow = get_current_time(fmt='%m-%d %H:%M')

create_zip(dir_model, f'model_FF-{timenow}')
create_zip(dir_logs, f'logs_FF-{timenow}')