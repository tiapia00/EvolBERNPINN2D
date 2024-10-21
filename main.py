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
from scipy.signal import find_peaks

def compute_cross_spectrum(signal1, signal2, fs):
    """
    Computes the cross-spectrum of two signals.
    
    Parameters:
    signal1 : numpy array
        The first time-domain signal.
    signal2 : numpy array
        The second time-domain signal.
    fs : float
        The sampling frequency of the signals.

    Returns:
    freqs : numpy array
        Frequencies corresponding to the cross-spectrum.
    cross_spectrum : numpy array
        The computed cross-spectrum of the two signals.
    """
    # Fourier Transform of both signals
    fft_signal1 = np.fft.fft(signal1)
    fft_signal2 = np.fft.fft(signal2)
    
    # Frequency axis
    freqs = np.fft.fftfreq(len(signal1), 1/fs)
    
    # Cross-spectrum (fft_signal1 * conj(fft_signal2))
    cross_spectrum = fft_signal1 * np.conj(fft_signal2)
    
    return freqs, cross_spectrum

def find_nearest_peak_difference(freqs, cross_spectrum):
    """
    Finds the difference in frequency between the two nearest peaks in the cross-spectrum.
    
    Parameters:
    freqs : numpy array
        Frequencies corresponding to the cross-spectrum.
    cross_spectrum : numpy array
        The computed cross-spectrum of the two signals.

    Returns:
    nearest_freq_diff : float
        The difference in frequency between the two nearest peaks.
    """
    # Magnitude of the cross-spectrum
    magnitude = np.abs(cross_spectrum)
    
    # Find peaks in the magnitude spectrum
    peaks, _ = find_peaks(magnitude)
    
    # Check if there are at least two peaks
    if len(peaks) >= 2:
        # Get the frequencies of these peaks
        peak_freqs = freqs[peaks]
        # Sort peak frequencies in ascending order
        peak_freqs = np.sort(peak_freqs)
        
        # Find the smallest difference between adjacent peaks
        nearest_freq_diff = np.min(np.diff(peak_freqs))
        return nearest_freq_diff
    else:
        # Return None if fewer than two peaks are found
        return None

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
scaley = 1 

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
        scaley,
        in_penalty,
        device,
        interpVbeam,
        interpTbeam,
        t_tild,
        lr
    )

if retrain_PINN:
    dir_model = pass_folder('model')
    dir_logs = pass_folder('model/logs')

    pinn_trained = train_model(pinn, loss_fn=loss_fn, points=points, learning_rate=lr,
                               max_epochs=epochs, path_logs=dir_logs)

    model_name = f'{lr}_{epochs}_{dim_hidden}.pth'
    model_path = os.path.join(dir_model, model_name)

    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(dim_hidden, w0, n_hidden, multux, multuy, device).to(device)
    filename = get_last_modified_file('model', '.pth')

    dir_model = os.path.dirname(filename)
    print(f'Target for outputs: {dir_model}\n')

    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

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
sol, V, T = obtainsolt_u(pinn_trained, space, t, nsamples, par, steps, device)
plot_energy(torch.unique(t, sorted=True).detach().cpu().numpy(), V, T, dir_model)

sol1D = sol[sol.shape[1]//2,sol.shape[1]//2,:,1]
nfft = sol1D.shape[0]
window = np.hanning(nfft)
beamdispl = interpdisplbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Van = interpVbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)
Tan = interpTbeam(torch.unique(t, sorted=True).detach().cpu().numpy() * t_tild)

errV = (calculateRMS(V, steps[2], tmax) - calculateRMS(Van, steps[2], tmax))/(
        calculateRMS(Van, steps[2], tmax)
).item()
errT = (calculateRMS(T, steps[2], tmax) - calculateRMS(Tan, steps[2], tmax))/(
        calculateRMS(Tan, steps[2], tmax)
).item()

freqs, crossspectrum = compute_cross_spectrum(sol1D, beamdispl, 1/steps[2].item())
freqdiff = find_nearest_peak_difference(freqs, crossspectrum)
print(freqdiff) 
errfreq = freqdiff/1

with open(f'{dir_model}/freqerr.txt', 'w') as file:
    file.write(f"errfreq = {errfreq}\n"
               f"errV = {-errV}\n"
               f"errT = {-errT}\n")

sol = sol.reshape(n_space**2, n_time, 2)
plot_sol(sol, spacein, t, dir_model)
plot_average_displ(sol, t, dir_model)

import os
import shutil

def create_zip(file_paths, zip_name):
    shutil.make_archive(zip_name, 'zip', file_paths)

timenow = get_current_time(fmt='%m-%d %H:%M')

create_zip(dir_model, f'model_FF-{timenow}')
create_zip(dir_logs, f'logs_FF-{timenow}')