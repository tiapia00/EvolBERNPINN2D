from plots import *
from beam import Beam
import os
import torch
from read_write import get_last_modified_file, pass_folder
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

restartraining = True 

def get_step(tensors: tuple):
    a, b, c = tensors

    step_a = torch.diff(a)[0]
    step_b = torch.diff(b)[0]
    step_c = torch.diff(c)[0]

    return (step_a, step_b, step_c)

par = Parameters()

Lx, t, h, w0 = get_params(par.beam_par)
E, rho, _ = get_params(par.mat_par)
my_beam = Beam(Lx, E, rho, h, 4e-3, 2000)

t_beam = np.linspace(0, t, 2000)
w, ens_an, T_tild = obtain_analytical_free(my_beam, w0, t_beam)
ens_an = {'V': ens_an[:,0], 'T': ens_an[:,1]}
print(T_tild)
#t_points, sol = obtain_analytical_forced(par, my_beam, load_dist, t_tild, n)

lam, mu = par.to_matpar_PINN()

Lx, Ly, T, n_space, n_time, w0, multdim, nax, ntrans, nlayers, lr, epochs = get_params(par.pinn_par)

x_domain = torch.linspace(0, Lx, n_space[0])/Lx
y_domain = torch.linspace(0, Ly, n_space[1])/Lx
t_domain = torch.linspace(0, T, n_time)/T_tild

steps = get_step((x_domain, y_domain, t_domain))

grid = Grid(x_domain, y_domain, t_domain, device)

points = {
    'res_points': grid.get_interior_points(),
    'initial_points': grid.get_initial_points(),
    'all_points': grid.get_all_points()
}

prop = {'E': E, 'J': my_beam.J, 'm': rho * my_beam.A, 'A': my_beam.A}
m_par = (lam, mu, rho)
nsamples = n_space + (n_time,)
penalties = np.array([1,1.5])

calculate = Calculate(
        initial_conditions,
        m_par,
        points,
        nsamples,
        steps,
        w0,
        penalties,
        device
    )
"""
eigenNN = MLNet(4, 60, 9)
eigenNN.load_state_dict(torch.load('data//eigenestmodel.pth'))
input_eigen = torch.tensor([E, rho, Lx, Ly]).reshape(1,-1)
eigen = eigenNN(input_eigen)
ef_range = torch.load('data//ef_range.pt')
eigen = denormalizematr(eigen, ef_range)

omega_trans = eigen.squeeze(0)[:1]
omega_ax = eigen.squeeze(0)[:1]

nninbcs = NNinbc(20, 1).to(device)
nndist = NNd(40, 2).to(device)

if retrainaux:
    nninbcs = train_inbcs(nninbcs, calculate, 1000, 1e-3)
    torch.save(nninbcs.state_dict(), 'data//nnInbcs.pth')
    nndist = train_dist(nndist, calculate, 5000, 1e-3)
    torch.save(nndist.state_dict(), 'data//nnDist.pth')
else:
    nninbcs.load_state_dict(torch.load('data//nnInbcs.pth'))
    nndist.load_state_dict(torch.load('data//nnDist.pth'))
"""

x, y, t = points['initial_points']
x_in = x.to(device)
y_in = y.to(device)
t_in = t.to(device)
in_points = torch.cat([x_in, y_in, t_in], dim=1)
all_points = torch.cat(points['all_points'], dim=1)

pinn = PINN(multdim, nax, ntrans, w0, nlayers).to(device)

Psi_0, K_0 = calculate.gete0(pinn)

model_name = f'{lr}_{epochs}_{ntrans}.pth'

dir_model = pass_folder('model')
dir_logs = pass_folder('model/logs')

model_name = f'{lr}_{epochs}_{ntrans}.pth'
model_path = os.path.join(dir_model, model_name)

if restartraining:
    pinn_trained, ens_NN = train_model(pinn, calc=calculate, lr=lr,
            max_epochs=epochs, path_logs=dir_logs)
    torch.save(pinn_trained.state_dict(), model_path)

else:
    pinn_trained = PINN(multdim, nax, ntrans, w0, nlayers).to(device)
    ### Specify here filename ###
    filename = 'model//08-04//1714//0.001_2000_1.pth' 
    pinn_trained.load_state_dict(torch.load(filename, map_location=device))
    print(f'{filename} loaded.\n')

    pinn_trained, ens_NN = train_model(pinn_trained, calc=calculate, lr=lr,
            max_epochs=epochs, path_logs=dir_logs)
    torch.save(pinn_trained.state_dict(), model_path)

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

plot_energy(ens_NN, ens_an, t, t_beam, dir_model)
dPi, dT = obtain_deren(ens_NN, steps[2])
plot_deren(dPi, dT, t, dir_model)

f, modPI, angPI = calculate_fft(ens_NN['Pi'], steps[2], torch.unique(t, sorted=True))
f, modT, angT = calculate_fft(ens_NN['T'], steps[2], torch.unique(t, sorted=True))
plot_fft(f, modPI, angPI, modT, angT, dir_model)
