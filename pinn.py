from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.optim as optim


def initial_conditions(initial_points: tuple, w0: float, i: float = 1) -> torch.tensor:
    x, y, _ = initial_points
    ux0 = torch.zeros_like(x)
    uy0 = w0*torch.sin(torch.pi*i*x)
    dotux0 = torch.zeros_like(x)
    dotuy0 = torch.zeros_like(x)
    return torch.cat((ux0, uy0, dotux0, dotuy0), dim=1)

def geteps(space, output, nsamples, device):
    duxdxy = torch.autograd.grad(output[:, 0].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    duydxy = torch.autograd.grad(output[:, 1].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    H = torch.zeros(space.size()[0], space.size()[1], space.size()[1], device=device)
    H[:, 0, :] = duxdxy 
    H[:, 1, :] = duydxy 
    H = H.reshape(nsamples[0], nsamples[1], 2, 2) 
    eps = H
    eps[:, :, [0, 1], [1, 0]] = 0.5 * (eps[:, :, 0, 1] + eps[:, :, 1, 0]).unsqueeze(2).expand(nsamples[0], nsamples[1], 2)

    return eps

def material_model(eps, mat_par: tuple,  device):
    tr_eps = eps.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) 
    lam = mat_par[0]
    mu = mat_par[1]
    sig = 2 * mu * eps + lam * torch.einsum('ij,kl->ijkl', tr_eps, torch.eye(eps.size()[-1], device=device)) 
    psi = torch.einsum('ijkl,ijkl->ij', eps, sig)

    return psi, sig

def getspeed(output, t, device):
    n = output.shape[0]

    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones(n, 1, device=device),
                create_graph=True, retain_graph=True)[0]
    vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones(n, 1, device=device),
                create_graph=True, retain_graph=True)[0]
    
    return torch.cat([vx, vy], dim=1)

def getkinetic(speed: torch.tensor, nsamples: tuple, rho: float, ds: tuple):
    dx = ds[0]
    dy = ds[1]

    speed = speed.reshape(nsamples[0], nsamples[1], nsamples[2], 2)
    magnitude = torch.norm(speed, p=2, dim=3)
    ### ASSUMPTION: t = 1 ###
    kinetic = 1/2 * rho * torch.trapezoid(torch.trapezoid(magnitude, dx=dy, dim=1),
            dx = dx, dim=0)

    return kinetic

def getPsi(psi: torch.tensor, ds: tuple):
    dx = ds[0]
    dy = ds[1]

    Psi = torch.trapezoid(torch.trapezoid(y = psi, dx = dy, dim=1), dx=dx, dim=0)

    return Psi
    

class Grid:
    def __init__(self, x_domain, y_domain, t_domain, device):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.device = device
        self.requires_grad = True
        self.grid_init = self.generate_grid_init()
        self.grid_bound = self.generate_grid_bound()

    def generate_grid_init(self):
        x = self.x_domain
        y = self.y_domain
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

        x_grid = x_grid.reshape(-1, 1)
        y_grid = y_grid.reshape(-1, 1)
        t0 = torch.zeros_like(x_grid)

        grid_init = torch.cat((x_grid, y_grid, t0), dim=1)

        return grid_init

    def generate_grid_bound(self):
        """
             .+------+
           .' |    .'|
          +---+--+'  |
          |   |  |   |
        x |  ,+--+---+
          |.'    | .' t
          +------+'
             y
        down , up : extremes of the beam
        """

        x_linspace = self.x_domain
        y_linspace = self.y_domain
        t_linspace = self.t_domain[1:]

        x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
        y_grid, _ = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

        x_grid = x_grid.reshape(-1, 1)
        y_grid = y_grid.reshape(-1, 1)
        t_grid = t_grid.reshape(-1, 1)

        x0 = torch.full_like(
            t_grid, self.x_domain[0])
        x1 = torch.full_like(
            t_grid, self.x_domain[1])
        y0 = torch.full_like(
            t_grid, self.y_domain[0])
        y1 = torch.full_like(
            t_grid, self.y_domain[1])

        down = torch.cat((x0, y_grid, t_grid), dim=1)

        up = torch.cat((x1, y_grid, t_grid), dim=1)

        left = torch.cat((x_grid, y0, t_grid), dim=1)

        right = torch.cat((x_grid, y1, t_grid), dim=1)
        bound_points = torch.cat((down, up, left, right), dim=0)

        return (down, up, left, right, bound_points)

    def get_initial_points(self):
        x_grid = self.grid_init[:, 0].unsqueeze(1).to(self.device)
        x_grid.requires_grad_(True)

        y_grid = self.grid_init[:, 1].unsqueeze(1).to(self.device)
        y_grid.requires_grad_(True)

        t0 = self.grid_init[:, 2].unsqueeze(1).to(self.device)
        t0.requires_grad_(True)

        return (x_grid, y_grid, t0)

    def get_boundary_points(self):
        down = tuple(self.grid_bound[0][:, i].unsqueeze(1).requires_grad_().to(
            self.device) for i in range(self.grid_bound[0].shape[1]))
        up = tuple(self.grid_bound[1][:, i].unsqueeze(1).requires_grad_().to(
            self.device) for i in range(self.grid_bound[1].shape[1]))
        left = tuple(self.grid_bound[2][:, i].unsqueeze(1).requires_grad_().to(
            self.device) for i in range(self.grid_bound[2].shape[1]))
        right = tuple(self.grid_bound[3][:, i].unsqueeze(1).requires_grad_().to(
            self.device) for i in range(self.grid_bound[3].shape[1]))
        return (down, up, left, right)

    def get_interior_points(self):
        x_raw = self.x_domain[1:-1]
        y_raw = self.y_domain[1:-1]
        t_raw = self.t_domain[1:]
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

        x = grids[0].reshape(-1, 1)
        y = grids[1].reshape(-1, 1)
        t = grids[2].reshape(-1, 1)

        grid = torch.cat((x, y, t), dim=1)

        x = grid[:, 0].unsqueeze(1).to(self.device)
        x.requires_grad = True
        y = grid[:, 1].unsqueeze(1).to(self.device)
        y.requires_grad = True
        t = grid[:, 2].unsqueeze(1).to(self.device)
        t.requires_grad = True

        return (x, y, t)

    def get_all_points(self):
        x_all, y_all, t_all = torch.meshgrid(self.x_domain, self.y_domain,
                                             self.t_domain, indexing='ij')
        x_all = x_all.reshape(-1,1).to(self.device)
        y_all = y_all.reshape(-1,1).to(self.device)
        t_all = t_all.reshape(-1,1).to(self.device)

        x_all.requires_grad_(True)
        y_all.requires_grad_(True)
        t_all.requires_grad_(True)

        return (x_all, y_all, t_all)


class RBF(nn.Module):
    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / \
            torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi

class TrigAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def parabolic(a, x):
    return (a * x ** 2 - a * x)


class PINN(nn.Module):
    def __init__(self,
                 dim_hidden: tuple,
                 n_hidden_space: int,
                 points: dict,
                 w0: float,
                 prop: dict,
                 initial_conditions: callable,
                 device,
                 a: float = 1,
                 act=nn.Sigmoid(),
                 n_hidden: int = 1,
                 ):

        super().__init__()

        self.w0 = w0
        self.a = a
        self.n_mode_spacex = dim_hidden[0]
        self.n_mode_spacey = dim_hidden[1]

        multipliers_x = torch.arange(1, self.n_mode_spacex + 1, device=device)
        self.Bx = 0.1 * torch.ones((2, self.n_mode_spacex), device=device)
        self.Bx[0,:] *= multipliers_x

        multipliers_y = torch.arange(1, self.n_mode_spacey + 1, device=device) **2
        self.By = 0.005 * torch.ones((2, self.n_mode_spacey), device=device)
        self.By[0,:] *= multipliers_y

        self.in_time = nn.Linear(1, dim_hidden[2])
        self.act_time = TrigAct()

        self.hid_space_layers_x = nn.ModuleList()
        for i in range(n_hidden - 1):
            self.hid_space_layers_x.append(nn.Linear(2 * self.n_mode_spacex, 2 * self.n_mode_spacex))
            self.hid_space_layers_x.append(act)
        self.outFC_space_x = nn.Linear(2 * self.n_mode_spacex, 1)

        self.hid_space_layers_y = nn.ModuleList()
        for i in range(n_hidden - 1):
            self.hid_space_layers_y.append(nn.Linear(4 * self.n_mode_spacey, 4 * self.n_mode_spacey))
            self.hid_space_layers_y.append(act)
        self.outFC_space_y = nn.Linear(4 * self.n_mode_spacey, 1)

        self.mid_time_layer = nn.Linear(dim_hidden[2], 2)

    def parabolic(self, x):
        return (self.a * x ** 2 - self.a * x)

    @staticmethod
    def apply_filter(alpha):
        return (torch.tanh(alpha))

    @staticmethod
    def apply_compl_filter(alpha):
        return (1-torch.tanh(alpha))

    def fourier_features_ux(self, space):
        x_proj = space @ self.Bx
        return torch.cat([torch.sin(np.pi * x_proj),
                torch.cos(np.pi * x_proj)], dim=1)

    def fourier_features_uy(self, space):
        x_proj = space @ self.By
        return torch.cat([torch.sin(np.pi * x_proj), torch.cos(np.pi * x_proj),
                torch.sinh(np.pi * x_proj), torch.cosh(np.pi * x_proj)], dim=1)

    def forward(self, space, t):
        time = t

        fourier_space_x = self.fourier_features_ux(space)
        fourier_space_y = self.fourier_features_uy(space)

        x_in = fourier_space_x
        y_in = fourier_space_y

        for layer in self.hid_space_layers_x:
            x_in= layer(x_in)

        for layer in self.hid_space_layers_y:
            y_in= layer(y_in)

        x_FC = self.outFC_space_x(x_in)
        y_FC = self.outFC_space_y(y_in)

        out_space_FC = torch.cat([x_FC, y_FC], dim=1)

        t = self.in_time(time)

        t_act = self.act_time(t)

        mid_t = self.mid_time_layer(t_act)

        mid_x = torch.sin(space[:,0].reshape(-1,1) * np.pi) * out_space_FC
        #mid_x = out_space_FC

        merged = mid_x * mid_t

        act_global = self.apply_filter(time.repeat(1, 2)) * merged

        x = space[:,0].unsqueeze(1)
        y = space[:,1].unsqueeze(1)
        init = initial_conditions((x,y,t), self.w0)[:,:2]
        act_init = self.apply_compl_filter(time.repeat(1, 2)) * init

        out = act_global + act_init

        return out

    def device(self):
        return next(self.parameters()).device


class Calculate:
    def __init__(
        self,
        initial_condition: Callable,
        # lam, mu, rho
        m_par: tuple,
        points: dict,
        n_space: tuple,
        n_time: int,
        steps_int: tuple,
        w0: float,
        in_penalty: np.ndarray,
        device: torch.device,
    ):
        self.initial_condition = initial_condition
        self.m_par = m_par
        self.points = points
        self.w0 = w0
        self.nsamples = n_space + (n_time,)
        self.steps = steps_int
        self.penalty = in_penalty
        self.device = device

    def gete0(self, pinn):
        x, y, t = self.points['initial_points']
        nsamples = (self.nsamples[0], self.nsamples[1], 1)
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        lam, mu, rho = self.m_par
        dx, dy, dt = self.steps

        eps = geteps(space, output, nsamples, self.device)
        psi, sig = material_model(eps, (lam, mu), self.device)
        Psi = getPsi(psi, (dx, dy)).reshape(-1)

        speed = getspeed(output, t, self.device)
        K = getkinetic(speed, nsamples, rho, (dx, dy)).reshape(-1)

        criterion = nn.MSELoss()

        loss = criterion(Psi - K, torch.zeros_like(Psi))

        return (Psi, K) 

    def getenergy(self, pinn, verbose=False):
        x, y, t = self.points['res_points']
        nsamples = (self.nsamples[0]-2, self.nsamples[1]-2, self.nsamples[2]-1)
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        lam, mu, rho = self.m_par
        dx, dy, dt = self.steps

        eps = geteps(space, output, nsamples , self.device)
        psi, sig = material_model(eps, (lam, mu), self.device)
        Psi = getPsi(psi, (dx, dy)).reshape(-1)

        speed = getspeed(output, t, self.device)
        K = getkinetic(speed, nsamples, rho, (dx, dy)).reshape(-1)

        criterion = nn.MSELoss()

        loss = criterion(Psi - K, torch.zeros_like(Psi))

        if verbose:
            return (Psi, K) 
        else:
            return loss


    def initial_loss(self, pinn):
        init_points = self.points['initial_points']
        x, y, t = init_points
        space = torch.cat([x, y], dim=1)

        output = pinn(space, t)

        initial_speed = initial_conditions(init_points, self.w0)[:,2:]

        m = self.penalty[0]
        loss = m * (output - initial_speed).pow(2).mean()

        return loss


    def bound_loss(self, pinn):
        ### IMPLEMENT ###

        regions = []
        for region in self.points['boundary_points']:
            x, y, t = region
            space_region = torch.cat([x, y], dim=1)
            regions.append((space_region, t))

        m = self.penalty[1]
        loss = m * loss

        return loss

    def update_penalty(self, max_grad: float, mean: list, alpha: float = 0.):
        lambda_o = np.array(self.penalty)
        mean = np.array(mean)
        
        lambda_n = max_grad / (lambda_o * (np.abs(mean)+0.1))

        self.penalty = (1-alpha) * lambda_o + alpha * lambda_n


    def verbose(self, pinn):
        res_loss = self.res_loss(pinn)
        en_dev = self.en_loss(pinn)
        init_loss = self.initial_loss(pinn)
        bound_loss = self.bound_loss(pinn)
        loss = res_loss + bound_loss + init_loss + en_dev

        return loss, res_loss, (bound_loss, init_loss, en_dev)

    def __call__(self, pinn):
        return self.verbose(pinn)


def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
    points: dict
) -> PINN:

    writer = SummaryWriter(log_dir=path_logs)

    optimizer = optim.Adam(nn_approximator.parameters(), lr = learning_rate)
    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        if epoch != 0 and epoch % 100 == 0 :
            _, res_loss, losses = loss_fn(nn_approximator)

            res_loss.backward()
            max_grad = get_max_grad(nn_approximator)
            optimizer.zero_grad()

            means = []

            for loss in losses:
                loss.backward()
                means.append(get_mean_grad(nn_approximator))
                optimizer.zero_grad()

            loss_fn.update_penalty(max_grad, means)

        loss, res_loss, losses = loss_fn(nn_approximator)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': res_loss.item(),
            'init': losses[0].item(),
            'boundary': losses[1].item(),
            'en_dev': losses[2].item()
        }, epoch)

        writer.add_scalars('Penalty_terms', {
            'init': loss_fn.penalty[0].item(),
            'boundary': loss_fn.penalty[1].item(),
            'en_dev': loss_fn.penalty[2].item()
        }, epoch)

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    writer.close()

    return nn_approximator


def return_adim(L_tild, t_tild, rho: float, mu: float, lam: float):
    z_1 = t_tild**2/(L_tild*rho)*mu
    z_2 = z_1/mu*lam
    z = np.array([z_1, z_2])
    z = torch.tensor(z)
    return z




    En = En.detach()

    return En

def calc_energy(pinn_trained: PINN, points: dict, n_space: int, n_time: int, device) -> tuple:
    x, y, t = points['all_points']
    space = torch.cat([x, y], dim=1)

    output = pinn_trained(space, t)

    d_en, d_en_p, d_en_k = calc_den(space, t, output, device)

    x = x.reshape(n_space, n_space, n_time)
    y = y.reshape(n_space, n_space, n_time)
    t = t.reshape(n_space, n_space, n_time)

    d_en_k = d_en_k.reshape(n_space, n_space, n_time)
    d_en_p = d_en_p.reshape(n_space, n_space, n_time)
    d_en = d_en.reshape(n_space, n_space, n_time)

    d_en_k_y = torch.trapz(d_en_k, y[0,:,0], dim=1)
    d_en_p_y = torch.trapz(d_en_p, y[0,:,0], dim=1)
    d_en_y = torch.trapz(d_en, y[0,:,0], dim=1)

    En_k_t = torch.trapz(d_en_k_y, x[:,0,0], dim=0)
    En_p_t = torch.trapz(d_en_p_y, x[:,0,0], dim=0)
    En_t = torch.trapz(d_en_y, x[:,0,0], dim=0)

    t = torch.unique(t)

    return (t, En_t, En_p_t, En_k_t)

def calculate_speed(pinn_trained: PINN, points: tuple, device: torch.device) -> torch.tensor:
    x, y, t = points
    space = torch.cat([x, y], dim=1)
    n = space.shape[0]

    output = pinn_trained(space, t)

    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones(n, 1, device=device),
             create_graph=True, retain_graph=True)
    vy = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones(n, 1, device=device),
             create_graph=True, retain_graph=True)

    return torch.cat([vx, vy], dim=1)

def df_num_torch(dx: float, y: torch.tensor):
    dy = torch.diff(y)

    derivative = torch.zeros_like(y)

    # Forward difference for the first point
    derivative[0] = dy[0] / dx

    # Central difference for the middle points
    for i in range(1, len(y) - 1):
        dy_avg = (y[i+1] - y[i-1]) / 2
        derivative[i] = dy_avg / dx

    # Backward difference for the last point
    derivative[-1] = dy[-1] / dx

    return derivative


def get_max_grad(pinn: PINN):
    max_grad = None

    for name, param in pinn.named_parameters():
        if param.requires_grad:
            param_max_grad = param.grad.abs().max().item()
            if max_grad is None or param_max_grad > max_grad:
                max_grad = param_max_grad

    return max_grad


def get_mean_grad(pinn: PINN):
    all_grads = []

    for param in pinn.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))

    all_grads = torch.cat(all_grads)

    mean = all_grads.mean().cpu().numpy()

    return mean




