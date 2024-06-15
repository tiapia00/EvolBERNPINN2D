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
        x_all = x_all.reshape(-1,1)
        y_all = y_all.reshape(-1,1)
        t_all = t_all.reshape(-1,1)

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
                 initial_conditions: callable,
                 a: float = 0.1,
                 act=nn.Tanh(),
                 fourier_scale: float = 0.1,
                 n_hidden: int = 3):

        super().__init__()

        self.w0 = w0
        self.a = a

        self.fourier_dim = dim_hidden[0]
        self.fourier_scale = fourier_scale
        self.B = torch.randn((2, self.fourier_dim)) * self.fourier_scale
        self.b = 2 * np.pi * torch.rand(self.fourier_dim)

        time_dim = int(0.5 * dim_hidden[1] + 0.5)
        self.in_time_disp = nn.Linear(1, time_dim)
        self.in_time_speed = nn.Linear(1, time_dim)
        self.act_time = TrigAct()

        self.hid_space_layers = nn.ModuleList()
        for i in range(n_hidden - 1):
            self.hid_space_layers.append(nn.Linear(2 * self.fourier_dim, 2 * self.fourier_dim))
            self.hid_space_layers.append(act)
        self.outFC_space = nn.Linear(2 * self.fourier_dim, 2)

        self.mid_time_layer = nn.Linear(dim_hidden[1], 4)

    def parabolic(self, x):
        return (self.a * x ** 2 - self.a * x)

    @staticmethod
    def apply_filter(alpha):
        return (torch.tanh(alpha))

    @staticmethod
    def apply_compl_filter(alpha):
        return (1-torch.tanh(alpha))

    def fourier_features(self, x):
        x_proj = x @ self.B + self.b  # (batch_size, fourier_dim)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x, y, t):
        space = torch.cat([x, y], dim=1)
        time = t

        fourier_space = self.fourier_features(space)
        x_in = fourier_space

        for layer in self.hid_space_layers:
            x_in = layer(x_in)

        x_FC = self.outFC_space(x_in)

        t_disp = self.in_time_disp(time)
        t_speed = self.in_time_speed(time)

        t_disp_act = self.act_time(t_disp)
        t_speed_act = self.act_time(t_speed)

        t_stack = torch.cat([t_disp_act, t_speed_act], dim=1)

        mid_t = self.mid_time_layer(t_stack)

        mid_x = self.parabolic(space[:,0].reshape(-1,1)) * x_FC
        mid_x = mid_x.repeat(1, 2)

        merged = mid_x * mid_t

        act_global = self.apply_filter(time.repeat(1, 4)) * merged

        init = initial_conditions((x,y,t), self.w0)
        act_init = self.apply_compl_filter(t.repeat(1, 4)) * init

        out = act_global + act_init

        return out

    def device(self):
        return next(self.parameters()).device


def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.tensor:
    return pinn(x, y, t)


def df(output: torch.Tensor, inputs: list, var: int = 0) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine
    var = 0 : dux
    var = 1 : duy
    var = 2 : dux_t
    var = 3 : duy_t
    """
    df_value = output[:, var].unsqueeze(1)
    for _ in np.arange(len(inputs)):
        df_value = torch.autograd.grad(
            df_value,
            inputs[_],
            grad_outputs=torch.ones_like(inputs[_]),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value


class Loss:
    """Loss:
       z = T^2/(L*rho)
       z[0] = mu*z
       z[1] = lambda*z"""

    def __init__(
        self,
        z: torch.Tensor,
        initial_condition: Callable,
        points: dict,
        n_train: int,
        w0: float,
        E0: float,
        verbose: bool = False
    ):
        self.z = z
        self.initial_condition = initial_condition
        self.points = points
        self.w0 = w0
        self.E0: float = E0
        self.n_train = n_train

    def en_loss(self, pinn):
        x, y, t = self.points['res_points']
        output = f(pinn, x, y, t)

        nx = self.n_train - 2
        ny = nx
        nt = self.n_train - 1

        d_en = calc_den(x, y, output)
        d_en = d_en.reshape(nx, ny, nt)

        x = x.reshape(nx, ny, nt)
        y = y.reshape(nx, ny, nt)
        t = t.reshape(nx, ny, nt)

        en_y = torch.trapezoid(d_en, y, dim=1)
        En = torch.trapezoid(en_y, x, dim=0)

        loss = (self.E0 * torch.ones_like(En) - En).pow(2).mean()

        return loss


    def res_loss(self, pinn):
        x, y, t = self.points['res_points']
        output = f(pinn, x, y, t)

        dvx_t = df(output, [t], 2)
        dvy_t = df(output, [t], 3)

        dux_x = df(output, [x], 0)
        dux_y = df(output, [y], 0)
        duy_x = df(output, [x], 1)
        duy_y = df(output, [y], 1)

        dux_xx = df(dux_x, [x])
        duy_yy = df(duy_y, [y])
        duy_xx = df(duy_x, [x])

        dux_yy = df(dux_y, [y])
        dux_xy = df(dux_x, [y])
        duy_xy = df(duy_x, [y])

        loss_v = []

        loss_v.append(dvx_t - 2*self.z[0]*(dux_xx + 1/2 *
                                      (dux_yy + duy_xy)) - self.z[1]*(dux_xx + duy_xy))
        loss_v.append(dvy_t - 2 * self.z[0]*(1/2*(duy_xx +
                                             dux_xy) + duy_yy) - self.z[1]*(dux_xy + duy_yy))

        loss_v.append(dvx_t - df(output, [t, t], 0))
        loss_v.append(dvy_t - df(output, [t, t], 1))

        vx = output[:, 2].reshape(-1, 1)
        vy = output[:, 3].reshape(-1, 1)

        d_en = (1/2*(vx+vy))**2 + 1/2*(dux_x + duy_y)**2 + \
            (dux_x**2 + duy_y**2 + (1/2*(dux_y+duy_x)) ** 2 +
               (1/2*(duy_x + dux_y)) ** 2)

        d_en_t = torch.autograd.grad(
            d_en,
            t,
            grad_outputs=torch.ones_like(t),
            create_graph=True,
        )[0]

        d_en_t = d_en_t.pow(2).mean()

        loss_v.append(d_en_t)

        loss = 0

        for loss_res in loss_v:
            loss += loss_res.pow(2).mean()

        return (loss, d_en_t)


    def bound_loss(self, pinn):
        down, up, left, right = self.points['boundary_points']
        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        ux_down = f(pinn, x_down, y_down, t_down)[:, 0]
        uy_down = f(pinn, x_down, y_down, t_down)[:, 1]

        ux_left = f(pinn, x_left, y_left, t_left)[:, 0]
        uy_left = f(pinn, x_left, y_left, t_left)[:, 1]
        left = torch.cat([ux_left[..., None], uy_left[..., None]], -1)
        duy_y_left = df(left, [y_left], 1)
        dux_y_left = df(left, [y_left], 0)
        duy_x_left = df(left, [x_left], 1)
        tr_left = df(left, [x_left], 0) + duy_y_left

        ux_right = f(pinn, x_right, y_right, t_right)[:, 0]
        uy_right = f(pinn, x_right, y_right, t_right)[:, 1]
        right = torch.cat([ux_right[..., None], uy_right[..., None]], -1)
        duy_y_right = df(right, [y_right], 1)
        dux_y_right = df(right, [y_right], 0)
        duy_x_right = df(right, [x_right], 1)
        tr_right = df(right, [x_right], 0) + duy_y_right

        loss_v = []

        loss_v.append(2*self.z[0]*(1/2*(dux_y_left + duy_x_left)))
        loss_v.append(2*self.z[0]*duy_y_left + self.z[1]*tr_left)

        loss_v.append(2*self.z[0]*(1/2*(dux_y_right + duy_x_right)))
        loss_v.append(2*self.z[0]*duy_y_right + self.z[1]*tr_right)

        loss = 0

        for loss_bound in loss_v:
            loss += loss_bound.pow(2).mean()

        return loss

    def verbose(self, pinn):
        res_loss, en_crit = self.res_loss(pinn)
        loss = res_loss + self.bound_loss(pinn)

        return (loss, res_loss, self.bound_loss(pinn), en_crit)

    def __call__(self, pinn):
        return self.verbose(pinn)


def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
    points: dict,
    n_train: int
) -> PINN:

    optimizer = optim.Adam(nn_approximator.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=path_logs)

    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss, res_loss, bound_loss, en_crit = loss_fn(nn_approximator)

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': res_loss.item(),
            'boundary': bound_loss.item(),
        }, epoch)

        writer.add_scalar('Energy_cons', en_crit.item(), epoch)

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

def calc_den(x, y, output):

    vx = output[:,2].reshape(-1)
    vy = output[:,3].reshape(-1)

    dux_x = df(output, [x], 0).reshape(-1)
    dux_y = df(output, [y], 0).reshape(-1)
    duy_x = df(output, [x], 1).reshape(-1)
    duy_y = df(output, [y], 1).reshape(-1)

    d_en_k = (1/2*(vx+vy))**2
    d_en_p = 1/2*(dux_x + duy_y)**2 + \
        (dux_x**2 + duy_y**2 + (1/2*(dux_y+duy_x)) ** 2 +
           (1/2*(duy_x + dux_y)**2))

    d_en = d_en_k + d_en_p

    return (d_en, d_en_k, d_en_k)

def calc_initial_energy(pinn_trained: PINN, n: int, points: dict, device):
    x, y, t = points['initial_points']

    output = f(pinn_trained, x, y, t)

    d_en, d_en_k, d_en_p = calc_den(x, y, output)
    d_en = d_en.reshape(n, n)

    x = x.reshape(n, n)
    y = y.reshape(n, n)

    en_x = torch.trapezoid(d_en, y[0,:], dim=1)
    En = torch.trapezoid(en_x, x[:,0], dim=0)

    return En

def calc_energy(pinn_trained: PINN, points: Grid, n_train, device) -> tuple:
    x, y, t = points['all_points']

    x.requires_grad_(True).to(device)
    y.requires_grad_(True).to(device)
    t.requires_grad_(True).to(device)

    output = f(pinn_trained, x, y, t)

    d_en, d_en_k, d_en_p = calc_den(x, y, output)

    x = x.reshape(n_train, n_train, n_train)
    y = y.reshape(n_train, n_train, n_train)
    t = t.reshape(n_train, n_train, n_train)

    d_en_k = d_en_k.reshape(n_train, n_train, n_train)
    d_en_p = d_en_p.reshape(n_train, n_train, n_train)
    d_en = d_en.reshape(n_train, n_train, n_train)

    d_en_k_y = torch.trapz(d_en_k, y[0,:,0], dim=1)
    d_en_p_y = torch.trapz(d_en_p, y[0,:,0], dim=1)
    d_en_y = torch.trapz(d_en, y[0,:,0], dim=1)

    En_k_t = torch.trapz(d_en_k_y, x[:,0,0], dim=0)
    En_p_t = torch.trapz(d_en_p_y, x[:,0,0], dim=0)
    En_t = torch.trapz(d_en_y, x[:,0,0], dim=0)

    t = torch.unique(t)

    return (t, En_k_t, En_p_t, En_t)
