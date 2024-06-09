from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
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
    def __init__(self, x_domain, y_domain, t_domain, n_points, device):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.n_points = n_points
        self.device = device
        self.requires_grad = True
        self.grid_init = self.generate_grid_init()
        self.grid_bound = self.generate_grid_bound()

    def delete_init(self, tensor) -> torch.tensor:
        """matching_indices: print idx of rows which are identical"""

        for vector in self.grid_init:
            comparison = torch.all(tensor == vector, dim=1)
            filtered_tensor = tensor[~comparison]
            tensor = filtered_tensor

        return filtered_tensor

    def delete_bound(self, tensor) -> torch.tensor:

        for vector in self.grid_bound[4]:
            comparison = torch.all(tensor == vector, dim=1)
            filtered_tensor = tensor[~comparison]
            tensor = filtered_tensor

        return filtered_tensor

    def generate_grid_init(self):
        x_linspace = torch.linspace(
            self.x_domain[0], self.x_domain[1], self.n_points)
        y_linspace = torch.linspace(
            self.y_domain[0], self.y_domain[1], self.n_points)
        x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")

        x_grid = x_grid.reshape(-1, 1)
        y_grid = y_grid.reshape(-1, 1)
        t0 = torch.full_like(
            x_grid, self.t_domain[0])

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

        x_linspace = torch.linspace(
            self.x_domain[0], self.x_domain[1], self.n_points)
        y_linspace = torch.linspace(
            self.y_domain[0], self.y_domain[1], self.n_points)
        t_linspace = torch.linspace(
            self.t_domain[0], self.t_domain[1], self.n_points)

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
        down = self.delete_init(down)

        up = torch.cat((x1, y_grid, t_grid), dim=1)
        up = self.delete_init(up)

        left = torch.cat((x_grid, y0, t_grid), dim=1)
        left = self.delete_init(left)

        right = torch.cat((x_grid, y1, t_grid), dim=1)
        right = self.delete_init(right)

        bound_points = torch.cat((down, up, left, right), dim=0)

        return (down, up, left, right, bound_points)

    def get_initial_points(self):
        x_grid = self.grid_init[:, 0].unsqueeze(1).to(self.device)
        x_grid.requires_grad = True
        y_grid = self.grid_init[:, 1].unsqueeze(1).to(self.device)
        y_grid.requires_grad = True
        t0 = self.grid_init[:, 2].unsqueeze(1).to(self.device)
        t0.requires_grad = True
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
        x_raw = torch.linspace(
            self.x_domain[0], self.x_domain[1], steps=self.n_points)
        y_raw = torch.linspace(
            self.y_domain[0], self.y_domain[1], steps=self.n_points)
        t_raw = torch.linspace(
            self.t_domain[0], self.t_domain[1], steps=self.n_points)
        grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

        x = grids[0].reshape(-1, 1)
        y = grids[1].reshape(-1, 1)
        t = grids[2].reshape(-1, 1)

        grid = torch.cat((x, y, t), dim=1)
        grid = self.delete_init(grid)
        grid = self.delete_bound(grid)

        x = grid[:, 0].unsqueeze(1).to(self.device)
        x.requires_grad = True
        y = grid[:, 1].unsqueeze(1).to(self.device)
        y.requires_grad = True
        t = grid[:, 2].unsqueeze(1).to(self.device)
        t.requires_grad = True

        return (x, y, t)


class SineActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)


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


def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi


class PINN(nn.Module):
    def __init__(self,
                 dim_hidden: tuple,
                 n_hidden: tuple,
                 points: dict,
                 w0: float,
                 initial_conditions: callable,
                 act=nn.Tanh()):

        super().__init__()

        self.w0 = w0

        self.n_hidden = (0, n_hidden[1], n_hidden[2])
        self.in_space = nn.Linear(2, dim_hidden[0])

        time_dim = int(0.5 * dim_hidden[1] + 0.5)
        self.in_time_disp = nn.Linear(1, time_dim)
        self.in_time_speed = nn.Linear(1, time_dim)

        # Assuming RBF is correctly defined or imported
        self.mid_space_layers = RBF(dim_hidden[0], 2, gaussian)

        self.hidden_time_layers = nn.ModuleList()
        for i in range(self.n_hidden[1] - 1):
            self.hidden_time_layers.append(nn.Linear(time_dim, time_dim))
            self.hidden_time_layers.append(act)

        self.mid_time_layer = nn.Linear(time_dim, 2)

        self.pre_out_layer = nn.Linear(4, dim_hidden[2])

        self.pre_out_layers = nn.ModuleList()
        for i in range(self.n_hidden[2] - 1):
            self.pre_out_layers.append(nn.Linear(dim_hidden[2], dim_hidden[2]))
            self.pre_out_layers.append(act)

        self.out_layer = nn.Linear(dim_hidden[2], 4)
        self.initial_points = points['initial_points']

    @staticmethod
    def apply_filter(alpha):
        return (torch.tanh(alpha))

    @staticmethod
    def apply_compl_filter(alpha):
        return (1-torch.tanh(alpha))

    def forward(self, x, y, t):
        space = torch.cat([x, y], dim=1)
        time = t

        x_in = self.in_space(space)
        mid_x = self.mid_space_layers(x_in)

        t_disp = self.in_time_disp(time)
        t_speed = self.in_time_speed(time)

        for layer_t in self.hidden_time_layers:
            t_disp = layer_t(t_disp)
            t_speed = layer_t(t_speed)

        t_disp = self.mid_time_layer(t_disp)
        t_speed = self.mid_time_layer(t_speed)

        mid_t = torch.cat([t_disp, t_speed], dim=1)

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
        w0: float,
        verbose: bool = False
    ):
        self.z = z
        self.initial_condition = initial_condition
        self.points = points
        self.w0 = w0

    def overall_loss(self, pinn):
        x_in, y_in, t_in = self.points['initial_points']
        down, up, left, right = self.points['boundary_points']

        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        x_bound = torch.cat((x_down, x_up,
                             x_left, x_right), dim=0)
        y_bound = torch.cat((y_down, y_up,
                             y_left, y_right), dim=0)
        t_bound = torch.cat((t_down, t_up,
                             t_left, t_right), dim=0)

        x_res, y_res, t_res = self.points['res_points']

        x = torch.cat((x_in, x_bound, x_res), dim=0)
        y = torch.cat((y_in, y_bound, y_res), dim=0)
        t = torch.cat((t_in, t_bound, t_res), dim=0)

        output = f(pinn, x, y, t)

        output_in = output[:x_in.shape[0],:]
        output_bound = output[x_in.shape[0]:x_in.shape[0]+x_bound.shape[0],:]
        output_res = output[x_in.shape[0]+x_bound.shape[0]:, :]

        dvx_t = df(output_res, [t_res], 2)
        dvy_t = df(output_res, [t_res], 3)

        dux_x = df(output_res, [x_res], 0)
        dux_y = df(output_res, [y_res], 0)
        duy_x = df(output_res, [x_res], 1)
        duy_y = df(output_res, [y_res], 1)

        dux_xx = df(dux_x, [x_res])
        duy_yy = df(duy_y, [y_res])
        duy_xx = df(duy_x, [x_res])

        dux_yy = df(dux_y, [y_res])
        dux_xy = df(dux_x, [y_res])
        duy_xy = df(duy_x, [y_res])

        lossres_1 = (dvx_t - 2*self.z[0]*(dux_xx + 1/2 *
                                      (dux_yy + duy_xy)) - self.z[1]*(dux_xx + duy_xy))
        lossres_2 = (dvy_t - 2 * self.z[0]*(1/2*(duy_xx +
                                             dux_xy) + duy_yy) - self.z[1]*(dux_xy + duy_yy))

        lossres_3 = (dvx_t - df(output_res, [t_res, t_res], 0))
        lossres_4 = (dvy_t - df(output_res, [t_res, t_res], 1))

        vx = output_res[:, 2].reshape(-1, 1)
        vy = output_res[:, 3].reshape(-1, 1)

        d_en = (1/2*(vx+vy))**2 + (dux_x + duy_y)**2 + \
            2*(dux_x**2 + duy_y**2 + (dux_y+duy_x) ** 2 +
               (duy_x + dux_y)**2)

        d_en_t = torch.autograd.grad(
            d_en,
            t_res,
            grad_outputs=torch.ones_like(t_res),
            create_graph=True,
        )[0]

        output_down = output_bound[:x_down.shape[0],:]
        output_up = output_bound[x_down.shape[0]:x_down.shape[0]+x_up.shape[0],:]
        output_left = output_bound[x_down.shape[0]+x_up.shape[0]:x_down.shape[0]+x_up.shape[0]+x_left.shape[0],:]
        output_right = output_bound[x_down.shape[0]+x_up.shape[0]+x_left.shape[0]:,:]

        ux_left = output_left[:, 0]
        uy_left = output_left[:, 1]
        left = torch.cat([ux_left[..., None], uy_left[..., None]], -1)
        duy_y_left = df(left, [y_left], 1)
        dux_y_left = df(left, [y_left], 0)
        duy_x_left = df(left, [x_left], 1)
        tr_left = df(left, [x_left], 0) + duy_y_left

        ux_right = output_right[:, 0]
        uy_right = output_right[:, 1]
        right = torch.cat([ux_right[..., None], uy_right[..., None]], -1)
        duy_y_right = df(right, [y_right], 1)
        dux_y_right = df(right, [y_right], 0)
        duy_x_right = df(right, [x_right], 1)
        tr_right = df(right, [x_right], 0) + duy_y_right

        loss_left1 = 2*self.z[0]*(1/2*(dux_y_left + duy_x_left))
        loss_left2 = 2*self.z[0]*duy_y_left + self.z[1]*tr_left

        loss_right1 = 2*self.z[0]*(1/2*(dux_y_right + duy_x_right))
        loss_right2 = 2*self.z[0]*duy_y_right + self.z[1]*tr_right

        ux_down = output_down[:,0]
        uy_down = output_down[:,1]
        vx_down = output_down[:,2]
        vy_down = output_down[:,3]

        ux_up = output_up[:,0]
        uy_up = output_up[:,1]
        vx_up = output_up[:,2]
        vy_up = output_up[:,3]

        lossdown_1 = ux_down
        lossdown_2 = uy_down
        lossdown_3 = vx_down
        lossdown_4 = vy_down

        lossup_1 = ux_up
        lossup_2 = uy_up
        lossup_3 = vx_up
        lossup_4 = vy_up

        return (lossres_1.pow(2).mean() + lossres_2.pow(2).mean() + lossres_3.pow(2).mean() + lossres_4.pow(2).mean() +
                lossdown_1.pow(2).mean() + lossdown_2.pow(2).mean() + lossdown_3.pow(2).mean() + lossdown_4.pow(2).mean() +
                lossup_1.pow(2).mean() + lossup_2.pow(2).mean() + lossup_3.pow(2).mean() + lossup_4.pow(2).mean(),
                d_en_t.pow(2).mean())


    def verbose(self, pinn, epoch):
        overall_loss, en_crit = self.overall_loss(pinn)

        return overall_loss, en_crit

    def __call__(self, pinn, epoch):
        return self.verbose(pinn, epoch)


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
        loss, en_crit = loss_fn(nn_approximator, epoch)

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Global loss: {loss.item():.3e}")

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


def calc_energy(pinn_trained: PINN, loss: Loss, n_train, device) -> tuple:
    x, y, t = loss.points['res_points']
    x.to(device)
    y.to(device)

    t = torch.unique(t)
    nx = n_train-2
    ny = nx
    nt = n_train-1

    en = []
    en_k = []
    en_p = []

    for t_raw in t:
        t_i = t_raw*torch.ones_like(x)
        t_i.to(device)

        output = f(pinn_trained, x, y, t_i)

        vx = output[:, 2]
        vy = output[:, 3]

        dux_x = df(output, [x], 0)
        dux_y = df(output, [y], 0)
        duy_x = df(output, [x], 1)
        duy_y = df(output, [y], 1)

        d_en_k = (1/2*(vx+vy)).pow(2)
        d_en_p = 1/2*((dux_x + duy_y)**2 + 2*(dux_x**2 + duy_y**2 + (dux_y+duy_x)
                                              ** 2 + (duy_x + dux_y)**2))

        d_en_k = d_en_k.reshape(nx, ny, nt)
        d_en_p = d_en_p.reshape(nx, ny, nt)

        d_en_k = d_en_k[:, :, 0]
        d_en_p = d_en_p[:, :, 0]

        y_int = y.reshape(nx, ny, nt).detach()
        y_int = y_int[0, :, 0]

        x_int = x.reshape(nx, ny, nt).detach()
        x_int = x_int[:, 0, 0]

        I_x_k = torch.trapz(y=d_en_k.detach(), x=x_int)
        I_x_p = torch.trapz(y=d_en_p.detach(), x=x_int)

        en_k_t = torch.trapz(y=I_x_k, x=y_int)
        en_p_t = torch.trapz(y=I_x_p, x=y_int)

        en_k.append(en_k_t)
        en_p.append(en_p_t)

        en.append(en_k_t + en_p_t)

    en_k = torch.stack(en_k)
    en_p = torch.stack(en_p)
    en = torch.stack(en)

    return (t, en_k, en_p, en)
