from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
import pytz
import datetime
from datetime import date
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from typing import Tuple
import os
from read_write import pass_folder, get_current_time, get_last_modified_file, get_current_time, create_folder_date


def initial_conditions(x: torch.tensor, y: torch.tensor, w0: float, i: float = 1) -> torch.tensor:
    res_ux = torch.zeros_like(x)
    res_uy = w0*torch.sin(torch.pi*i/x[-1]*x)
    return res_ux, res_uy


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
            filtered_tensor = tensor[comparison == False]
            tensor = filtered_tensor

        return filtered_tensor

    def delete_bound(self, tensor) -> torch.tensor:

        for vector in self.grid_bound[4]:
            comparison = torch.all(tensor == vector, dim=1)
            filtered_tensor = tensor[comparison == False]
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
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """

    def __init__(self, dim_hidden: int, n_hidden: int, points: dict, dim_input: int = 3, dim_output: int = 4, act=nn.Sigmoid()):

        super().__init__()
        self.layer_in = RBF(dim_input, dim_hidden, gaussian)
        self.sine = SineActivation()

        self.num_middle = n_hidden - 1
        self.middle_layers = nn.ModuleList()

        for _ in range(self.num_middle):
            middle_layer = nn.Linear(dim_hidden, dim_hidden)
            self.act = act
            self.middle_layers.append(middle_layer)

        self.layer_out = nn.Linear(dim_hidden, dim_output)

        self.weights = nn.ParameterList([])

        for i, (key, value) in enumerate(zip(points.keys(), points.values())):
            if i == len(points)-1:
                self.weights.append(nn.Parameter(torch.tensor([1.])))
            elif i == 1:
                self.weights.append(nn.Parameter(torch.ones(value[0].shape)))
            else:
                self.weights.append(nn.Parameter(torch.ones(value[0].shape)))

    def forward(self, x, y, t):
        x_stack = torch.cat([x, y, t], dim=1)
        out = self.sine(self.layer_in(x_stack))
        logits = self.layer_out(out)
"""
        hard_enc = torch.sin(x*np.pi)
        hard_enc = hard_enc.view(-1, 1)
        hard_enc_both = hard_enc.expand(hard_enc.shape[0], 4)

        out = logits*hard_enc_both"""
        return logits

    def forward_mask(self, idx: int):
        masked_weights = torch.sigmoid(self.weights[idx])
        return masked_weights

    def device(self):
        return next(self.parameters()).device


def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        self.h = torch.max(self.points['res_points'][1]) -\
            torch.min(self.points['res_points'][1])

    def residual_loss(self, pinn):
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

        m = pinn.forward_mask(0)

        loss1 = m*(dvx_t - 2*self.z[0]*(dux_xx + 1/2 *
                   (dux_yy + duy_xy)) - self.z[1]*(dux_xx + duy_xy))
        loss2 = m*(dvy_t - 2 * self.z[0]*(1/2*(duy_xx +
                   dux_xy) + duy_yy) - self.z[1]*(dux_xy + duy_yy))

        loss3 = m*(dvx_t - df(output, [t, t], 0))
        loss4 = m*(dvy_t - df(output, [t, t], 1))

        vx = output[:, 2].reshape(-1, 1)
        vy = output[:, 3].reshape(-1, 1)

        d_en = (1/2*self.h*(vx+vy)).pow(2) + self.z[0]*(dux_x + duy_y)**2 +\
            self.z[1]*2*(dux_x**2 + duy_y**2 + (dux_y+duy_x)
                         ** 2 + (duy_x + dux_y)**2)

        d_en_t = torch.autograd.grad(
            d_en,
            t,
            grad_outputs=torch.ones_like(t),
            create_graph=True,
        )[0]

        return (loss1.pow(2).mean() + loss2.pow(2).mean() + loss3.pow(2).mean() + loss4.pow(2).mean(), d_en_t.pow(2).mean())

    def initial_loss(self, pinn, epochs):
        x, y, t = self.points['initial_points']
        pinn_init_ux, pinn_init_uy = self.initial_condition(
            x, y, x[-1], self.w0)
        output = f(pinn, x, y, t)

        ux = output[:, 0].reshape(-1, 1)
        uy = output[:, 1].reshape(-1, 1)

        vx = output[:, 2].reshape(-1, 1)
        vy = output[:, 3].reshape(-1, 1)

        m = pinn.forward_mask(1)

        loss1 = m*(ux - pinn_init_ux)
        loss2 = m*(uy - pinn_init_uy)

        loss3 = vx
        loss4 = vy

        return (loss1.pow(2).mean() + loss2.pow(2).mean() + loss3.pow(2).mean() + loss4.pow(2).mean())

    def boundary_loss(self, pinn):
        down, up, left, right = self.points['boundary_points']
        x_down, y_down, t_down = down
        x_up, y_up, t_up = up
        x_left, y_left, t_left = left
        x_right, y_right, t_right = right

        ux_down = f(pinn, x_down, y_down, t_down)[:, 0]
        uy_down = f(pinn, x_down, y_down, t_down)[:, 1]

        ux_up = f(pinn, x_up, y_up, t_up)[:, 0]
        uy_up = f(pinn, x_up, y_up, t_up)[:, 1]

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

        # loss_upx = ux_up
        # loss_upy = uy_up

        # loss_downx = ux_down
        # loss_downy = uy_down

        loss_left1 = 2*self.z[0]*(1/2*(dux_y_left + duy_x_left))
        loss_left2 = 2*self.z[0]*duy_y_left + self.z[1]*tr_left

        loss_right1 = 2*self.z[0]*(1/2*(dux_y_right + duy_x_right))
        loss_right2 = 2*self.z[0]*duy_y_right + self.z[1]*tr_right

        return pinn.forward_mask(2)*(  # loss_upx.pow(2).mean() + loss_upy.pow(2).mean() +
            # loss_downx.pow(2).mean() + loss_downy.pow(2).mean() +
            loss_left1.pow(2).mean() + loss_left2.pow(2).mean() +
            loss_right1.pow(2).mean() + loss_right2.pow(2).mean())

    def verbose(self, pinn, epoch):
        residual_loss, en_crit = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn, epoch)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = residual_loss + initial_loss + boundary_loss

        return final_loss, residual_loss, initial_loss, boundary_loss, en_crit

    def __call__(self, pinn, epoch):
        return self.verbose(pinn, epoch)[0]


def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
    points: dict,
    n_train: int
) -> PINN:
    from plots import scatter_penalty_loss2D, scatter_penalty_loss3D

    from plots import scatter_penalty_loss2D, scatter_penalty_loss3D

    optimizer = optim.Adam([
        {'params': nn_approximator.layer_in.parameters()},
        {'params': nn_approximator.layer_out.parameters()},
        {'params': nn_approximator.weights, 'lr': -0.001},
    ], lr=learning_rate)

    writer = SummaryWriter(log_dir=path_logs)

    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        _, residual_loss, initial_loss, boundary_loss, en_crit = loss_fn.verbose(
            nn_approximator, epoch)
        loss: torch.Tensor = loss_fn(nn_approximator, epoch)

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Global loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': residual_loss.item(),
            'initial': initial_loss.item(),
            'boundary': boundary_loss.item(),
        }, epoch)

        writer.add_scalar('Energy_cons', en_crit.item(), epoch)

        if epoch % 100 == 0:
            image_penalty_res = scatter_penalty_loss3D(
                points['res_points'][0], points['res_points'][1], points['res_points'][2], n_train, nn_approximator.weights[0].data)
            image_penalty_in = scatter_penalty_loss2D(
                points['initial_points'][0], points['initial_points'][1], n_train, nn_approximator.weights[1].data)

            writer.add_image('res_penalty', image_penalty_res, epoch)
            writer.add_image('in_penalty', image_penalty_in, epoch)

        writer.add_scalar(
            'bound_penalty', nn_approximator.weights[2].data, epoch)
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
        

        d_en_k = (1/2*loss.h*(vx+vy)).pow(2)
        d_en_p = loss.z[0]*(dux_x + duy_y)**2 +\
            loss.z[1]*2*(dux_x**2 + duy_y**2 + (dux_y+duy_x)
                         ** 2 + (duy_x + dux_y)**2)

        d_en_k = d_en_k.reshape(nx, ny, nt)
        d_en_p = d_en_p.reshape(nx, ny, nt)

        d_en_k = d_en_k[:, :, 0]
        d_en_p = d_en_p[:, :, 0]
        
        y_int = y.reshape(nx, ny, nt)
        y_int = y_int[:, 0, 0]

        x_int = x.reshape(nx, ny, nt)
        x_int = x_int[:, 0, 0]

        I_y_k = torch.trapz(y=d_en_k, x=y_int, dim=1)
        I_y_p = torch.trapz(y=d_en_p, x=y_int, dim=1)

        en_k_t = torch.trapz(y=I_y_k, x=x_int)
        en_p_t = torch.trapz(y=I_y_p, x=x_int)
        
        print(en_k_t)
        print(en_p_t)

        en_k.append(en_k_t)
        en_p.append(en_p_t)

        en.append(en_k_t + en_p_t)

    en_k = torch.stack(en_k)
    en_p = torch.stack(en_p)
    en = torch.stack(en)

    return (t, en_k, en_p, en)
