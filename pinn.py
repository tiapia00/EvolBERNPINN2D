from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from typing import Tuple
import os
from read_write import pass_folder, get_current_time, get_last_modified_file, get_current_time, create_folder_date
import matplotlib.pyplot as plt

def initial_conditions(x: torch.tensor, y : torch.tensor, Lx: float, i: float = 1) -> torch.tensor:
    res_ux = torch.zeros_like(x)
    res_uy = torch.sin(torch.pi*i/x[-1]*x)
    return res_ux, res_uy

def get_initial_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

def get_boundary_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), requires_grad=True):
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
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    t_linspace = torch.linspace(t_domain[0], t_domain[1], n_points)

    x_grid, t_grid = torch.meshgrid(x_linspace, t_linspace, indexing="ij")
    y_grid, _      = torch.meshgrid(y_linspace, t_linspace, indexing="ij")

    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t_grid = t_grid.reshape(-1, 1).to(device)
    t_grid.requires_grad = requires_grad

    x0 = torch.full_like(t_grid, x_domain[0], requires_grad=requires_grad)
    x1 = torch.full_like(t_grid, x_domain[1], requires_grad=requires_grad)
    y0 = torch.full_like(t_grid, y_domain[0], requires_grad=requires_grad)
    y1 = torch.full_like(t_grid, y_domain[1], requires_grad=requires_grad)

    down    = (y_grid, x0,     t_grid)
    up      = (y_grid, x1,     t_grid)
    left    = (y0,     x_grid, t_grid)
    right   = (y1,     x_grid, t_grid)

    return down, up, left, right

def get_interior_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), requires_grad=True):
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=n_points, requires_grad=requires_grad)
    y_raw = torch.linspace(y_domain[0], y_domain[1], steps=n_points, requires_grad=requires_grad)
    t_raw = torch.linspace(t_domain[0], t_domain[1], steps=n_points, requires_grad=requires_grad)
    grids = torch.meshgrid(x_raw, y_raw, t_raw, indexing="ij")

    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    t = grids[2].reshape(-1, 1).to(device)

    return x, y, t

class PINN(nn.Module):
    """Simple neural network accepting two features as input and returning a single output

    In the context of PINNs, the neural network is used as universal function approximator
    to approximate the solution of the differential equation
    """
    def __init__(self, num_hidden: int, dim_hidden: int, dim_input : int = 3, dim_output : int = 4, act=nn.Tanh()):

        super().__init__()
        self.dim_hidden = dim_hidden
        self.layer_in = nn.Linear(dim_input, self.dim_hidden)

        self.num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList()

        for _ in range(self.num_middle):
            middle_layer = nn.Linear(dim_hidden, dim_hidden)
            self.act = act
            self.middle_layers.append(middle_layer)

        self.layer_out = nn.Linear(dim_hidden, dim_output)

        self.weights = nn.Parameter(torch.tensor([1., 3., 1.]))

    def forward(self, x, y, t):
        if x.dim() == 1:
            x_stack = torch.cat([x, y, t], dim=0)
            x_stack = x_stack.reshape(1,-1)
        else:
            x_stack = torch.cat([x, y, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)
        return logits

    def forward_mask(self):
        masked = torch.sigmoid(self.weights)
        return masked

    def device(self):
        return next(self.parameters()).device

def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model
    Internally calling the forward method when calling the class as a function"""
    hard_enc = torch.sin(x*np.pi)
    hard_enc = hard_enc.view(-1, 1)
    hard_enc_both = hard_enc.expand(hard_enc.shape[0], 4)
    return hard_enc_both*pinn(x, y, t)

def df(output: torch.Tensor, inputs: list, var : int) -> torch.Tensor:
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
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
        t_domain: Tuple[float, float],
        n_points: int,
        z: torch.Tensor,
        initial_condition: Callable,
        verbose: bool = False,
    ):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.n_points = n_points
        self.z = z
        self.initial_condition = initial_condition

    def residual_loss(self, pinn):
        x, y, t = get_interior_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        output = f(pinn, x, y, t)

        dvx_t = df(output, [t], 2)
        dvy_t = df(output, [t], 3)

        dux_xx = df(output, [x, x], 0)
        duy_yy = df(output, [y, y], 1)
        duy_xx = df(output, [x, x], 1)

        dux_yy = df(output, [y, y], 0)
        dux_xy = df(output, [x, y], 0)
        duy_xy = df(output, [x, y], 1)

        loss1 = dvx_t - 2*self.z[0]*(dux_xx + 1/2*(dux_yy + duy_xy)) - self.z[1]*(dux_xx + duy_xy)
        loss2 = dvy_t - 2*self.z[0]*(1/2*(duy_xx + dux_xy) + duy_yy) - self.z[1]*(dux_xy + duy_yy)
        loss3 = dvx_t - df(output, [t,t], 0)
        loss4 = dvy_t - df(output, [t,t], 1)

        return (loss1.pow(2).mean() + loss2.pow(2).mean() + loss3.pow(2).mean() + loss4.pow(2).mean())

    def initial_loss(self, pinn, epochs):
        x, y, t = get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        pinn_init_ux, pinn_init_uy = self.initial_condition(x, y, x[-1])
        output = f(pinn, x, y, t)

        if epochs == 0:
            fig = plt.figure()
            plt.scatter(x.cpu().detach().numpy()+pinn_init_ux.cpu().detach().numpy(),
                        y.cpu().detach().numpy()+pinn_init_uy.cpu().detach().numpy())
            plt.savefig('initial_cond.png')

        ux = output[:, 0].reshape(-1,1)
        uy = output[:, 1].reshape(-1,1)

        vx = output[:, 2].reshape(-1,1)
        vy = output[:, 3].reshape(-1,1)

        loss1 = ux - pinn_init_ux
        loss2 = uy - pinn_init_uy

        loss3 = vx
        loss4 = vy

        return (loss1.pow(2).mean() + loss2.pow(2).mean() + loss3.pow(2).mean() + loss4.pow(2).mean())

    def boundary_loss(self, pinn):
        down, up, left, right = get_boundary_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
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

        loss_left1 = 2*self.z[0]*(1/2*(dux_y_left + duy_x_left))
        loss_left2 = 2*self.z[0]*duy_y_left + self.z[1]*tr_left

        loss_right1 = 2*self.z[0]*(1/2*(dux_y_right + duy_x_right))
        loss_right2 = 2*self.z[0]*duy_y_right + self.z[1]*tr_right

        return (loss_left1.pow(2).mean() + loss_left2.pow(2).mean() +
                loss_right1.pow(2).mean() + loss_right2.pow(2).mean())

    def verbose(self, pinn, epoch):
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn, epoch)
        boundary_loss = self.boundary_loss(pinn)

        masked = pinn.forward_mask()

        loss_cat = torch.stack((residual_loss, initial_loss, boundary_loss))
        final_loss = torch.dot(loss_cat, masked)

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn, epoch):
        return self.verbose(pinn, epoch)[0]

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import date
import datetime
import pytz

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
) -> PINN:

    optimizer = optim.Adam([
                        {'params': nn_approximator.layer_in.parameters()},
                        {'params': nn_approximator.middle_layers.parameters()},
                        {'params': nn_approximator.layer_out.parameters()},
                        {'params': nn_approximator.weights, 'lr': -0.01},
                        ], lr=0.001)
    loss_values = []
    loss: torch.Tensor = torch.inf

    writer = SummaryWriter(log_dir=path_logs)

    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):
        grads = []

        optimizer.zero_grad()
        _, residual_loss, initial_loss, boundary_loss = loss_fn.verbose(nn_approximator, epoch)
        loss: torch.Tensor = loss_fn(nn_approximator, epoch)

        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        pbar.set_description(f"Global loss: {loss.item():.2f}")

        writer.add_scalars(f'Loss', {
                                    'global': loss.item(),
                                    'residual': residual_loss.item(),
                                    'initial': initial_loss.item(),
                                    'boundary': boundary_loss.item(),
                                    }, epoch)
        writer.add_scalars(f'Weights', {
                                    'residual': pinn.weights[0].item(),
                                    'initial': pinn.weights[1].item(),
                                    'boundary': pinn.weights[2].item(),
                                    }, epoch)

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    writer.close()

    return nn_approximator, np.array(loss_values)

def return_adim(x_dom : np.ndarray, t_dom:np.ndarray, rho: float, mu : float, lam : float):
    L_ast = x_dom[-1]
    T_ast = t_dom[-1]
    z_1 = T_ast**2/(L_ast*rho)*mu
    z_2 = z_1/mu*lam
    z = np.array([z_1, z_2])
    z = torch.tensor(z)
    return z
