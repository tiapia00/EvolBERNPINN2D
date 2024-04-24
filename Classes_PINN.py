from typing import Callable

import numpy as np
import torch
from torch import nn
from typing import Tuple
import os

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_last_modified_file(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    if not files:
        return None  # Return None if the folder is empty

    # Initialize variables to keep track of the most recently modified file
    last_modified_file = None
    last_modified_time = 0

    # Iterate over each file in the folder
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # Check if the current path is a file (not a directory)
        if os.path.isfile(file_path):
            # Get the modification time of the file
            file_modified_time = os.path.getmtime(file_path)

            # Compare with the last modified time found
            if file_modified_time > last_modified_time:
                last_modified_time = file_modified_time
                last_modified_file = file_path
                

    return last_modified_file

def initial_conditions(x: torch.tensor, y : torch.tensor, Lx: float, i: float = 1) -> torch.tensor:
    # description of displacements, so i don't have to add anything
    res_ux = torch.zeros_like(x)
    res_uy = torch.sin(torch.pi*i/x[-1]*x)/Lx
    return res_ux, res_uy

def get_initial_points(x_domain, y_domain, t_domain, n_points, device = torch.device(device), requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

def get_boundary_points(x_domain, y_domain, t_domain, n_points, device = torch.device(device), requires_grad=True):
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

def get_interior_points(x_domain, y_domain, t_domain, n_points, device = torch.device(device), requires_grad=True):
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
    def __init__(self, num_hidden: int, dim_hidden: int, dim_input : int = 3, dim_output : int = 2, act=nn.Tanh()):

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

    def forward(self, x, y, t):

        x_stack = torch.cat([x, y, t], dim=1)
        out = self.act(self.layer_in(x_stack))
        for layer in self.middle_layers:
            out = self.act(layer(out))
        logits = self.layer_out(out)

        return logits

    def device(self):
        return next(self.parameters()).device

def f(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute the value of the approximate solution from the NN model
    Internally calling the forward method when calling the class as a function"""
    return pinn(x, y, t)

def df(output: torch.Tensor, inputs: list, var : int) -> torch.Tensor:
    """Compute neural network derivative with respect to input features using PyTorch autograd engine
    var = 0 : dux
    var = 1 : duy
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
    def __init__(
        self,
        x_domain: Tuple[float, float],
        y_domain: Tuple[float, float],
        t_domain: Tuple[float, float],
        n_points: int,
        z : torch.tensor,
        initial_condition: Callable,
        weight_r: float = 1.0,
        weight_b: float = 1.0,
        weight_i: float = 1.0,
        verbose: bool = False,
    ):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.n_points = n_points
        self.z = z
        self.initial_condition = initial_condition
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        

    def residual_loss(self, pinn: PINN):
        x, y, t = get_interior_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        output = f(pinn, x, y, t)
        dux_tt = df(output, [t, t], 0)
        duy_tt = df(output, [t, t], 1)

        dux_xx = df(output, [x, x], 0)
        duy_yy = df(output, [y, y], 1)

        dux_xy = df(output, [x, y], 0)
        duy_xy = df(output, [x, y], 1)

        loss1 = dux_tt - 2*self.z[0]*(dux_xx+1/2*(duy_xy+dux_xy)) - self.z[1]*(dux_xx+duy_xy)
        loss2 = duy_tt - 2*self.z[0]*(1/2*(dux_xy+duy_xy)+duy_yy) - self.z[1]*(dux_xy+duy_yy)
        return self.weight_r*(loss1.pow(2).mean() + loss2.pow(2).mean())

    def initial_loss(self, pinn: PINN):
        x, y, t = get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        pinn_init_ux, pinn_init_uy = self.initial_condition(x, y, x[-1])
        output = f(pinn, x, y, t)
        ux = output[:, 0]
        uy = output[:, 1]
        loss1 = ux - pinn_init_ux
        loss2 = uy - pinn_init_uy
        return self.weight_i*(loss1.pow(2).mean() + loss2.pow(2).mean())

    def boundary_loss(self, pinn: PINN):
        """For now,
            - down, up: Dirichlet conditions
            - left, right : Neumann conditions"""
        # n (normal vector) assumed constant during deformation

        down, up, left, right = get_boundary_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        x_down,  y_down,  t_down    = down
        x_up,    y_up,    t_up      = up
        x_left,  y_left,  t_left    = left
        x_right, y_right, t_right   = right

        # Dirichlet conditions on both functions
        loss_down1 = f(pinn, x_down, y_down, t_down)[:,0]
        loss_down2 = f(pinn, x_down, y_down, t_down)[:,1]
        loss_up1 = f(pinn, x_up, y_up, t_up)[:,0]
        loss_up2 = f(pinn, x_up, y_up, t_up)[:,1]

        ux_left = f(pinn, x_left, y_left, t_left)[:,0]
        uy_left = f(pinn, x_left, y_left, t_left)[:,1]

        left = torch.cat([ux_left[..., None], uy_left[..., None]], -1)
        duy_y_left = df(left, [y_left], 1)
        dux_y_left = df(left, [y_left], 0)
        duy_x_left = df(left, [x_left], 1)
        tr_left = df(left, [x_left], 0) + duy_y_left

        ux_right = f(pinn, x_right, y_right, t_right)[:,0]
        uy_right = f(pinn, x_right, y_right, t_right)[:,1]

        right = torch.cat([ux_right[..., None], uy_right[..., None]], -1)
        duy_y_right = df(right, [y_right], 1)
        dux_y_right = df(right, [y_right], 0)
        duy_x_right = df(right, [x_right], 1)
        tr_right = df(right, [x_right], 0) + duy_y_right

        loss_left1  = 2*self.z[0]*(1/2*(dux_y_left+duy_x_left))
        loss_left2 =  2*self.z[0]*duy_y_left+self.z[1]*tr_left

        loss_right1 = 2*self.z[0]*(1/2*(dux_y_right+duy_x_right))
        loss_right2 = 2*self.z[0]*duy_y_right+self.z[1]*tr_right

        return self.weight_b*(loss_left1.pow(2).mean()  + \
            loss_left2.pow(2).mean()    + \
            loss_right1.pow(2).mean()  + \
            loss_right2.pow(2).mean()  + \
            loss_down1.pow(2).mean()   + \
            loss_down2.pow(2).mean()    + \
            loss_up1.pow(2).mean()      + \
            loss_up2.pow(2).mean())

    def verbose(self, pinn: PINN):
        """
        Returns all parts of the loss function

        Not used during training! Only for checking the results later.
        """
        residual_loss = self.residual_loss(pinn)
        initial_loss = self.initial_loss(pinn)
        boundary_loss = self.boundary_loss(pinn)

        final_loss = \
            self.weight_r * residual_loss + \
            self.weight_i * initial_loss + \
            self.weight_b * boundary_loss

        return final_loss, residual_loss, initial_loss, boundary_loss

    def __call__(self, pinn: PINN):
        """
        Allows you to use instance of this class as if it was a function:

        ```
            >>> loss = Loss(*some_args)
            >>> calculated_loss = loss(pinn)
        ```
        """
        return self.verbose(pinn)[0]

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import date
import datetime
import pytz

def create_folder_date(directory, folder_name):
    folder_path = os.path.join(directory, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        
def get_current_time():
    current_time_utc = datetime.datetime.utcnow()
    target_timezone = pytz.timezone('Europe/Paris')
    return current_time_utc.astimezone(target_timezone)

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int
) -> PINN:
    
    folder_name = date.today().isoformat()
    
    create_folder_date('logs', folder_name)
    
    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    loss: torch.Tensor = torch.inf

    # Logging

    pbar = tqdm(total=max_epochs, desc="Training", position=0)
    log_dir = f'logs/{folder_name}'

    current_time = get_current_time()
    subfolder = '/' + current_time.strftime("%H:%M")
    writer = SummaryWriter(log_dir=log_dir + f'/lr = {learning_rate}, max_e = {max_epochs}, hidden_n = {nn_approximator.dim_hidden}' + subfolder)

    for epoch in range(max_epochs):

        loss: torch.Tensor = loss_fn(nn_approximator)
        _, residual_loss, initial_loss, boundary_loss = loss_fn.verbose(nn_approximator)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        # Log loss
        pbar.set_description(f"Loss: {loss.item():.4f}")

        writer.add_scalar("Global loss", loss.item(), epoch)
        writer.add_scalar("Residual loss", residual_loss.item(), epoch)
        writer.add_scalar("Initial loss", initial_loss.item(), epoch)
        writer.add_scalar("Boundary loss", boundary_loss.item(), epoch)

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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FuncFormatter

def plot_initial_conditions(z: torch.tensor, y: torch.tensor, x:torch.tensor, name : str, n_train : int, from_pinn : bool = 1):
    """ For this function, z is the full tensor with both components"""
    fig, ax = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': '3d'}, figsize=(15, 8))

    nbins = 7
    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()

    if from_pinn:
        ux_raw = z.detach().cpu().numpy()[:, 0]
        uy_raw = z.detach().cpu().numpy()[:, 1]
    else:
        z = z.cpu()
        ux_raw = z.detach().numpy()[:, 0]
        uy_raw = z.detach().numpy()[:, 1]

    X = x_raw.reshape(n_train, n_train)
    Y = y_raw.reshape(n_train, n_train)

    ux = ux_raw.reshape(n_train, n_train)
    uy = uy_raw.reshape(n_train, n_train)

    def format_ticks(value, pos):
        return f'{value:.3f}'

    cmap = 'viridis'
    p1 = ax[0].plot_surface(Y, X, ux, cmap=cmap)
    ax[0].set_xlabel('$\\hat{y}$')
    ax[0].set_ylabel('$\\hat{x}$')
    ax[0].set_title('$\\hat{u}_x$')
    ax[0].view_init(elev=30, azim=45)
    ax[0].xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax[0].yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax[0].zaxis.set_major_locator(MaxNLocator(nbins=nbins-2))
    ax[0].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax[0].yaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax[0].zaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax[0].set_box_aspect([2, 2, 1])

    p2 = ax[1].plot_surface(Y, X, uy, cmap=cmap)
    ax[1].set_xlabel('$\\hat{y}$')
    ax[1].set_ylabel('$\\hat{x}$')
    ax[1].set_title('$\\hat{u}_y$')
    ax[1].view_init(elev=30, azim=45)
    ax[1].xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax[1].yaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax[1].zaxis.set_major_locator(MaxNLocator(nbins=nbins-2))
    ax[1].xaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax[1].yaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax[1].zaxis.set_major_formatter(FuncFormatter(format_ticks))
    ax[1].set_box_aspect([2, 2, 1])

    fig.suptitle(name)

def plot_solution(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, n_train : int, figsize=(12, 8), dpi=100):

    rc('animation', html='jshtml')

    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': '3d'}, figsize=(15, 8))
    x_raw = x.reshape(n_train, n_train, n_train)
    y_raw = y.reshape(n_train, n_train, n_train)
    t_raw = torch.unique(t)

    def animate(i):

        if not i % 10 == 0:
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f_final = f(pinn, x_raw, y_raw, t_partial)
            ax.clear()
            ax.plot(
                x_raw.detach().numpy()[:,:,i], y_raw.detach().numpy()[:,:,i], f_final.detach().numpy()[:,0], label=f"Time {float(t[i])}"
            )
            ax.legend()

    n_frames = t_raw.shape[0]
    return animation.FuncAnimation(fig, animate, frames=n_frames, blit=False, repeat=True)

