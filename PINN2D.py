from typing import Callable

import numpy as np
import torch
from torch import nn
from typing import Tuple

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_file = "PINN2D.torch"

def initial_conditions(x: torch.tensor, y : torch.tensor, i: float = 1) -> torch.tensor:
    # description of displacements, so i don't have to add anything
    res_ux = torch.zeros_like(res_uy)
    res_uy = torch.sin(i*torch.pi/x[-1]*x)
    return res_ux, res_uy

def get_initial_points(x_domain, y_domain, t_domain, n_points, device = torch.device('cpu'), requires_grad=True):
    x_linspace = torch.linspace(x_domain[0], x_domain[1], n_points)
    y_linspace = torch.linspace(y_domain[0], y_domain[1], n_points)
    x_grid, y_grid = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    x_grid = x_grid.reshape(-1, 1).to(device)
    x_grid.requires_grad = requires_grad
    y_grid = y_grid.reshape(-1, 1).to(device)
    y_grid.requires_grad = requires_grad
    t0 = torch.full_like(x_grid, t_domain[0], requires_grad=requires_grad)
    return (x_grid, y_grid, t0)

def get_boundary_points(x_domain, y_domain, t_domain, n_points, device = torch.device('cpu'), requires_grad=True):
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

def get_interior_points(x_domain, y_domain, t_domain, n_points, device = torch.device("cpu"), requires_grad=True):
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
    def __init__(self, num_hidden: int, dim_hidden: int, act=nn.Tanh()):

        super().__init__()

        self.layer_in = nn.Linear(3, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, 1)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act


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
        z_1 : float,
        z_2 : float,
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
        self.initial_condition = initial_condition
        self.weight_r = weight_r
        self.weight_b = weight_b
        self.weight_i = weight_i
        self.z = [z_1, z_2]

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
        return loss1.pow(2).mean() + loss2.pow(2).mean()

    def initial_loss(self, pinn: PINN):
        x, y, t = get_initial_points(self.x_domain, self.y_domain, self.t_domain, self.n_points, pinn.device())
        pinn_init_ux, pinn_init_uy = self.initial_condition(x, y)
        output = f(pinn, x, y, t)
        ux = output[:, 0]
        uy = output[:, 1]
        loss1 = ux - pinn_init_ux
        loss2 = uy - pinn_init_uy
        return loss1.pow(2).mean() + loss2.pow(2).mean()

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

        duy_y_left = df(uy_left, [y_left], 1)
        dux_y_left = df(ux_left, [y_left], 0)
        duy_x_left = df(uy_left, [x_left], 1)
        tr_left = df(ux_left, [x_left], 0) + duy_y_left

        ux_right = f(pinn, x_right, y_right, t_right)[:,0]
        uy_right = f(pinn, x_right, y_right, t_right)[:,1]

        duy_y_right = df(uy_right, [y_right], 1)
        dux_y_right = df(ux_right, [y_right], 0)
        duy_x_right = df(uy_right, [x_right], 1)
        tr_right = df(ux_right, [x_right], 0) + duy_y_right

        loss_left1  = 2*self.z[0]*(1/2*(dux_y_left+duy_x_left))
        loss_left2 =  2*self.z[0]*duy_y_left+self.z[1]*tr_left

        loss_right1 = 2*self.z[0]*(1/2*(dux_y_right+duy_x_right))
        loss_right2 = 2*self.z[0]*duy_y_right+self.z[1]*tr_right

        return loss_left1.pow(2).mean()  + \
            loss_left2.pow(2).mean()    + \
            loss_right1.pow(2).mean()  + \
            loss_right2.pow(2).mean()  + \
            loss_down1.pow(2).mean()   + \
            loss_down2.pow(2).mean()    + \
            loss_up1.pow(2).mean()      + \
            loss_up2.pow(2).mean()

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


def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int = 0.01,
    max_epochs: int = 1_000
) -> PINN:

    optimizer = torch.optim.Adam(nn_approximator.parameters(), lr=learning_rate)
    loss_values = []
    for epoch in range(max_epochs):

        try:

            loss: torch.Tensor = loss_fn(nn_approximator)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch: {epoch + 1} - Loss: {float(loss):>7f}")

        except KeyboardInterrupt:
            break

    return nn_approximator, np.array(loss_values)


# Parameters model
rho = 0.5
mu = 79300 # MPa
lam = 100000 # MPa

# Parameters NN
epochs = 5000
n_train = 200
dim_hidden = 50
lr = 0.001
om_RESIDUAL = 1.0           
om_INITIAL = 1.0            
om_BOUNDARY = 1.0

Lx = 2
Ly = 0.02
T = 10

x_domain = [0.0, Lx]
y_domain = [0.0, Ly]
t_domain = [0.0, T]


def return_adim(x_dom : np.ndarray, t_dom:np.ndarray, rho: float, mu : float, lam : float):
    L_ast = x_dom[-1]
    T_ast = t_dom[-1]
    z_1 = T_ast**2/(L_ast*rho)*mu
    z_2 = z_1/mu*lam
    z = [z_1, z_2]
    return z

pinn = PINN(2, dim_hidden)

# train the PINN
loss_fn = Loss(
    x_domain,
    y_domain,
    t_domain,
    n_train,
    return_adim(x_domain, t_domain, rho, mu, lam),
    initial_conditions,
    om_RESIDUAL,
    om_INITIAL,
    om_BOUNDARY
)

pinn_trained, loss_values = train_model(
    pinn, loss_fn=loss_fn, learning_rate=lr, max_epochs=epochs)