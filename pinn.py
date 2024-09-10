from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.optim as optim


def simps(y, dx, dim=0):
    device = y.device
    n = y.size(dim)
    if n % 2 == 0:
        raise ValueError(
            "The number of samples must be odd for Simpson's rule.")

    shape = list(y.shape)
    del(shape[dim])
    shape = tuple(shape)

    integral = torch.zeros(shape, device=device)
    odd_sum = torch.sum(y.index_select(dim, torch.arange(1, n-1, 2, device=device)), dim=dim)
    even_sum = torch.sum(y.index_select(dim, torch.arange(2, n-1, 2, device=device)), dim=dim)

    integral += 4 * odd_sum + 2 * even_sum

    integral *= dx / 3

    return integral


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
        #self.grid_bound = self.generate_grid_bound()

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

        down = torch.cat((x0, y_grid, t_grid), dim=1).to(self.device)

        up = torch.cat((x1, y_grid, t_grid), dim=1).to(self.device)

        left = torch.cat((x_grid, y0, t_grid), dim=1).to(self.device)

        right = torch.cat((x_grid, y1, t_grid), dim=1).to(self.device)
        bound_points = torch.cat((down, up, left, right), dim=0).to(self.device)

        return (down, up, left, right, bound_points)

    def get_initial_points(self):
        x_grid = self.grid_init[:, 0].unsqueeze(1).to(self.device)
        x_grid.requires_grad_(True)

        y_grid = self.grid_init[:, 1].unsqueeze(1).to(self.device)
        y_grid.requires_grad_(True)

        t0 = self.grid_init[:, 2].unsqueeze(1).to(self.device)
        t0.requires_grad_(True)

        return (x_grid, y_grid, t0)

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

class PINN(nn.Module):
    def __init__(self,
                 hiddendim: int,
                 nhidden: int,
                 adim_NN: tuple,
                 act=nn.Tanh(),
                 ):

        super().__init__()
        self.hiddendim = hiddendim
        self.nhidden = nhidden
        self.adim = adim_NN
        self.act = act

        self.U = nn.ModuleList([
            nn.Linear(3, hiddendim),
            act
        ])

        self.V = nn.ModuleList([
            nn.Linear(3, hiddendim),
            act
        ])

        self.initlayer = nn.Linear(3, hiddendim)
        self.layers = nn.ModuleList([])
        
        for _ in range(nhidden):
            self.layers.append(nn.Linear(hiddendim, hiddendim))
        
        self.outlayer = nn.Linear(hiddendim, 5)

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, space, t):
        input = torch.cat([space, t], dim=1)
        input0 = input
        for layer in self.U:
            U = layer(input)
            input = U
        
        input = input0
        for layer in self.V:
            V = layer(input)
            input = V
        
        input = input0
        out = self.initlayer(input)

        for layer in self.layers:
            out = layer(out)
            out = self.act(out) * U + (1-self.act(out)) * V

        out = self.outlayer(out)

        if not self.training:
            out[:,:2] *= self.adim[0]
            out[:,2:] *= self.adim[1]

        return out

    def device(self):
        return next(self.parameters()).device


class Loss:
    def __init__(
        self,
        points: dict,
        n_space: int,
        n_time: int,
        w0: float,
        steps_int: tuple,
        in_penalty: np.ndarray,
        adim: tuple,
        verbose: bool = False
    ):
        self.points = points
        self.w0 = w0
        self.n_space = n_space
        self.n_time = n_time
        self.steps = steps_int
        self.penalty = in_penalty
        self.adim = adim
        self.device: torch.device

    def res_loss(self, pinn):
        x, y, t = self.points['res_points']
        space = torch.cat([x,y], dim=1)
        output = pinn(space, t) 
        self.device = x.device
        loss = 0

        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        ax = torch.autograd.grad(vx, t, torch.ones_like(t, device=self.device),
                create_graph=False, retain_graph=True)[0]        
        ay = torch.autograd.grad(vy, t, torch.ones_like(t, device=self.device),
                create_graph=False, retain_graph=True)[0]        
        
        dxsigxx = torch.autograd.grad(output[:,2].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=self.device),
                create_graph=False, retain_graph=True)[0][:,0]
        dxysigxy = torch.autograd.grad(output[:,3].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=self.device),
                create_graph=False, retain_graph=True)[0]
        dysigyy = torch.autograd.grad(output[:,4].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=self.device),
                create_graph=False, retain_graph=True)[0][:,1]
        
        loss += (self.adim[0]*(dxsigxx + dxysigxy[:,1]) - ax.squeeze()).pow(2).mean()
        loss += (self.adim[0]*(dysigyy + dxysigxy[:,0]) - ay.squeeze()).pow(2).mean()

        dxyux = torch.autograd.grad(output[:,0].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=self.device),
                create_graph=False, retain_graph=True)[0]
        dxyuy = torch.autograd.grad(output[:,1].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=self.device),
                create_graph=False, retain_graph=True)[0]

        loss += (output[:,2] - self.adim[1]*dxyux[:,0] - self.adim[2]*dxyuy[:,1]).pow(2).mean()
        loss += (output[:,4] - self.adim[2]*dxyux[:,0] - self.adim[1]*dxyuy[:,1]).pow(2).mean()
        loss += (output[:,3] - self.adim[3]*(dxyux[:,1] - dxyuy[:,0])).pow(2).mean()

        return loss 


    def initial_loss(self, pinn):
        init_points = self.points['initial_points']
        x, y, t = init_points
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        init = initial_conditions(init_points, self.w0)
        init[:,:2] *= 1/self.adim[4]

        m = self.penalty[0]
        loss = 0 
        u = output[:,:2]
        loss += (u - init[:,:2]).pow(2).mean(dim=0).sum()

        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        v = torch.cat([vx, vy], dim=1)

        loss += (v - init[:,-2:]).pow(2).mean(dim=0).sum()
        loss *= m

        return loss


    def bound_loss(self, pinn):
        pass


    def en_loss(self, pinn):
        x, y, t = self.points['all_points']
        space = torch.cat([x,y], dim=1)

        output = pinn(space, t)

        n_space = self.n_space
        n_time = self.n_time

        d_en, d_en_p, d_en_k = calc_den(space, t, output)

        d_en = d_en.reshape(n_space, n_space, n_time)
        d_en_p = d_en_p.reshape(n_space, n_space, n_time)
        d_en_k = d_en_k.reshape(n_space, n_space, n_time)

        x = space[:,0].reshape(n_space, n_space, n_time)
        y = space[:,1].reshape(n_space, n_space, n_time)
        t = t.reshape(n_space, n_space, n_time)

        dx = self.steps[0]
        dy = self.steps[1]
        dt = self.steps[2]

        d_en_y = simps(y=d_en, dx=dy, dim=1)
        d_en_p_y = simps(y=d_en_p, dx=dy, dim=1)
        d_en_k_y = simps(y=d_en_k, dx=dy, dim=1)

        En = simps(y=d_en_y, dx=dx, dim=0)
        En_p = simps(y=d_en_p_y, dx=dx, dim=0)
        En_k = simps(y=d_en_k_y, dx=dx, dim=0)

        m = self.penalty[-1]
        loss = m * (self.E0 * torch.ones_like(En) - En).pow(2).mean()

        return loss


    def update_penalty(self, max_grad: float, mean: list, alpha: float = 0.4):
        lambda_o = np.array(self.penalty)
        mean = np.array(mean)
        
        lambda_n = max_grad / (lambda_o * (np.abs(mean)))

        self.penalty = (1-alpha) * lambda_o + alpha * lambda_n


    def verbose(self, pinn):
        res_loss = self.res_loss(pinn)
        #en_dev = self.en_loss(pinn)
        init_loss = self.initial_loss(pinn)
        #bound_loss = self.bound_loss(pinn)
        loss = res_loss + init_loss

        return loss, res_loss, (init_loss,)

    def __call__(self, pinn):
        return self.verbose(pinn)


def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
) -> PINN:

    writer = SummaryWriter(log_dir=path_logs)

    optimizer = optim.Adam(nn_approximator.parameters(), lr = learning_rate)
    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):
        optimizer.zero_grad()

        if epoch != 0 and epoch % 300 == 0 :
            _, res_loss, losses = loss_fn(nn_approximator)

            res_loss.backward()
            max_grad = get_max_grad(nn_approximator)
            optimizer.zero_grad()

            means = []

            i = 0
            for loss in losses:
                loss.backward()
                if i != 1:
                    means.append(get_mean_grad(nn_approximator))
                optimizer.zero_grad()
                i += 1

            loss_fn.update_penalty(max_grad, means)

        loss, res_loss, losses = loss_fn(nn_approximator)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': res_loss.item(),
            'init': losses[0].item()
            #'boundary': losses[1].item(),
            #'en_dev': losses[2].item()
        }, epoch)

        writer.add_scalars('Penalty_terms', {
            'init': loss_fn.penalty[0].item(),
            'en_dev': loss_fn.penalty[1].item()
        }, epoch)

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    writer.close()

    return nn_approximator


def calc_den(space, t, output):
    device = space.device
    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=device),
                    create_graph=True, retain_graph=True)[0]
    vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=device),
            create_graph=True, retain_graph=True)[0]
    dxyux = torch.autograd.grad(output[:,0].unsqueeze(1), space, torch.ones_like(space, device=device),
                create_graph=True, retain_graph=False)[0]
    dxyuy = torch.autograd.grad(output[:,1].unsqueeze(1), space, torch.ones_like(space, device=device),
                create_graph=True, retain_graph=False)[0]
    dxux = dxyux[:,0]
    dyux = dxyux[:,1]
    dxuy = dxyuy[:,0]
    dyuy = dxyuy[:,1]

    d_en_k = (1/2*(vx+vy))**2
    d_en_p = 1/2*(dxux + dyuy)**2 + \
        (dxux**2 + dyuy**2 + (1/2*(dyux+dxuy)) ** 2 +
           (1/2*(dxuy + dyux)**2))

    d_en = d_en_k + d_en_p

    return (d_en, d_en_p, d_en_k)

def calc_initial_energy(pinn: PINN, n_space: int, points: dict, device):
    x, y, t = points['initial_points']
    space = torch.cat([x,y], dim=1)

    output = pinn(space, t)

    d_en, d_en_k, d_en_p = calc_den(space, t, output)
    d_en = d_en.reshape(n_space, n_space)

    x = space[:,0].reshape(n_space, n_space)
    y = space[:,1].reshape(n_space, n_space)

    en_x = torch.trapezoid(d_en, y[0,:], dim=1)
    En = torch.trapezoid(en_x, x[:,0], dim=0)
    En = En.detach()

    return En

def calc_energy(pinn_trained: PINN, points: dict, n_space: int, n_time: int, dx: float, dy: float) -> tuple:
    x, y, t = points['all_points']

    output = f(pinn_trained, x, y, t)

    d_en, d_en_p, d_en_k = calc_den(x, y, t, output)

    x = x.reshape(n_space, n_space, n_time)
    y = y.reshape(n_space, n_space, n_time)
    t = t.reshape(n_space, n_space, n_time)

    d_en_k = d_en_k.reshape(n_space, n_space, n_time)
    d_en_p = d_en_p.reshape(n_space, n_space, n_time)
    d_en = d_en.reshape(n_space, n_space, n_time)

    d_en_k_y = simps(d_en_k, dy, dim=1)
    d_en_p_y = simps(d_en_p, dy, dim=1)
    d_en_y = simps(d_en, dy, dim=1)

    En_k_t = simps(d_en_k_y, dx, dim=0)
    En_p_t = simps(d_en_p_y, dx, dim=0)
    En_t = simps(d_en_y, dx, dim=0)

    t = torch.unique(t)

    return (t, En_t, En_p_t, En_k_t)


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


def obtainsolt(pinn: PINN, space_in: torch.tensor, t:torch.tensor, nsamples: tuple, device):
    nx, ny, nt = nsamples
    sol = torch.zeros(nx, ny, nt, 2)
    ts = torch.unique(t, sorted=True)
    ts = ts.reshape(-1,1)

    for i in range(ts.shape[0]):
        t = ts[i]*torch.ones(space_in.shape[0], 1, device=device)
        output = pinn(space_in, t)[:,:2]
        gridoutput = output.reshape(nx, ny, 2)
        sol[:,:,i,:] = gridoutput
    
    sol = sol.reshape(nx*ny, -1, 2)
    return sol.detach().cpu().numpy()





