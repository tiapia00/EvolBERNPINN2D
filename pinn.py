from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.init as init


def simps(y, dx, dim=0):
    device = y.device
    n = y.size(dim)
    
    if n < 2:
        raise ValueError("At least two samples are required for integration.")

    shape = list(y.shape)
    del(shape[dim])
    shape = tuple(shape)

    # Initialize integral to zeros
    integral = torch.zeros(shape, device=device)
    
    # If n is odd, we can directly apply Simpson's rule to all points
    if n % 2 == 1:
        odd_sum = torch.sum(y.index_select(dim, torch.arange(1, n-1, 2, device=device)), dim=dim)
        even_sum = torch.sum(y.index_select(dim, torch.arange(2, n-1, 2, device=device)), dim=dim)

        integral += (y.index_select(dim, torch.tensor([0], device=device)) + 
                     4 * odd_sum + 2 * even_sum + 
                     y.index_select(dim, torch.tensor([n-1], device=device)))
        
        integral *= dx / 3

    else:
        odd_sum = torch.sum(y.index_select(dim, torch.arange(1, n-2, 2, device=device)), dim=dim)
        even_sum = torch.sum(y.index_select(dim, torch.arange(2, n-2, 2, device=device)), dim=dim)

        integral += (y.index_select(dim, torch.tensor([0], device=device)).squeeze(dim) + 
                     4 * odd_sum + 2 * even_sum + 
                     y.index_select(dim, torch.tensor([n-2], device=device)).squeeze(dim))
        
        integral *= dx / 3
        
        integral += 0.5 * dx * (y.index_select(dim, torch.tensor([n-2], device=device)).squeeze(dim) + 
                                y.index_select(dim, torch.tensor([n-1], device=device)).squeeze(dim))

    return integral


def initial_conditions(space: torch.Tensor, w0: float, i: float = 1) -> torch.tensor:
    x = space[:,0].unsqueeze(1)
    ux0 = torch.zeros_like(x)
    uy0 = w0 * torch.sin(torch.pi*i*x)
    dotux0 = torch.zeros_like(x)
    dotuy0 = torch.zeros_like(x)
    return torch.cat((ux0, uy0, dotux0, dotuy0), dim=1)


class Grid:
    def __init__(self, x_domain, multx_in, y_domain, t_domain, device):
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.t_domain = t_domain
        self.device = device
        self.multx_in = multx_in
        self.requires_grad = True
        self.grid_init = self.generate_grid_init()
        self.grid_init_hyper = self.generate_grid_init_hyper()
        self.grid_bound = self.generate_grid_bound()

    def generate_grid_init_hyper(self):
        xmax = torch.max(self.x_domain)

        x = torch.linspace(0, xmax, self.multx_in * len(self.x_domain))
        y = self.y_domain
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")

        x_grid = x_grid.reshape(-1, 1)
        y_grid = y_grid.reshape(-1, 1)
        t0 = torch.zeros_like(x_grid)

        grid_init = torch.cat((x_grid, y_grid, t0), dim=1)

        return grid_init

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

    def get_initial_points_hyper(self):
        x_grid = self.grid_init_hyper[:, 0].unsqueeze(1).to(self.device)
        x_grid.requires_grad_(True)

        y_grid = self.grid_init_hyper[:, 1].unsqueeze(1).to(self.device)
        y_grid.requires_grad_(True)

        t0 = self.grid_init_hyper[:, 2].unsqueeze(1).to(self.device)
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
                 hiddendim: int,
                 w0: float, 
                 nhidden: int,
                 act=nn.LeakyReLU(),
                 ):

        super().__init__()
        self.hiddendim = hiddendim
        self.nhidden = nhidden
        self.act = act
        self.w0 = w0

        self.U =  nn.Linear(3, hiddendim)

        self.V = nn.Linear(3, hiddendim)

        init.normal_(self.U.weight, mean=0.0, std=0.1)
        init.normal_(self.V.weight, mean=0.0, std=0.1)

        init.normal_(self.U.bias, mean=0., std=0.1)
        init.normal_(self.V.bias, mean=0., std=0.1)

        for param in self.U.parameters():
            param.requires_grad = False

        for param in self.V.parameters():
            param.requires_grad = False

        self.initlayer = nn.Linear(3, 2*hiddendim, bias=False)
        nn.init.xavier_normal_(self.initlayer.weight)

        self.layers = nn.ModuleList([])
        for _ in range(nhidden):
            self.layers.append(nn.Linear(2*hiddendim, 2*hiddendim, bias=False))
            self.layers.append(act)
            nn.init.xavier_normal_(self.layers[-2].weight)
        
        self.outlayerx = nn.Linear(2*hiddendim, 1, bias=False)
        self.outlayerx.weight.data *= 0

        self.outlayery = nn.Linear(2*hiddendim, 1, bias=False)

    def forward(self, space, t):
        input = torch.cat([space, t], dim=1)
        U = self.U(input)
        U = torch.cat([torch.cos(np.pi * U), torch.sin(np.pi * U)], dim=1)
        
        V = self.V(input)
        V = torch.cat([torch.cos(np.pi * V), torch.sin(np.pi * V)], dim=1)
        
        out = self.initlayer(input)

        for layer in self.layers:
            out = layer(out)
            out = out * U + (1-out) * V

        outNNx = self.outlayerx(out)
        outNNy = self.outlayery(out)

        outNN = torch.cat([outNNx, outNNy], dim=1)

        out = space[:,0].unsqueeze(1) * outNN * (1 - space[:,0].unsqueeze(1))

        return out


class Loss:
    def __init__(
        self,
        points: dict,
        n_space: int,
        n_time: int,
        b: float,
        w0: float,
        steps_int: tuple,
        adim: tuple,
        par: dict,
        part_time: int,
        adapt_in: list,
        device: torch.device,
        verbose: bool = False
    ):
        self.points = points
        self.w0 = w0
        self.n_space = n_space
        self.n_time = n_time
        self.steps = steps_int
        self.device = device
        self.adim = adim
        self.par = par
        self.b = b
        self.lambdas = adapt_in
        self.weights_t = torch.ones(part_time, device=device)

    def res_loss(self, pinn):
        x, y, t = self.points['res_points']
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        ax = torch.autograd.grad(vx, t, torch.ones_like(t, device=self.device),
                create_graph=False, retain_graph=True)[0]
        ay = torch.autograd.grad(vy, t, torch.ones_like(t, device=self.device),
                create_graph=False, retain_graph=True)[0]
        
        dxyux = torch.autograd.grad(output[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dxyuy = torch.autograd.grad(output[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        dxx_xy2ux = torch.autograd.grad(dxyux[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dyx_yy2ux = torch.autograd.grad(dxyux[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]

        dxx_xy2uy = torch.autograd.grad(dxyuy[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dyx_yy2uy = torch.autograd.grad(dxyuy[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        eps = torch.stack([dxyux[:,0], 1/2*(dxyux[:,1]+dxyuy[:,0]), dxyuy[:,1]], dim=1)
        dV = ((self.par['w0']/self.par['Lx'])**2*(self.par['mu']*torch.sum(eps**2, dim=1)) + self.par['lam']/2 * torch.sum(eps, dim=1)**2)

        v = torch.cat([vx, vy], dim=1)
        vnorm = torch.norm(v, dim=1)
        dT = (1/2*(self.par['w0']/self.par['t_ast'])**2*self.par['rho']*vnorm**2)
        dT = torch.max(dV)/torch.max(dT)*dT

        tgrid = torch.unique(t, sorted=True)

        V = torch.zeros(tgrid.shape[0])
        T = torch.zeros_like(V)
        loss = 0
        loss_time = torch.zeros(self.weights_t.shape[0], requires_grad=False)
        tidx_par = torch.zeros(0, device=self.device, dtype=torch.int32, requires_grad=False)
        for i, ts in enumerate(tgrid):
            tidx = torch.nonzero(t.squeeze() == ts).squeeze()
            dVt = dV[tidx].reshape(self.n_space - 2, self.n_space - 2)
            dTt = dT[tidx].reshape(self.n_space - 2, self.n_space - 2)
            tidx_par = torch.cat([tidx_par, tidx], dim=0)

            V[i] = self.b*simps(simps(dVt, self.steps[1]), self.steps[0])
            T[i] = self.b*simps(simps(dTt, self.steps[1]), self.steps[0])

            if loss_time.shape[0] % (i + 1) == 0:
                """
                loss_i = (self.adim[0] * (dxx_xy2ux[:,0] + dyx_yy2ux[:,1]) + self.adim[1] * 
                        (dxx_xy2ux[:,0] + dxx_xy2uy[:,1]) - self.adim[2] * ax.squeeze())[tidx_par].pow(2).mean()
                """
                loss_i = (self.adim[0] * (dxx_xy2uy[:,0] + dyx_yy2uy[:,1]) + self.adim[1] * 
                        (dyx_yy2ux[:,0] + dyx_yy2uy[:,1]) - self.adim[2] * ay.squeeze())[tidx_par].pow(2).mean()
                loss_time[i] = loss_i
                loss += self.weights_t[i].item()*loss_i
                tidx_par = torch.zeros(0, device=self.device, dtype=torch.int32, requires_grad=False)

        loss *= 1/self.weights_t.shape[0]
        loss *= self.lambdas[0].item()

        return loss, V, T, loss_time

    def initial_loss(self, pinn):
        init_points = self.points['initial_points_hyper']
        x, y, t = init_points
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        init = initial_conditions(space, pinn.w0)

        lossx = self.lambdas[1].item() * (output[:,0].unsqueeze(1) - init[:,0].unsqueeze(1)/self.w0).pow(2).mean()
        lossy = self.lambdas[2].item() * (output[:,1].unsqueeze(1) - init[:,1].unsqueeze(1)/self.w0).pow(2).mean()
        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        lossvx = (vx * self.par['w0']/self.par['t_ast'] - init[:,2].unsqueeze(1)).pow(2).mean()
        lossvy = (vy * self.par['w0']/self.par['t_ast'] - init[:,3].unsqueeze(1)).pow(2).mean()
        lossv = self.lambdas[3].item() * (lossvx + lossvy)

        inloss = lossx + lossy + lossv

        return inloss, (lossx, lossy, lossv)

    def verbose(self, pinn):
        res_loss, V, T, loss_time = self.res_loss(pinn)
        enloss = ((V[0] + T[0]) - (V + T)).pow(2).mean()
        in_loss, in_losses = self.initial_loss(pinn)
        loss = res_loss + in_loss

        return loss, res_loss, in_loss, (in_losses, V, T, (V+T).mean(), enloss.detach(), loss_time.detach())

    def __call__(self, pinn):
        return self.verbose(pinn)


def calculate_norm(pinn: PINN):
    total_norm = 0
    for param in pinn.parameters():
        if param.grad is not None:  # Ensure the parameter has gradients
            param_norm = param.grad.data.norm(2)  # Compute the L2 norm for the parameter's gradient
            total_norm += param_norm.item() ** 2  # Sum the squares of the norms
    
    return total_norm

def update_adaptive(loss_fn: Loss, norm: tuple, total: float, alpha: float):
    for i in range(len(norm)):
        if norm[i] == 0:
            continue
        loss_fn.lambdas[i] = alpha * loss_fn.lambdas[i] + (1-alpha) * total/norm[i]

def update_weights_t(weights_t: torch.Tensor, eps: float, loss_time: torch.Tensor):
    for i in torch.arange(weights_t.shape[0]-1):
        sum = torch.sum(weights_t[:i+1]* loss_time[:i+1])
        weights_t[i+1] = torch.exp(-eps*sum)

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
    modeldir: str
) -> PINN:

    writer = SummaryWriter(log_dir=path_logs)

    from plots import plot_energy

    optimizer = optim.Adam(nn_approximator.parameters(), lr = learning_rate)
    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs + 1):
        optimizer.zero_grad()

        loss, res_loss, init_loss, losses = loss_fn(nn_approximator)

        if epoch % 200 == 0 and epoch != 0:
            res_loss.backward(retain_graph=True)
            norm_res = calculate_norm(nn_approximator)
            optimizer.zero_grad()

            norms = []
            for lossinit in losses[0]:
                lossinit.backward(retain_graph=True)
                norms.append(calculate_norm(nn_approximator))
                optimizer.zero_grad()
            
            norms.insert(0, norm_res)
            update_adaptive(loss_fn, norms, loss.detach(), 0.9)

        loss.backward(retain_graph=False)
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': res_loss.item(),
            'init': init_loss.item(),
            'enloss': losses[4].item(),
            'weights_time': torch.mean(loss_fn.weights_t).detach().item()
        }, epoch)

        writer.add_scalars('Energy', {
            'V+T': losses[3].detach().item(),
            'V': losses[1].mean().detach().item(),
            'T': losses[2].mean().detach().item()
        }, epoch)

        writer.add_scalars('Adaptive', {
            'res': loss_fn.lambdas[0],
            'initx': loss_fn.lambdas[1],
            'inity': loss_fn.lambdas[2],
            'initv': loss_fn.lambdas[3]
        }, epoch)

        if epoch % 500 == 0:
            t = loss_fn.points['all_points'][-1].unsqueeze(1)
            t = torch.unique(t, sorted=True)[1:]
            plot_energy(t.detach().cpu().numpy(), losses[1].detach().cpu().numpy(), losses[2].detach().cpu().numpy(), epoch, modeldir) 

        update_weights_t(loss_fn.weights_t, 2, losses[-1])

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    writer.close()

    return nn_approximator

def obtainsolt_u(pinn: PINN, space: torch.Tensor, t: torch.Tensor, nsamples: tuple):
    nx, ny, nt = nsamples
    sol = torch.zeros(nx, ny, nt, 2)
    spaceidx = torch.zeros(nx, ny, nt, 2)
    tsv = torch.unique(t, sorted=True)
    output = pinn(space, t)

    for i in range(len(tsv)):
        idxt = torch.nonzero(t.squeeze() == tsv[i])
        spaceidx[:,:,i,:] = space[idxt].reshape(nx, ny, 2)
        sol[:,:,i,:] = output[idxt,:2].reshape(nx, ny, 2)
    
    spaceexpand = spaceidx[:,:,0,:].unsqueeze(2).expand_as(spaceidx)
    check = torch.all(spaceexpand == spaceidx).item()

    if not check:
        raise ValueError('Extracted space tensors not matching')
    
    sol = sol.reshape(nx*ny, nt, 2)

    return sol.detach().cpu().numpy()

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

def calculate_speed(output: torch.Tensor, t: torch.Tensor, par: dict):
    device = output.device
    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=device),
            create_graph=True, retain_graph=True)[0]
    vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=device),
            create_graph=True, retain_graph=True)[0]
    
    v = par['w0']/par['t_ast']*torch.cat([vx, vy], dim=1)

    return v