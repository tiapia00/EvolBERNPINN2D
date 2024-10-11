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

        down = torch.cat((x0, y_grid, t_grid), dim=1).to(self.device)
        down.requires_grad_(True)
        up = torch.cat((x1, y_grid, t_grid), dim=1).to(self.device)
        up.requires_grad_(True)
        left = torch.cat((x_grid, y0, t_grid), dim=1).to(self.device)
        left.requires_grad_(True)
        right = torch.cat((x_grid, y1, t_grid), dim=1).to(self.device)
        right.requires_grad_(True)
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
                 dim_hidden: tuple,
                 w0: float,
                 n_hidden: int,
                 multux: int,
                 multuy: int,
                 device,
                 act=nn.Tanh(),
                 ):

        super().__init__()

        self.w0 = w0
        n_mode_spacex = dim_hidden[0]
        n_mode_spacey = dim_hidden[1]

        self.Bx = torch.randn([2, n_mode_spacex], device=device)
        self.By = torch.randn((2, n_mode_spacey), device=device)
        self.By[0,:] = torch.ones(n_mode_spacey, device=device)
        self.By[1,:] *= 0.01
        
        self.Btx = torch.randn((1, multux*n_mode_spacex), device=device)
        self.Bty = 2 * torch.ones((1, multuy*n_mode_spacey), device=device)

        self.hid_space_layers_x = nn.ModuleList()
        hiddimx = multux * 2 * n_mode_spacex
        self.hid_space_layers_x.append(nn.Linear(2*n_mode_spacex, hiddimx))
        for _ in range(n_hidden):
            self.hid_space_layers_x.append(nn.Linear(hiddimx, hiddimx))

        self.hid_space_layers_y = nn.ModuleList()
        hiddimy = multuy * 2 * n_mode_spacey
        self.hid_space_layers_y.append(nn.Linear(2*n_mode_spacex, hiddimy))
        for _ in range(n_hidden):
            self.hid_space_layers_y.append(nn.Linear(hiddimy, hiddimy))

        self.layerxmodes = nn.Linear(hiddimx, n_mode_spacex)
        self.layerymodes = nn.Linear(hiddimy, n_mode_spacey)
        self._initialize_weights()
        self.outlayerx = nn.Linear(n_mode_spacex, 1, bias=False)
        self.outlayerx.weight.data *= 0
        self.outlayery = nn.Linear(n_mode_spacey, 1, bias=False)

    def fourier_features(self, input, B):
        x_proj = input @ B
        return torch.cat([torch.sin(np.pi * x_proj),
                torch.cos(np.pi * x_proj)], dim=1)

    def _initialize_weights(self):
        # Initialize all layers with Xavier initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Glorot uniform initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize bias with zeros

    def forward(self, space, t):
        fourier_space_x = self.fourier_features(space, self.Bx)
        fourier_space_y = self.fourier_features(space, self.By)
        fourier_tx = self.fourier_features(t, self.Btx)
        fourier_ty = self.fourier_features(t, self.Bty)

        x_in = fourier_space_x
        y_in = fourier_space_y
        tx = fourier_tx
        ty = fourier_ty

        x_in = x_in * tx

        y_in = y_in * ty

        xout = self.layerxmodes(x_in)
        yout = self.layerymodes(y_in)

        xout = self.outlayerx(xout)
        yout = self.outlayery(yout)

        out = torch.cat([xout, yout], dim=1)

        out = space[:,0].unsqueeze(1) * (1 - space[:,0].unsqueeze(1)) * out

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
        in_penalty: np.ndarray,
        adim: tuple,
        par: dict,
        device: torch.device,
        interpVbeam,
        interpTbeam
    ):
        self.points = points
        self.w0 = w0
        self.n_space = n_space
        self.n_time = n_time
        self.steps = steps_int
        self.penalty = in_penalty
        self.device = device
        self.adim = adim
        self.par = par
        self.b = b
        self.interpV = interpVbeam
        self.interpT = interpTbeam

    def res_loss(self, pinn):
        x, y, t = self.points['all_points']
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        ax = torch.autograd.grad(vx, t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        ay = torch.autograd.grad(vy, t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
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
        
        loss = ((dxx_xy2ux[:,0] + dyx_yy2ux[:,1]) + self.adim[1] * 
                (dxx_xy2ux[:,0] + dxx_xy2uy[:,1]) - self.adim[2] * ax.squeeze()).pow(2).mean()

        loss += (self.adim[0] * (dxx_xy2uy[:,0] + dyx_yy2uy[:,1]) + self.adim[1] * 
                (dyx_yy2ux[:,0] + dyx_yy2uy[:,1]) - self.adim[2] * ay.squeeze()).pow(2).mean()
        
        loss *= self.penalty[0].item()
        
        eps = torch.stack([dxyux[:,0], 1/2*(dxyux[:,1]+dxyuy[:,0]), dxyuy[:,1]], dim=1)
        dV = ((self.par['w0']/self.par['Lx'])**2*(self.par['mu']*torch.sum(eps**2, dim=1)) + self.par['lam']/2 * torch.sum(eps, dim=1)**2).detach()

        v = torch.cat([vx, vy], dim=1)
        vnorm = torch.norm(v, dim=1)
        dT = (1/2*(self.par['w0']/self.par['t_ast'])**2*self.par['rho']*vnorm**2).detach()
        dT *= torch.max(dV)/torch.max(dT)

        tgrid = torch.unique(t, sorted=True)

        V = torch.zeros(tgrid.shape[0])
        T = torch.zeros_like(V)
        for i, ts in enumerate(tgrid):
            tidx = torch.nonzero(t.squeeze() == ts).squeeze()
            dVt = dV[tidx].reshape(self.n_space, self.n_space)
            dTt = dT[tidx].reshape(self.n_space, self.n_space)

            V[i] = self.b*simps(simps(dVt, self.steps[1]), self.steps[0])
            T[i] = self.b*simps(simps(dTt, self.steps[1]), self.steps[0])

        t_tild = self.par['t_ast']
        Vbeam = self.interpV(torch.unique(t).detach().cpu().numpy() * t_tild) 
        Ekbeam = self.interpT(torch.unique(t).detach().cpu().numpy() * t_tild) 
        Vbeam *= np.max(V.detach().cpu().numpy())/np.max(Vbeam)
        Ekbeam *= np.max(T.detach().cpu().numpy())/np.max(Ekbeam)

        errV = (((V.detach().cpu().numpy() - Vbeam)/Vbeam)**2).mean()
        errT = (((T.detach().cpu().numpy() - Ekbeam))/Ekbeam**2).mean()

        return loss, V, T, errV, errT

    def initial_loss(self, pinn):
        init_points = self.points['initial_points']
        x, y, t = init_points
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        init = initial_conditions(space, self.w0)

        lossx = self.penalty[1].item() * (output[:,0].unsqueeze(1) - init[:,0].unsqueeze(1)/self.w0).pow(2).mean()
        lossy = self.penalty[2].item() * (output[:,1].unsqueeze(1) - init[:,1].unsqueeze(1)/self.w0).pow(2).mean()
        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        lossvx = (vx * self.par['w0']/self.par['t_ast'] - init[:,2].unsqueeze(1)).pow(2).mean()
        lossvy = (vy * self.par['w0']/self.par['t_ast'] - init[:,3].unsqueeze(1)).pow(2).mean()
        lossv = self.penalty[3].item() * (lossvx + lossvy)

        inloss = lossx + lossy + lossv

        return inloss, (lossx, lossy, lossv)

    def bound_N_loss(self, pinn):
        _, _, left, right, _ = self.points['boundary_points']
        
        neumann = torch.cat([left, right], dim=0)
        space = neumann[:,:2]
        time = neumann[:,1].unsqueeze(1)

        output = pinn(space, time)

        dxyux = torch.autograd.grad(output[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dxyuy = torch.autograd.grad(output[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]

        eps = torch.stack([dxyux[:,0], 1/2*(dxyux[:,1]+dxyuy[:,0]), dxyuy[:,1]], dim=1)
        ekk = torch.sum(eps[:,[0,-1]])
        sigmayy = self.par['w0']/self.par['Lx'] * (2 * self.par['mu'] * eps[:,-1] + self.par['lam'] * ekk)

        loss = sigmayy.pow(2).mean()

        return loss

    def update_penalty(self, max_grad: float, mean: list, alpha: float = 0.4):
        lambda_o = np.array(self.penalty)
        mean = np.array(mean)
        
        lambda_n = max_grad / (lambda_o * (np.abs(mean)))

        self.penalty = (1-alpha) * lambda_o + alpha * lambda_n

    def verbose(self, pinn):
        res_loss, V, T, errV, errT = self.res_loss(pinn)
        enloss = self.penalty[4].item() * ((V[0] + T[0]) - (V + T)).detach().pow(2).mean()
        bound_loss = self.bound_N_loss(pinn)
        in_loss, in_losses = self.initial_loss(pinn)
        loss = res_loss + in_loss

        dict = {
            "in_losses": in_losses,
            "in_loss": in_loss,
            "bound_loss": bound_loss,
            "V": V,
            "T": T,
            "V+T": (V+T).mean(),
            "enloss": enloss.detach(),
            "errV": errV,
            "errT": errT
        }

        return loss, res_loss, dict 

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
        loss_fn.adaptive[i] = alpha * loss_fn.adaptive[i] + (1-alpha) * total/norm[i]

def train_model(
    pinn: PINN,
    loss_fn: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
    modeldir: str
) -> PINN:

    writer = SummaryWriter(log_dir=path_logs)

    from plots import plot_energy

    optimizer = optim.Adam(pinn.parameters(), lr=learning_rate)
    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs + 1):
        optimizer.zero_grad()

        loss, res_loss, losses = loss_fn(pinn)

        if epoch % 400 == 0 and epoch != 0:
            res_loss.backward(retain_graph=True)
            norm_res = calculate_norm(pinn)
            optimizer.zero_grad()

            norms = []
            for lossinit in losses["in_losses"]:
                lossinit.backward(retain_graph=True)
                norms.append(calculate_norm(pinn))
                optimizer.zero_grad()
            
            norms.insert(0, norm_res)
            update_adaptive(loss_fn, norms, loss.detach(), 0.9)

        loss.backward(retain_graph=False)
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': res_loss.item(),
            'boundary': losses["bound_loss"].item(),
            'init': losses['in_loss'].item(),
            'enloss': losses["enloss"].item(),
            'V-V_an': losses["errV"],
            'T-T_an': losses["errT"]
        }, epoch)

        writer.add_scalars('Energy', {
            'V+T': losses["V+T"].item(),
            'V': losses["V"].mean().item(),
            'T': losses["T"].mean().item()
        }, epoch)

        writer.add_scalars('Adaptive', {
            'res': loss_fn.penalty[0].item(),
            'initx': loss_fn.penalty[1].item(),
            'inity': loss_fn.penalty[2].item(),
            'initv': loss_fn.penalty[3].item()
        }, epoch)

        if epoch % 500 == 0:
            t = loss_fn.points['all_points'][-1].unsqueeze(1)
            t = torch.unique(t, sorted=True)
            plot_energy(t.detach().cpu().numpy(), losses["V"].detach().cpu().numpy(), losses["T"].detach().cpu().numpy(), epoch, modeldir) 

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    writer.close()

    return pinn 

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