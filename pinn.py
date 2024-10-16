from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
import math
from torch import nn
from torch.func import functional_call, vmap, vjp, jvp, jacrev
import torch.optim as optim
from scipy.integrate import simpson

def fnet_single(params, pinn, x, t):
    return functional_call(pinn, params, (x.unsqueeze(0), t.unsqueeze(0))).squeeze(0)

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, t1, x2, t2, pinn):
    # Compute J(x1, t1)
    jac1 = vmap(jacrev(fnet_single), (None, None, 0, 0))(params, pinn, x1, t1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]  # Flatten if needed

    # Compute J(x2, t2)
    jac2 = vmap(jacrev(fnet_single), (None, None, 0, 0))(params, pinn, x2, t2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]  # Flatten if needed

    # Compute J(x1) @ J(x2).T using einsum for the tensor contraction
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result[:,:,1,1]

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

        integral += (y.index_select(dim, torch.tensor([0], device=device)).squeeze() + 
                     4 * odd_sum + 2 * even_sum + 
                     y.index_select(dim, torch.tensor([n-1], device=device)).squeeze())
        
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
        y = torch.linspace(0, torch.max(self.y_domain), int(self.y_domain.shape[0]/2))
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

    def get_initial_points_hyper(self):
        x_grid = self.grid_init_hyper[:, 0].unsqueeze(1).to(self.device)
        x_grid.requires_grad_(True)

        y_grid = self.grid_init_hyper[:, 1].unsqueeze(1).to(self.device)
        y_grid.requires_grad_(True)

        t0 = self.grid_init_hyper[:, 2].unsqueeze(1).to(self.device)
        t0.requires_grad_(True)

        return (x_grid, y_grid, t0)

    def get_interior_points_train(self, scaley):
        x_raw = self.x_domain[1:-1]
        y_raw = self.y_domain[1:-1]
        y_raw = torch.linspace(0, torch.max(y_raw), x_raw.shape[0] // scaley)
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

    def get_all_points_eval(self):
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

def calculate_fft(signal: np.ndarray, dx: float, x: np.ndarray):
    window = np.hanning(signal.size)
    signal = window * signal
    yf = np.fft.fft(signal)
    freq = np.fft.fftfreq(x.size, d=dx)
    return yf, freq

class PINN(nn.Module):
    def __init__(self,
                 dim_hidden: tuple,
                 w0: float,
                 n_hidden: int,
                 multux: int,
                 multuy: int,
                 penalties: torch.Tensor,
                 device,
                 act = nn.GELU()
                 ):

        super().__init__()

        self.w0 = w0
        n_mode_spacex = dim_hidden[0]
        n_mode_spacey = dim_hidden[1]
        self.penalties = nn.Parameter(penalties)

        self.Bx = torch.randn([2, n_mode_spacex], device=device)
        self.By = torch.randn((2, n_mode_spacey), device=device)
        self.By[1,:] *= 0
        
        self.Btx = torch.randn((1, n_mode_spacex), device=device)
        self.Bty = 0.7 * torch.randn((1, n_mode_spacey), device=device)

        self.hid_space_layers_x = nn.ModuleList()
        hiddimx = multux * 2 * n_mode_spacex
        self.hid_space_layers_x.append(nn.Linear(2*n_mode_spacex, hiddimx))
        for _ in range(n_hidden):
            self.hid_space_layers_x.append(nn.Linear(hiddimx, hiddimx))

        self.hid_space_layers_y = nn.ModuleList()
        hiddimy = multuy * 2 * n_mode_spacey
        self.hid_space_layers_y.append(nn.Linear(2*n_mode_spacey, hiddimy))
        for _ in range(n_hidden):
            self.hid_space_layers_y.append(nn.Linear(hiddimy, hiddimy))
            self.hid_space_layers_y.append(act)

        self.layerxmodes = nn.Linear(hiddimx, n_mode_spacex)
        self.layerymodes = nn.Linear(hiddimy, n_mode_spacey)

        self.outlayerx = nn.Linear(n_mode_spacex, 1)
        self.outlayery = nn.Linear(n_mode_spacey, 1)
        self._initialize_weights()

        self.outlayerx.weight.data *= 0

        for param in self.outlayerx.parameters():
            param.requires_grad_(False)

    def fourier_features(self, input, B):
        x_proj = input @ B
        return torch.cat([torch.sin(np.pi * x_proj),
                torch.cos(np.pi * x_proj)], dim=1)

    def _initialize_weights(self):
        # Initialize all layers with Xavier initialization
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)  # Glorot uniform initialization
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # Initialize bias with zeros

    def forward(self, space, t, use_init: bool = False):
        fourier_space_x = self.fourier_features(space, self.Bx)
        fourier_space_y = self.fourier_features(space, self.By)
        fourier_tx = self.fourier_features(t, self.Btx)
        fourier_ty = self.fourier_features(t, self.Bty)

        x_in = fourier_space_x
        y_in = fourier_space_y
        tx = fourier_tx
        ty = fourier_ty

        for layer in self.hid_space_layers_x:
            x_in = layer(x_in)
            tx = layer(tx)
        
        for layer in self.hid_space_layers_y:
            y_in = layer(y_in)
            ty = layer(ty)
        
        xout = self.layerxmodes(x_in)
        tx = self.layerxmodes(tx)
        yout = self.layerymodes(y_in)
        ty = self.layerymodes(ty)

        xout = xout * tx
        yout = yout * ty

        xout = self.outlayerx(xout)
        yout = self.outlayery(yout)

        out = torch.cat([xout, yout], dim=1)

        out = out * (1 - space[:,0].unsqueeze(1))* (space[:,0].unsqueeze(1))

        if use_init:
            init = initial_conditions(space, self.w0)[:,:2]
            out = t * out + init

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
        scaley: int,
        device: torch.device,
        interpVbeam,
        interpEkbeam,
        t_tild: float,
        verbose: bool = False
    ):
        self.points = points
        self.w0 = w0
        self.n_space = n_space
        self.n_time = n_time
        self.steps = steps_int
        self.device = device
        self.adim = adim
        self.scaley = scaley
        self.par = par
        self.b = b
        self.interpVbeam = interpVbeam
        self.interpEkbeam = interpEkbeam
        self.t_tild = t_tild
        self.V0: float
        self.T0: float

    def res_loss(self, pinn, use_init: bool = False):
        x, y, t = self.points['res_points']
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t, use_init)

        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        """        
        ax = torch.autograd.grad(vx, t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        """
        ay = torch.autograd.grad(vy, t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]

        dxyux = torch.autograd.grad(output[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dxyuy = torch.autograd.grad(output[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        """
        dxx_xy2ux = torch.autograd.grad(dxyux[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dyx_yy2ux = torch.autograd.grad(dxyux[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        """
        dxx_xy2uy = torch.autograd.grad(dxyuy[:,0].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        dyx_yy2uy = torch.autograd.grad(dxyuy[:,1].unsqueeze(1), space, torch.ones(space.shape[0], 1, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        """
        loss = (self.adim[0] * (dxx_xy2ux[:,0] + dyx_yy2ux[:,1]) + self.adim[1] * 
                (dxx_xy2ux[:,0] + dxx_xy2uy[:,1]) - self.adim[2] * ax.squeeze()).pow(2).mean()

        loss += (self.adim[0] * (dxx_xy2uy[:,0] + dyx_yy2uy[:,1]) + self.adim[1] * 
                (dyx_yy2ux[:,0] + dyx_yy2uy[:,1]) - self.adim[2] * ay.squeeze()).pow(2).mean()
        """
        loss = (self.adim[0] * (dxx_xy2uy[:,0] + dyx_yy2uy[:,1]) + self.adim[1] * 
                (dyx_yy2uy[:,1]) - self.adim[2] * ay.squeeze()).pow(2).mean()
        
        loss *= pinn.penalties[0].pow(2)

        eps = torch.stack([dxyux[:,0], 1/2*(dxyux[:,1]+dxyuy[:,0]), dxyuy[:,1]], dim=1)
        dV = ((self.par['w0']/self.par['Lx'])**2*(self.par['mu']*torch.sum(eps**2, dim=1)) + self.par['lam']/2 * torch.sum(eps, dim=1)**2)

        v = torch.cat([vx, vy], dim=1)
        vnorm = torch.norm(v, dim=1)
        dT = (1/2*(self.par['w0']/self.par['t_ast'])**2*self.par['rho']*vnorm**2)
        dT = dT * torch.max(dV)/torch.max(dT)

        tgrid = torch.unique(t, sorted=True)

        V = torch.zeros(tgrid.shape[0])
        T = torch.zeros_like(V)
        for i, ts in enumerate(tgrid):
            tidx = torch.nonzero(t.squeeze() == ts).squeeze()
            dVt = dV[tidx].reshape(self.n_space - 2, (self.n_space - 2) // self.scaley) 
            dTt = dT[tidx].reshape(self.n_space - 2, (self.n_space - 2) // self.scaley) 

            V[i] = self.b*simps(simps(dVt, self.steps[1] * self.scaley, dim=1), self.steps[0])
            T[i] = self.b*simps(simps(dTt, self.steps[1] * self.scaley, dim=1), self.steps[0])

        Vbeam = self.interpVbeam(torch.unique(t).detach().cpu().numpy() * self.t_tild) 
        Ekbeam = self.interpEkbeam(torch.unique(t).detach().cpu().numpy() * self.t_tild) 
        Vbeam *= np.max(V.detach().cpu().numpy())/np.max(Vbeam)
        Ekbeam *= np.max(T.detach().cpu().numpy())/np.max(Ekbeam)

        errV = simpson((V.detach().cpu().numpy() - Vbeam)**2, dx=self.steps[2])/simpson(Vbeam**2, dx=self.steps[2])
        errT = simpson((T.detach().cpu().numpy() - Ekbeam)**2, dx=self.steps[2])/simpson(Ekbeam **2, dx=self.steps[2])
         
        return loss, V, T, errV, errT

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
        sigmayy = self.par['w0']/self.par['Lx'] * (2 * self.adim[0] * eps[:,-1] + ekk)

        loss = sigmayy.pow(2).mean()

        return loss

    def initial_loss(self, pinn):
        init_points = self.points['initial_points_hyper']
        x, y, t = init_points
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        init = initial_conditions(space, pinn.w0)
        losspos = pinn.penalties[1].pow(2) * torch.abs(output[:,1] - init[:,1]).mean()
        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        v = torch.cat([vx, vy], dim=1)

        lossv = pinn.penalties[2].pow(2) * (v * self.par['w0']/self.par['t_ast']- init[:,2:]).pow(2).mean(dim=0).sum()

        loss = losspos + lossv

        return loss, (losspos, lossv)

    def verbose(self, pinn, inc_enloss: bool = False):
        res_loss, V, T, errV, errT = self.res_loss(pinn)
        enloss = pinn.penalties[3].pow(2) * ((V+T)).abs().mean() 
        #enloss += pinn.penalties[4].pow(2)*((self.V0 + self.T0) - (V+T)).pow(2).mean()
        boundloss = self.bound_N_loss(pinn)
        init_loss, init_losses = self.initial_loss(pinn)
        loss = res_loss + init_loss

        if inc_enloss:
            loss += enloss

        losses = {
            "in_losses": init_losses,
            "in_loss": init_loss,
            "bound_loss": boundloss,
            "V": V,
            "T": T,
            "V+T": (V+T).mean(),
            "enloss": enloss,
            "errV": errV,
            "errT": errT
        }

        return loss, res_loss, losses 

    def __call__(self, pinn, inc_enloss = False):
        return self.verbose(pinn, inc_enloss)

def train_model(
    nn_approximator: PINN,
    loss_fn: Callable,
    points: dict,
    learning_rate: int,
    max_epochs: int,
    path_logs: str,
    modeldir: str,
) -> PINN:

    writer = SummaryWriter(log_dir=path_logs)

    from plots import plot_energy

    exclude_params = ['penalties']
    params_to_optimize = [
        {'params': [p for n, p in nn_approximator.named_parameters() if n not in exclude_params], 'lr': learning_rate},
        {'params': [nn_approximator.penalties], 'lr':  -1e-3}
    ]
    optimizer = optim.Adam(params_to_optimize)
    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs + 1):
        optimizer.zero_grad()

        use_en = False 
        loss, res_loss, losses = loss_fn(nn_approximator, use_en)

        pbar.set_description(f"Loss: {loss.item():.3e}")

        loss.backward(retain_graph=False)

        if epoch % 100 == 0:
            params = {k: v.detach() for k, v in nn_approximator.named_parameters()}
            idx_res = torch.randperm(points['res_points'][0].shape[0])[:10]
            res_space = torch.cat(points['res_points'], dim=1)[idx_res,:2].detach()
            res_t = points['res_points'][-1][idx_res, :].detach()
            idx_init = torch.randperm(points['initial_points_hyper'][0].shape[0])[:10]
            init_space = torch.cat(points['initial_points_hyper'], dim=1)[idx_init,:2].detach()
            init_t = points['initial_points_hyper'][-1][idx_init,:].detach()
            ntk = empirical_ntk_jacobian_contraction(fnet_single, params, res_space, res_t, init_space, init_t, nn_approximator)
            trntk = torch.einsum('ii', ntk).item()

        optimizer.step()
        upper_sum = torch.einsum('ij->', torch.triu(ntk, diagonal=1))
        lower_sum = torch.einsum('ij->', torch.tril(ntk, diagonal=-1))
        meantrintk = 1/(ntk.shape[0]**2 - ntk.shape[0]) * (upper_sum + lower_sum)
        """
        l1_norm = sum(p.abs().sum() for p in nn_approximator.parameters())
        loss += lambda_reg * l1_norm
        """

        writer.add_scalars('Loss', {
            'global': loss.item(),
            'residual': res_loss.item(),
            'boundary': losses["bound_loss"].item(),
            'init': losses['in_loss'].item(),
            'enlosses': losses["enloss"].item(),
            'V-V_an': losses["errV"],
            'T-T_an': losses["errT"]
        }, epoch)

        writer.add_scalars('NTK', {
            'tr': trntk,
            'meandiag': ntk.diag().mean(),
            'meantri': meantrintk
        }, epoch)

        writer.add_scalars('Energy', {
            'V+T': losses["V+T"].item(),
            'V': losses["V"].mean().detach().item(),
            'T': losses["T"].mean().detach().item(),
        }, epoch)

        writer.add_scalars('Adaptive', {
            'res': nn_approximator.penalties[0].item(),
            'initpos': nn_approximator.penalties[1].item(),
            'initv': nn_approximator.penalties[2].item(),
            'enlosscons': nn_approximator.penalties[3].item(),
            'enlossinit': nn_approximator.penalties[4].item()
        }, epoch)

        if epoch % 500 == 0:
            t = loss_fn.points['res_points'][-1].unsqueeze(1)
            t = torch.unique(t, sorted=True)
            plot_energy(t.detach().cpu().numpy(), losses["V"].detach().cpu().numpy(), losses["T"].detach().cpu().numpy(), epoch, modeldir) 

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