from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.func import functional_call, vmap, jacrev
import torch.optim as optim
import matplotlib.colors as mcolors
from scipy.integrate import simpson
from scipy.stats import kurtosis, skew

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
                 hiddendim: tuple,
                 nhidden: int,
                 n_space: int,
                 n_time: int,
                 scaley: int,
                 act = nn.Tanh()
                 ):

        super().__init__()
        self.hiddendim = hiddendim
        self.res_penalties = nn.Parameter(torch.ones(n_space - 2, (n_space - 2) // scaley, n_time - 1))
        self.in_penalties_p = nn.Parameter(torch.ones(n_space, n_space // scaley))
        self.in_penalties_v = nn.Parameter(torch.ones(n_space, n_space // scaley))
        self.nhidden = nhidden
        self.act = act

        self.U =  nn.Linear(3, hiddendim)

        self.V = nn.Linear(3, hiddendim)

        nn.init.normal_(self.U.weight, mean=0.0, std=0.7)
        nn.init.normal_(self.V.weight, mean=0.0, std=0.7)

        nn.init.normal_(self.U.bias, mean=0., std=0.5)
        nn.init.normal_(self.V.bias, mean=0., std=0.5)

        for param in self.U.parameters():
            param.requires_grad = False

        for param in self.V.parameters():
            param.requires_grad = False

        self.initlayer = nn.Linear(3, hiddendim)
        nn.init.xavier_normal_(self.initlayer.weight)
        self.outlayer = nn.Linear(hiddendim, 2)

        self.layers = nn.ModuleList([])
        for _ in range(nhidden):
            self.layers.append(nn.Linear(hiddendim, hiddendim))
            self.layers.append(act)
            nn.init.xavier_normal_(self.layers[-2].weight)
        
    def forward(self, space, t):
        input = torch.cat([space, t], dim=1)
        U = self.U(input)
        U = torch.tanh(input)
        #U = torch.sin(np.pi * U)
        
        V = self.V(input)
        V = torch.tanh(V)
        #V = torch.sin(np.pi * V)
        
        out = self.initlayer(input)

        for layer in self.layers:
            out = layer(out)
            out = out * U + (1-out) * V
        
        outNN = self.outlayer(out)

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

    def res_loss(self, pinn):
        x, y, t = self.points['res_points']
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

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
        lossesall = (self.adim[0] * (dxx_xy2uy[:,0] + dyx_yy2uy[:,1]) + self.adim[1] * 
                (dyx_yy2uy[:,1]) - self.adim[2] * ay.squeeze())
        
        loss_skew = skew(lossesall.detach().cpu().numpy()) 
        loss_kurt = kurtosis(lossesall.detach().cpu().numpy())
        
        loss = torch.tanh(pinn.res_penalties) * lossesall.reshape(self.n_space - 2, (self.n_space - 2) // self.scaley, self.n_time -1).pow(2)
        loss = loss.mean()

        eps = torch.stack([dxyux[:,0], 1/2*(dxyux[:,1]+dxyuy[:,0]), dxyuy[:,1]], dim=1)
        dV = ((self.par['w0']/self.par['Lx'])**2*(self.par['mu']*torch.sum(eps**2, dim=1)) + self.par['lam']/2 * torch.sum(eps, dim=1)**2)

        v = torch.cat([vx, vy], dim=1)
        vnorm = torch.norm(v, dim=1)
        dT = (1/2*(self.par['w0']/self.par['t_ast'])**2*self.par['rho']*vnorm**2)
        dT = dT * torch.max(dV)/torch.max(dT)

        dVt = dV.reshape(self.n_space - 2, (self.n_space - 2) // self.scaley, self.n_time - 1) 
        dTt = dT.reshape(self.n_space - 2, (self.n_space - 2) // self.scaley, self.n_time - 1) 

        V = self.b*simps(simps(dVt, self.steps[1], dim=1), self.steps[0])
        T = self.b*simps(simps(dTt, self.steps[1], dim=1), self.steps[0])

        Vbeam = self.interpVbeam(torch.unique(t).detach().cpu().numpy() * self.t_tild) 
        Ekbeam = self.interpEkbeam(torch.unique(t).detach().cpu().numpy() * self.t_tild) 
        Vbeam *= np.max(V.detach().cpu().numpy())/np.max(Vbeam)
        Ekbeam *= np.max(T.detach().cpu().numpy())/np.max(Ekbeam)

        errV = simpson((V.detach().cpu().numpy() - Vbeam)**2, dx=self.steps[2])/simpson(Vbeam**2, dx=self.steps[2])
        errT = simpson((T.detach().cpu().numpy() - Ekbeam)**2, dx=self.steps[2])/simpson(Ekbeam **2, dx=self.steps[2])
         
        return loss, V, T, errV, errT, loss_kurt, loss_skew, lossesall

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

        init = initial_conditions(space, self.w0)
        lossgridpos = (output[:,1] - init[:,1]).reshape(self.n_space, self.n_space // self.scaley)
        losspos = torch.tanh(pinn.in_penalties_p) * lossgridpos.pow(2)
        losspos = losspos.mean()
        vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones_like(t, device=self.device),
                create_graph=True, retain_graph=True)[0]
        
        v = torch.cat([vx, vy], dim=1)

        lossv = torch.tanh(pinn.in_penalties_v).unsqueeze(2) * (v * self.par['w0'] - init[:,2:]).reshape(self.n_space, self.n_space // self.scaley, 2).pow(2)
        lossv = lossv.mean()

        loss = losspos + lossv

        return loss, (losspos, lossv)

    def verbose(self, pinn):
        res_loss, V, T, errV, errT, kurt, skew, lossesall = self.res_loss(pinn)
        enloss = ((V+T)).pow(2).mean() 
        boundloss = self.bound_N_loss(pinn)
        init_loss, init_losses = self.initial_loss(pinn)
        loss = res_loss + init_loss

        losses = {
            "in_losses": init_losses,
            "in_loss": init_loss,
            "bound_loss": boundloss,
            "V": V,
            "T": T,
            "V+T": (V+T).mean(),
            "enloss": enloss,
            "errV": errV,
            "errT": errT,
            "kurt_res": kurt,
            "skew_res": skew,
            'loss_distr': lossesall
        }

        return loss, res_loss, losses 

    def __call__(self, pinn, inc_enloss = False):
        return self.verbose(pinn)

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

    exclude_params = ['res_penalties', 'in_penalties_p', 'in_penalties_v']
    params_non_excluded = [p for n, p in nn_approximator.named_parameters() if n not in exclude_params]
    params_excluded = [p for n, p in nn_approximator.named_parameters() if n in exclude_params]

    params_to_optimize = [
        {'params': params_non_excluded, 'lr': learning_rate},
        {'params': params_excluded, 'lr': -5e-3}
    ]
    adam_optimizer = optim.AdamW(params_to_optimize, weight_decay=0.01)
    lbfgs_optimizer = optim.LBFGS(params_non_excluded, lr=learning_rate)

    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    res_loss = None
    losses = None

    for epoch in range(max_epochs + 1):
        if epoch > max_epochs - 100:
            optimizer = lbfgs_optimizer
        else:
            optimizer = adam_optimizer
        def closure():
            nonlocal res_loss, losses
            optimizer.zero_grad()
            loss, res_loss, losses = loss_fn(nn_approximator)
            loss.backward(retain_graph=False)
            return loss

        loss = optimizer.step(closure)
        pbar.set_description(f"Loss: {loss.item():.3e}")

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

        writer.add_scalars('Loss/Distr_res', {
            "kurt_res": losses['kurt_res'],
            "skew_res": losses['skew_res'],
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
        
        if epoch % 200 == 0 :
            fig, ax = plt.subplots()
            cax = ax.imshow(nn_approximator.res_penalties[:,:,0].detach().cpu().numpy(), cmap='viridis')
            fig.colorbar(cax)
            ax.axis('off')
            plt.tight_layout()
            fig.canvas.draw()
            plt.close()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.add_image('Penalty res t=0', img, global_step=epoch, dataformats='HWC')
            fig, ax = plt.subplots()
            cax = ax.imshow(nn_approximator.in_penalties_p[:,:].detach().cpu().numpy(), cmap='viridis')
            fig.colorbar(cax)
            ax.axis('off')
            plt.tight_layout()
            fig.canvas.draw()
            plt.close()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.add_image('Penalty init', img, global_step=epoch, dataformats='HWC')

        if epoch % 500 == 0:
            t = loss_fn.points['res_points'][-1].unsqueeze(1)
            t = torch.unique(t, sorted=True)
            plot_energy(t.detach().cpu().numpy(), losses["V"].detach().cpu().numpy(), losses["T"].detach().cpu().numpy(), epoch, modeldir) 

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    writer.close()
    x, y, t = loss_fn.points['res_points']
    ydim = (loss_fn.n_space - 2) // loss_fn.scaley
    x = x.reshape(loss_fn.n_space - 2, ydim, loss_fn.n_time - 1).detach().cpu().numpy()[:,0,:]
    t = t.reshape(loss_fn.n_space - 2, ydim, loss_fn.n_time - 1).detach().cpu().numpy()[:,0,:]
    loss, res_loss, losses = loss_fn(nn_approximator, True)
    lossesdistr = losses['loss_distr'].reshape(loss_fn.n_space - 2, ydim, loss_fn.n_time - 1)
    lossesdistr = lossesdistr.detach().cpu().numpy()
    lossesdistr = np.abs(np.mean(lossesdistr, axis=1))
    fig, ax = plt.subplots()
    norm = mcolors.LogNorm(vmin=np.min(lossesdistr), vmax=np.max(lossesdistr))
    heatmap = ax.imshow(lossesdistr, extent=[t.min(), t.max(), x.min(), x.max()], origin='lower', 
                    aspect='auto', cmap='inferno', norm=norm)
    plt.colorbar(heatmap, ax=ax)
    ax.set_title(r'PDE Residuals')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    plt.savefig(f'{modeldir}/PDEres.png')

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