from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Callable
import numpy as np
import torch
from torch import device, nn
import torch.optim as optim


def initial_conditions(x: torch.tensor, w0: float, i: float = 1) -> torch.tensor:
    ux0 = torch.zeros_like(x)
    uy0 = w0*torch.sin(torch.pi*x/torch.max(x))
    dotux0 = torch.zeros_like(x)
    dotuy0 = torch.zeros_like(x)
    return torch.cat((ux0, uy0, dotux0, dotuy0), dim=1)

def geteps(space, output, nsamples, device):
    duxdxy = torch.autograd.grad(output[:, 0].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    duydxy = torch.autograd.grad(output[:, 1].unsqueeze(1), space, torch.ones(space.size()[0], 1, device=device),
                                 create_graph=True, retain_graph=True)[0]
    H = torch.zeros(nsamples[0], nsamples[1], nsamples[2], 2, 2, device=device)
    duxdxy = duxdxy.reshape(nsamples[0], nsamples[1], nsamples[2], 2)
    duydxy = duydxy.reshape(nsamples[0], nsamples[1], nsamples[2], 2)
    # last 2 is related to the the components of the derivative
    # nx, ny, nt, 2, 2
    H[:, :, :, 0, :] = duxdxy
    H[:, :, :, 1, :] = duydxy
    eps = H
    eps[:, :, :, [0, 1], [1, 0]] = 0.5 * (eps[:, :, :, 0, 1] + 
            eps[:, :, :, 1, 0]).unsqueeze(3).expand(-1,-1,-1,2)

    return eps

def material_model(eps, mat_par: tuple,  device):
    tr_eps = eps.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) 
    lam = mat_par[0]
    mu = mat_par[1]
    sig = 2 * mu * eps + lam * torch.einsum('ijk,lm->ijklm', tr_eps, torch.eye(eps.size()[-1], device=device)) 
    psi = torch.einsum('ijklm,ijklm->ijk', eps, sig)

    return psi, sig

def getspeed(output: torch.tensor, t: torch.tensor, device: torch.device):
    n = output.shape[0]

    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones(n, 1, device=device),
                create_graph=True, retain_graph=True)[0]
    vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones(n, 1, device=device),
                create_graph=True, retain_graph=True)[0]
    
    return torch.cat([vx, vy], dim=1)


def getacc(output: torch.tensor, t: torch.tensor, device: torch.device):
    n = output.shape[0]

    v = getspeed(output, t, device)
    ax = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones(n, 1, device=device),
             create_graph=True, retain_graph=True)[0]
    ay = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones(n, 1, device=device),
             create_graph=True, retain_graph=True)[0]
    
    return torch.cat([ax, ay], dim=1)


def getkinetic(speed: torch.tensor, nsamples: tuple, rho: float, ds: tuple):
    dx = ds[0]
    dy = ds[1]

    speed = speed.reshape(nsamples[0], nsamples[1], nsamples[2], 2)
    magnitude = torch.norm(speed, p=2, dim=3)
    ### ASSUMPTION: t = 1 ###
    kinetic = 1/2 * rho * torch.trapezoid(torch.trapezoid(magnitude, dx=dy, dim=1),
            dx = dx, dim=0)

    return kinetic

def getPsi(psi: torch.tensor, ds: tuple):
    dx = ds[0]
    dy = ds[1]

    Psi = torch.trapezoid(torch.trapezoid(y = psi, dx = dy, dim=1), dx=dx, dim=0)

    return Psi
    

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


def obtain_centers(points, step):
    device = points.device
    max_x = torch.max(points[:,0])
    max_y = torch.max(points[:,1])
    x, y = torch.meshgrid(torch.linspace(0, max_x, step, device=device), 
            torch.linspace(0, max_y, step, device=device), indexing='ij')

    centers = torch.cat([x.reshape(-1,1), y.reshape(-1,1)], dim=1)
    centers.requires_grad_(False)
    return centers


class RBFLayer(nn.Module):
    def __init__(self, basis_fun, all_points, step):
        super().__init__()
        self.centers = obtain_centers(all_points, step)  # Centers of the RBFs
        self.basis_fun = basis_fun

    def forward(self, x):
        dists = torch.cdist(x, self.centers)
        return self.basis_fun(dists) 


class RBF(nn.Module):
    def __init__(self, basis_fun, all_points, step, out_features):
        super().__init__()
        self.rbf = RBFLayer(basis_fun, all_points, step)
        self.fc = nn.Linear(self.rbf.centers.shape[0], out_features)

    def forward(self, x):
        x = self.rbf(x)
        x = self.fc(x)
        return x


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def gaussian(alpha, beta):
    phi = torch.exp(-beta*alpha.pow(2))
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi

class TrigAct(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def parabolic(a, x):
    return (a * x ** 2 - a * x)


class NNinbc(nn.Module):
    def __init__(self, dim_hidden, n_hidden):

        super().__init__()

        self.layerin = nn.Linear(3, dim_hidden)

        self.layers = nn.ModuleList()
        for _ in range(n_hidden - 1):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(TrigAct())
        
        self.layerout = nn.Linear(dim_hidden, 2)

    def forward(self, points):
        output = self.layerin(points)

        for layer in self.layers:
            output = layer(output)
        
        output = self.layerout(output)

        ### Transformation for Dirichlet BCs ###
        output *= torch.sin(np.pi * points[:,0]/torch.max(points[:,0])).unsqueeze(1).expand(-1,2)

        return output


class NNd(nn.Module):
    def __init__(self, dim_hidden, n_hidden):

        super().__init__()

        self.dim_hidden = dim_hidden
        self.n_hidden = n_hidden

        self.layerin = nn.Linear(3, dim_hidden)

        self.layers = nn.ModuleList()
        for _ in range(n_hidden - 1):
            self.layers.append(nn.Linear(dim_hidden, dim_hidden))
            self.layers.append(nn.Tanh())
        
        self.layerout = nn.Linear(dim_hidden, 1)

    def forward(self, points):
        output = self.layerin(points)

        for layer in self.layers:
            output = layer(output)
        
        output = self.layerout(output)
        output *= points[:,2].unsqueeze(1)

        return output


def omtogam_trans(omega: torch.tensor, prop: dict):
    E = prop['E']
    m = prop['m']
    J = prop['J']
    gamma = ((m/(E*J))**(1/4)*omega**(1/2))

    return gamma

def omtogam_ax(omega: torch.tensor, prop: dict):
    E = prop['E']
    m = prop['m']
    A = prop['A']
    c = (E*A/m)**(1/2)
    gamma = omega/c

    return gamma


class PINN(nn.Module):
    def __init__(self,
                 dim_hidden: int,
                 nhidden_t: int,
                 inbcsNN: NNinbc,
                 all_points: torch.tensor,
                 device: torch.device,
                 act=TrigAct(),
                 ):

        super().__init__()

        self.inbcsNN = inbcsNN
        
        for param in self.inbcsNN.parameters():
            param.requires_grad = False
        
        self.axial = RBF(matern52, all_points, 20, 1, device)
        self.trans = RBF(matern52, all_points, 20, 1, device)

        self.timelayers = nn.ModuleList()
        self.timelayers.append(nn.Linear(1, dim_hidden))
        for _ in range(nhidden_t):
            self.timelayers.append(nn.Linear(dim_hidden, dim_hidden))
            self.timelayers.append(act)
        self.timelayers.append(nn.Linear(dim_hidden, 2))

    def forward(self, space, t):
        axial = self.axial(space)
        trans = self.trans(space)
        time = t

        points = torch.cat([space, t], dim=1)

        for layer in self.timelayers:
            t = layer(t)

        out = torch.cat([axial, trans], dim=1) * t
        
        out_inbcs =  self.inbcsNN(points)

        out = out * time.expand(-1,2) + out_inbcs
        out *= torch.sin(np.pi * points[:,0]/torch.max(points[:,0])).unsqueeze(1).expand(-1,2)

        return out

    def device(self):
        return next(self.parameters()).device


class Calculate:
    def __init__(
        self,
        initial_condition: Callable,
        # lam, mu, rho
        m_par: tuple,
        points: dict,
        nsamples: tuple,
        steps_int: tuple,
        w0: float,
        device: torch.device,
    ):
        self.initial_condition = initial_condition
        self.m_par = m_par
        self.points = points
        self.w0 = w0
        self.nsamples = nsamples 
        self.steps = steps_int
        self.device = device
        self.dists = self.gtdistance()

    def gettraction(self, pinn):
        pass

    def gete0(self, pinn):
        x, y, t = self.points['initial_points']
        nsamples = (self.nsamples[0], self.nsamples[1], 1)
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        lam, mu, rho = self.m_par
        dx, dy, dt = self.steps

        eps = geteps(space, output, nsamples, self.device)
        psi, sig = material_model(eps, (lam, mu), self.device)
        Psi = getPsi(psi, (dx, dy)).reshape(-1)

        speed = getspeed(output, t, self.device)
        K = getkinetic(speed, nsamples, rho, (dx, dy)).reshape(-1)

        return (Psi, K) 

    def pdeloss(self, pinn):
        x, y, t = self.points['all_points']
        nsamples = (self.nsamples[0], self.nsamples[1], self.nsamples[2])
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        lam, mu, rho = self.m_par
        dx, dy, dt = self.steps

        eps = geteps(space, output, nsamples , self.device)
        _, sig = material_model(eps, (lam, mu), self.device)

        a = getacc(output, t, self.device)

        div_sig = torch.zeros((sig.shape[0], sig.shape[1], sig.shape[2], 2), device=self.device)
        partial_div = torch.zeros((sig.shape[0], sig.shape[1], sig.shape[2]), device=self.device)

        for i in range(2):
            for j in range(2):
                grad = torch.autograd.grad(sig[:,:,:,i, j], space, torch.ones((sig.shape[0], 
                        sig.shape[1], sig.shape[2]), device=self.device),
                        create_graph=True, retain_graph=True)[0]
                partial_div += grad[:,j].reshape(sig.shape[0], sig.shape[1], sig.shape[2])
            div_sig[:,:,:,i] = partial_div
        
        div_sig = div_sig.reshape(-1, 2)
        diff = div_sig - self.m_par[2]*a
        
        loss = 0
        
        loss += (diff[:,0]).pow(2).mean()
        loss += (diff[:,1]).pow(2).mean()

        return loss


    def enloss(self, pinn, verbose: bool):
        x, y, t = self.points['all_points']
        nsamples = (self.nsamples[0], self.nsamples[1], self.nsamples[2])
        space = torch.cat([x, y], dim=1)
        output = pinn(space, t)

        lam, mu, rho = self.m_par
        dx, dy, dt = self.steps

        eps = geteps(space, output, nsamples , self.device)
        psi, sig = material_model(eps, (lam, mu), self.device)
        Pi = getPsi(psi, (dx, dy)).reshape(-1)
        # Pi should take into account also external forces applied

        speed = getspeed(output, t, self.device)
        T = getkinetic(speed, nsamples, rho, (dx, dy)).reshape(-1)

        deren = torch.autograd.grad((T - Pi).unsqueeze(1), t, torch.ones(Pi.shape[0], 1, device=self.device),
                 create_graph=True, retain_graph=True)[0]

        loss = deren.pow(2).mean()

        if verbose:
            return (Pi, T)
        else:
            return loss

        
    def gtdistance(self):
        x, y, t = self.points['all_points']
        points = torch.cat([x, y, t], dim=1)
        down, up, _, _, _ = self.points['boundary_points']
        bound_points = torch.cat([down, up], dim=0)

        x = points[:,:-1]
        t = points[:,-1]

        x_bc = bound_points[:,:-1]
        t_bc = bound_points[:,-1]

        n = x.shape[0]
        dists = torch.zeros(n, device=self.device)
        
        for i in range(n):
            dist = torch.norm(x[i,:] - x_bc, dim=1)**2 + (t[i] - t_bc)**2
            dists[i] = torch.sqrt(torch.min(dist))
        
        return dists
    
    def distance_loss(self, nn):
        x, y, t = self.points['all_points']
        points = torch.cat([x, y, t], dim=1)

        output = nn(points)
        loss = (output.squeeze() - self.dists.detach()).pow(2).mean()

        ddot = torch.autograd.grad(output, t, torch.ones(output.shape[0], 1, device=self.device),
                 create_graph=True, retain_graph=True)[0]
        idx_0 = torch.nonzero(t == 0, as_tuple=False)
        loss += ddot[idx_0].pow(2).mean()
        
        return loss

    def initial_loss(self, nn):
        x, y, t = self.points['initial_points']
        points = torch.cat([x, y, t], dim=1)

        output = nn(points)

        gt = initial_conditions(x, self.w0)
        v0gt = gt[:,2:]
        v0 = getspeed(output, t, self.device)
        loss_speed = (v0gt - v0).pow(2).mean()

        posgt = gt[:,:2] 
        loss_position = (posgt - output).pow(2).mean()

        loss = loss_speed + loss_position

        return loss

    def boundary_loss(self, pinn):
        ### Maybe just for Neumann ##
        pass


def train_model(
    nn_approximator: PINN,
    calc: Callable,
    learning_rate: int,
    max_epochs: int,
    path_logs: str
) -> PINN:

    writer = SummaryWriter(log_dir=path_logs)

    optimizer = optim.Adam(nn_approximator.parameters(), lr = learning_rate)
    pbar = tqdm(total=max_epochs, desc="Training", position=0)

    for epoch in range(max_epochs):

        optimizer.zero_grad()
        loss = calc.pdeloss(nn_approximator) + calc.enloss(nn_approximator)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        writer.add_scalars('Loss', {
            'action': loss.item(),
        }, epoch)

        pbar.update(1)

    losses = calc.enloss(nn_approximator, True)
    Pi, T = losses
    variables = {'Pi': Pi.detach().cpu().numpy(), 'T': T.detach().cpu().numpy()}

    pbar.update(1)
    pbar.close()

    writer.close()

    return nn_approximator, variables

def train_inbcs(nn: NNinbc, calc: Calculate, epochs: int, learning_rate: float):
    optimizer = optim.Adam(nn.parameters(), lr = learning_rate)
    pbar = tqdm(total=epochs, desc="Training", position=0)

    for epoch in range(epochs):

        optimizer.zero_grad()
        loss = calc.initial_loss(nn)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    return nn

def train_dist(nn: NNd, calc: Calculate, epochs: int, learning_rate: float):
    optimizer = optim.Adam(nn.parameters(), lr = learning_rate)
    pbar = tqdm(total=epochs, desc="Training", position=0)

    for epoch in range(epochs):

        optimizer.zero_grad()
        loss = calc.distance_loss(nn)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.3e}")

        pbar.update(1)

    pbar.update(1)
    pbar.close()

    return nn



def calculate_speed(pinn_trained: PINN, points: tuple, device: torch.device) -> torch.tensor:
    x, y, t = points
    space = torch.cat([x, y], dim=1)
    n = space.shape[0]

    output = pinn_trained(space, t)

    vx = torch.autograd.grad(output[:,0].unsqueeze(1), t, torch.ones(n, 1, device=device),
             create_graph=True, retain_graph=True)[0]
    vy = torch.autograd.grad(output[:,1].unsqueeze(1), t, torch.ones(n, 1, device=device),
             create_graph=True, retain_graph=True)[0]

    return torch.cat([vx, vy], dim=1)


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


def obtainsolt(pinn: PINN, space_in: torch.tensor, t:torch.tensor, nsamples: tuple, device):
    nx, ny, nt = nsamples
    sol = torch.zeros(nx, ny, nt, 2)
    ts = torch.unique(t, sorted=True)
    ts = ts.reshape(-1,1)

    for i in range(ts.shape[0]):
        t = ts[i]*torch.ones(space_in.shape[0], 1, device=device)
        output = pinn(space_in, t)
        gridoutput = output.reshape(nx, ny, 2)
        sol[:,:,i,:] = gridoutput
    
    sol = sol.reshape(nx*ny, -1, 2)
    return sol.detach().cpu().numpy()