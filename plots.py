import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import torch
import matplotlib.animation as animation
from matplotlib import rc
from pinn import PINN, f

def plot_initial_conditions(z: torch.tensor, y: torch.tensor, x: torch.tensor, name: str, n_train: int, from_pinn: bool = True):
    """Plot initial conditions."""
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 20)
    
    ax.set_xlabel('$\\hat{y}$')
    ax.set_ylabel('$\\hat{x}$')
    ax.set_title('$\\hat{u}_y(t)$')
    ax.view_init(elev=30, azim=45)
    t_raw = torch.unique(t)
    
    def animate(i):

        if not i % 10 == 0:
            ax.clear()
            t_partial = torch.ones_like(x_raw) * t_raw[i]
            f = f(pinn, x_raw, y_raw, t_partial)
            uy_raw = f.detach().cpu().numpy()[:,1]
            
            x_raw = x.detach().cpu().numpy()
            y_raw = y.detach().cpu().numpy()
            
            x_mesh = x_raw.reshape(n_train, n_train, n_train)
            y_mesh = y_raw.reshape(n_train, n_train, n_train)
            uy = uy_raw.reshape(n_train, n_train, n_train)
            
            ax.plot_surface(x_mesh[:,:,i], y_mesh[:,:,i], uy[:,:,i], label=f"Time {float(t_raw[i])}", cmap='viridis')
            
            return ax
        
    n_frames = t_raw.shape[0]
    return animation.FuncAnimation(fig, animate, frames=n_frames, blit=False, repeat=True)
    
    