import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import torch
from matplotlib.animation import FuncAnimation
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
    
def plot_uy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, n_train : int, figsize=(12, 8), dpi=100):

    rc('animation', html='jshtml')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('$u_y$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 20)
    
    t_raw = torch.unique(t, sorted=True)
    t_raw = t_raw.reshape(-1, 1)
    
    x_raw = x.reshape(n_train, n_train, n_train)
    y_raw = y.reshape(n_train, n_train, n_train)
    
    x = x_raw[:,:,0]
    y = y_raw[:,:,0]
    
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    
    t_shaped = torch.ones_like(x)
    t = t_shaped*t_raw[0]
    
    output = f(pinn, x, y, t)
    
    x = x.reshape(n_train, n_train)
    y = y.reshape(n_train, n_train)
    
    x_plot = x.cpu().detach().numpy()
    y_plot = y.cpu().detach().numpy()
    uy = output[:, 1].reshape(n_train, n_train).cpu().detach().numpy()
    
    legend = fr'$\hat{{t}}={{{float(t[0]):.2f}}}$' 
    ax.plot_surface(x_plot, y_plot, uy, cmap='viridis', label=legend)
    ax.legend()
    
    def update(frame):
        t = t_shaped*t_raw[frame]
        output = f(pinn, x, y, t)
        
        uy = output[:, 1].reshape(n_train, n_train).cpu()
        
        t_value = float(t[0])
        legend = fr'$\hat{{t}}={{{t_value:.2f}}}$' 
        surf.set_array(uy)
        
        ax.legend()
        
        return surf
    
    n_frames = len(t_raw)
    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    
    ani.save('uy.mp4', writer='ffmpeg', fps=30)


        
        
        
        
        
    