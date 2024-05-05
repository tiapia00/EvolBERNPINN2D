import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import torch
from matplotlib.animation import FuncAnimation
from pinn import PINN, f
import numpy as np

def plot_initial_conditions(z: torch.tensor, z0: torch.tensor, x: torch.tensor, y: torch.tensor, n_train: int, path: str):
    """Plot initial conditions.
    z0: tensor describing analytical initial conditions
    z: tensor describing predicted initial conditions"""
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()

    z = z.cpu().detach().numpy()
    z0 = z0.cpu().detach().numpy()
   
    X = x_raw
    Y = y_raw

    cmap = 'viridis'
    
    norm_z0 = np.linalg.norm(z0, axis=1).reshape(-1)
    norm_z = np.linalg.norm(z, axis=1).reshape(-1)
    
    a_scatter = ax[0].scatter(X.reshape(-1)+z0[:,0], Y.reshape(-1)+z0[:,1], c=norm_z0, cmap=cmap)
    ax[0].set_xlabel('$\\hat{x}$')
    ax[0].set_ylabel('$\\hat{y}$')
    ax[0].set_title('Analyitcal initial conditions')
    cbar1 = fig.colorbar(a_scatter, ax=ax[0], orientation='vertical')
    cbar1.set_label('$|\\mathbf{u}|$')
    
    p_scatter = ax[1].scatter(X.reshape(-1)+z[:,0], Y.reshape(-1)+z[:,1], c=norm_z, cmap=cmap)
    ax[1].set_xlabel('$\\hat{x}$')
    ax[1].set_ylabel('$\\hat{y}$')
    ax[1].set_title('Predicted initial conditions')
    cbar2 = fig.colorbar(p_scatter, ax=ax[1], orientation='vertical')
    cbar2.set_label('$|\\mathbf{u}|$')
    
    plt.tight_layout()
    
    plt.savefig(f'{path}/init.png')
    
def plot_uy(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, n_train : int, path : str, figsize=(12, 8), dpi=100):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title('$u_y$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 2)
    
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
    
    x_plot = x.cpu().detach().numpy().reshape(n_train, n_train)
    y_plot = y.cpu().detach().numpy().reshape(n_train, n_train)
    uy = output[:, 1].reshape(n_train, n_train).cpu().detach().numpy()
    
    ax.plot_surface(x_plot, y_plot, uy, cmap='viridis')
    
    def update(
        frame,
        x: torch.tensor,
        y: torch.tensor,
        x_plot: np.ndarray, 
        y_plot: np.ndarray,
        t_raw: torch.tensor,
        t_shaped: torch.tensor,
        pinn: PINN,
        ax):
        
        t = t_shaped*t_raw[frame]
        
        output = f(pinn, x, y, t)
        
        uy = output[:, 1].reshape(n_train, n_train).cpu().detach().numpy()
        
        t_value = float(t[0])
        
        ax.clear()
        
        ax.set_xlabel('$\\hat{x}$')
        ax.set_ylabel('$\\hat{y}$')
        ax.set_zlabel('$\\hat{u}_y$')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 0.5)
        ax.text(0, 0.5, 0.8, s=fr'$\hat{{t}} = {t_value:.2f}$', fontsize=10, color='black', ha='center')
        ax.plot_surface(x_plot, y_plot, uy, cmap='viridis')
        
        return ax
    
    n_frames = len(t_raw)
    ani = FuncAnimation(fig, update, frames=n_frames, fargs=(x, y, x_plot, y_plot, t_raw, t_shaped, pinn, ax), interval=50, blit=False)
    
    file = f'{path}/uy.gif'
    ani.save(file, fps=30)


        
        
        
        
        
    