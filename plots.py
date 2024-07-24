import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib.animation import FuncAnimation
import numpy as np

def plot_distance0(output: torch.tensor, space: torch.tensor, path: str):
    ### t=0 ###
    plt.figure()
    plt.scatter(space[:,0].detach().cpu().numpy(), space[:,1].detach().cpu().numpy(), 
            c=output.detach().cpu().numpy(), cmap = 'viridis')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.colorbar()
    plt.title('$D(x_i, 0)$')

    plt.savefig(f'{path}/distance0.png')
    
def plot_initial_conditions(z: torch.tensor, z0: torch.tensor, x: torch.tensor, y: torch.tensor, n_space: int, path: str):
    """Plot initial conditions.
    z0: tensor describing analytical initial conditions
    z: tensor describing predicted initial conditions"""
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

    x_raw = x.detach().cpu().numpy()
    y_raw = y.detach().cpu().numpy()

    z = z.cpu().detach().numpy()
    z0 = z0.cpu().detach().numpy()

    X = x_raw
    Y = y_raw

    cmap = 'coolwarm'

    u_a_scatter = ax[0, 0].scatter(X.reshape(-1)+z0[:, 0],
                                 Y.reshape(-1)+z0[:, 1])
    ax[0, 0].set_xlabel('${x}$')
    ax[0, 0].set_ylabel('${y}$')

    vx_a_scatter = ax[0, 1].scatter(X.reshape(-1)+z0[:, 0],
                                 Y.reshape(-1)+z0[:, 1], c=z0[:,2], cmap=cmap)
    ax[0, 1].set_xlabel('${x}$')
    ax[0, 1].set_ylabel('${y}$')
    cbar1 = fig.colorbar(vx_a_scatter, ax=ax[0, 1], orientation='vertical')
    cbar1.set_label('$v_x$')
    
    vy_a_scatter = ax[0, 2].scatter(X.reshape(-1)+z0[:, 0],
                                 Y.reshape(-1)+z0[:, 1], c=z0[:,3], cmap=cmap)
    ax[0, 2].set_xlabel('${x}$')
    ax[0, 2].set_ylabel('${y}$')
    cbar2 = fig.colorbar(vy_a_scatter, ax=ax[0, 2], orientation='vertical')
    cbar2.set_label('$v_y$')
    
    nn_scatter = ax[1, 0].scatter(X.reshape(-1)+z[:, 0],
                                 Y.reshape(-1)+z[:, 1])
    ax[1, 0].set_xlabel('${x}$')
    ax[1, 0].set_ylabel('${y}$')

    vx_nn_scatter = ax[1, 1].scatter(X.reshape(-1)+z[:, 0],
            Y.reshape(-1)+z[:, 1], c=z[:, 2], cmap=cmap)
    ax[1, 1].set_xlabel('${x}$')
    ax[1, 1].set_xlabel('${y}$')
    cbar3 = fig.colorbar(vx_nn_scatter, ax=ax[1, 1], orientation='vertical')
    cbar3.set_label('$v_x$')
    
    vy_nn_scatter = ax[1, 2].scatter(X.reshape(-1)+z[:, 0],
            Y.reshape(-1)+z[:, 1], c=z[:, 3], cmap=cmap)
    ax[1, 2].set_xlabel('${x}$')
    ax[1, 2].set_xlabel('${y}$')
    cbar4 = fig.colorbar(vy_nn_scatter, ax=ax[1, 2], orientation='vertical')
    cbar4.set_label('$v_y$')

    fig.text(0.5, 0.96, 'Analytical', ha='center', va='center', fontsize=16)
    fig.text(0.5, 0.48, 'Predicted', ha='center', va='center', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{path}/init.png')


def plot_sol_comparison(sol: torch.tensor, space: torch.tensor, t: torch.tensor, w_num: np.ndarray,
        path: str):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    t = torch.unique(t, sorted=True).detach().cpu().numpy()
    space = space.detach().cpu().numpy()

    ax.scatter(space[:,0]+sol[:,0,0], space[:,1]+sol[:,0,1])
    x = np.unique(space[:,0])
    ax.scatter(x, w_num[:,0])

    ax.set_title(f'$t = {t[0]:.2f}$')

    def update(frame):
        y_limts = np.array([-0.5, 0.5])

        ax.clear()

        ax.set_xlabel('${x}$')
        ax.set_ylabel('${y}$')
        ax.set_title(f'${{t}} = {t[frame]:.2f}$')

        ax.set_ylim(np.min(y_limts), np.max(y_limts))
        ax.scatter(space[:,0]+sol[:,frame,0], space[:,1]+sol[:,frame,1])
        ax.scatter(x, w_num[:,frame])

        return ax

    n_frames = t.shape[0]
    ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    file = f'{path}/sol_time_comparison.gif'
    ani.save(file, fps=5)


def plot_sol(sol: torch.tensor, space: torch.tensor, t: torch.tensor, path: str):
    # y_plot squeezed for better visualization purposes, anyway is not encoded in the 1D solution, displacements not squeezed

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    t = torch.unique(t, sorted=True).detach().cpu().numpy()
    space = space.detach().cpu().numpy()
    ax.scatter(space[:,0]+sol[:,0,0], space[:,1]+sol[:,0,1])

    ax.set_title(f'${{t}} = {t[0]:.2f}$')

    def update(frame):
        y_limts = np.array([-0.5, 0.5])

        ax.clear()

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(f'$t = {t[frame]:.2f}$')

        ax.set_ylim(np.min(y_limts), np.max(y_limts))
        ax.scatter(space[:,0]+sol[:,frame,0], space[:,1]+sol[:,frame,1])

        return ax

    n_frames = t.shape[0]
    ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)

    file = f'{path}/sol_time.gif'
    ani.save(file, fps=5)


def plot_compliance(x: torch.tensor, y: torch.tensor,
                    t: torch.tensor, w_ad: np.ndarray, path: str, device):
    pass

def plot_indicators(indicators: dict, t: torch.tensor, path: str):
    t = torch.unique(t, sorted=True).detach().cpu().numpy()

    T = indicators['T']
    Pi = indicators['Pi']

    plt.figure()
    plt.plot(t, T, label='$T$')
    plt.plot(t, Pi, label='$\\Pi$')
    plt.legend()

    file = f'{path}/energyfinal.png'
    plt.savefig(file)
