import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib.animation import FuncAnimation
from pinn import PINN
import numpy as np


def plot_initial_conditions(z: torch.tensor, z0: torch.tensor, x: torch.tensor, y: torch.tensor, path: str):
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

    ax[0, 0].scatter(X.reshape(-1)+z0[:, 0], Y.reshape(-1)+z0[:, 1])
    ax[0, 0].set_xlabel('$\\hat{x}$')
    ax[0, 0].set_ylabel('$\\hat{y}$')

    vx_a_scatter = ax[0, 1].scatter(X.reshape(-1)+z0[:, 0],
                                 Y.reshape(-1)+z0[:, 1], c=z0[:,2], cmap=cmap)
    ax[0, 1].set_xlabel('$\\hat{x}$')
    ax[0, 1].set_ylabel('$\\hat{y}$')
    cbar1 = fig.colorbar(vx_a_scatter, ax=ax[0, 1], orientation='vertical')
    cbar1.set_label('$v_x$')
    
    vy_a_scatter = ax[0, 2].scatter(X.reshape(-1)+z0[:, 0],
                                 Y.reshape(-1)+z0[:, 1], c=z0[:,3], cmap=cmap)
    ax[0, 2].set_xlabel('$\\hat{x}$')
    ax[0, 2].set_ylabel('$\\hat{y}$')
    cbar2 = fig.colorbar(vy_a_scatter, ax=ax[0, 2], orientation='vertical')
    cbar2.set_label('$v_y$')
    
    ax[1, 0].scatter(X.reshape(-1)+z[:, 0], Y.reshape(-1)+z[:, 1])
    ax[1, 0].set_xlabel('$\\hat{x}$')
    ax[1, 0].set_ylabel('$\\hat{y}$')

    vx_nn_scatter = ax[1, 1].scatter(X.reshape(-1)+z[:, 0],
            Y.reshape(-1)+z[:, 1], c=z[:, 2], cmap=cmap)
    ax[1, 1].set_xlabel('$\\hat{x}$')
    ax[1, 1].set_xlabel('$\\hat{y}$')
    cbar3 = fig.colorbar(vx_nn_scatter, ax=ax[1, 1], orientation='vertical')
    cbar3.set_label('$v_x$')
    
    vy_nn_scatter = ax[1, 2].scatter(X.reshape(-1)+z[:, 0],
            Y.reshape(-1)+z[:, 1], c=z[:, 3], cmap=cmap)
    ax[1, 2].set_xlabel('$\\hat{x}$')
    ax[1, 2].set_xlabel('$\\hat{y}$')
    cbar4 = fig.colorbar(vy_nn_scatter, ax=ax[1, 2], orientation='vertical')
    cbar4.set_label('$v_y$')

    fig.text(0.5, 0.96, 'Analytical', ha='center', va='center', fontsize=16)
    fig.text(0.5, 0.48, 'Predicted', ha='center', va='center', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{path}/init.png')

def plot_sol(sol: torch.tensor, space: torch.tensor, t: torch.tensor, path: str):
    # y_plot squeezed for better visualization purposes, anyway is not encoded in the 1D solution, displacements not squeezed

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    t = torch.unique(t, sorted=True).detach().cpu().numpy()
    space = space.detach().cpu().numpy()
    ax.scatter(space[:,0]+sol[:,0,0], space[:,1]+sol[:,0,1])

    ax.set_title(f'$\\hat{{t}} = {t[0]:.2f}$')

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
    ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)

    file = f'{path}/sol_time.gif'
    ani.save(file, fps=5)


def plot_compliance(pinn: PINN, x: torch.tensor, y: torch.tensor,
                    t: torch.Tensor, w_ad: np.ndarray, path: str, device):

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    fig.suptitle('Compliance')

    x = x.to(device)
    y = y.to(device)

    t_raw = torch.unique(t, sorted=True)
    mean_y = []

    for t in t_raw:
        t = t*torch.ones_like(x).to(device)
        output = f(pinn, x, y, t)
        uy = output[:, 1].cpu().detach().numpy()
        mean_y.append(np.mean(uy))

    mean_y = np.array(mean_y)

    ax[0].plot(t_raw.cpu().detach().numpy(), mean_y, color='blue')
    ax[0].set_title('Prediction from PINN')
    ax[0].set_xlabel('$\\hat{t}$')
    ax[0].set_ylabel('$\\overline{u}_y$')

    ax[1].plot(t_raw.cpu().detach().numpy(), mean_y - np.mean(w_ad, axis=0), color='red')
    ax[1].set_title('Deviation from analytical')
    ax[1].set_xlabel('$\\hat{t}$')
    ax[1].set_ylabel('$\\overline{u}_{y,an}-\\overline{u}_{y,PINN}$')

    plt.grid()
    plt.tight_layout()

    file = f'{path}/compliance.png'
    plt.savefig(file)


def plot_energy(t: torch.tensor, en_k: torch.tensor, en_p: torch.tensor, en: torch.tensor, e0: float, path):
    fig = plt.figure(figsize=(10, 8))
    plt.xlabel('$\\hat{t}$')

    t = t.detach().cpu().numpy()
    en = en.detach().cpu().numpy()
    en_k = en_k.detach().cpu().numpy()
    en_p = en_p.detach().cpu().numpy()
    e0 = e0.detach().cpu().numpy()

    plt.plot(t, en, label='Total energy')
    plt.plot(t, en_k, label='Kinetic energy')
    plt.plot(t, en_p, label='Potential energy')

    plt.legend()

    file = f'{path}/energy.png'
    plt.savefig(file)

    fig = plt.figure(figsize=(10, 8))

    plt.plot(t, e0 - en)

    plt.xlabel('$\\hat{t}$')
    plt.ylabel('$E_0 - E(t)$')

    file = f'{path}/diff_energy.png'
    plt.savefig(file)
