import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib.animation import FuncAnimation
from pinn import PINN
import numpy as np


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
    
    nn_scatter = ax[1, 0].scatter(X.reshape(-1)+z[:, 0],
                                 Y.reshape(-1)+z[:, 1])
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


def plot_sol_comparison(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, w_ad: np.ndarray,
                        n_space: int, n_time: int, n_beam: int, path: str, device):

    # y_plot squeezed for better visualization purposes, anyway is not encoded in the 1D solution, displacements not squeezed

    nx = n_space
    ny = n_space
    nt = n_time

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    t_raw = torch.unique(t, sorted=True)
    t_raw = t_raw.reshape(-1, 1)

    x_raw = x.reshape(nx, ny, nt)
    y_raw = y.reshape(nx, ny, nt)

    x = x_raw[:, :, 0]
    y = y_raw[:, :, 0]

    x = x.reshape(-1, 1).to(device)
    y = y.reshape(-1, 1).to(device)

    x_mid = torch.unique(x).reshape(-1, 1)
    y_mid = torch.zeros_like(x_mid)

    t_shaped = torch.ones_like(x).to(device)
    t = t_shaped*t_raw[0].to(device)

    output = f(pinn, x, y, t)

    x_plot = x.cpu().detach().numpy().reshape(nx, ny).reshape(-1)
    y_plot = 1/10*y.cpu().detach().numpy().reshape(nx, ny).reshape(-1)

    z0 = output.cpu().detach().numpy()

    sc = ax.scatter(x_plot+z0[:, 0], y_plot+z0[:, 1])
    ax.scatter(np.unique(x_plot), w_ad[::int(n_beam/n_space),0])
    t_value = float(t_raw[0])

    ax.set_title(f'$\\hat{{t}} = {t_value:.2f}$')

    def update(
            frame,
            x: torch.tensor,
            y: torch.tensor,
            x_mid: torch.tensor,
            y_mid: torch.tensor,
            n: int,
            w_ad: np.ndarray,
            x_plot: np.ndarray,
            y_plot: np.ndarray,
            t_raw: torch.tensor,
            t_shaped: torch.tensor,
            pinn: PINN,
            ax):

        x_limts = np.array([0, 2])
        y_limts = np.array([-0.5, 0.5])
        t = t_shaped*t_raw[frame].to(device)

        output = f(pinn, x, y, t)

        z = output.cpu().detach().numpy()

        t_value = float(t[0])

        ax.clear()

        ax.set_xlabel('$\\hat{x}$')
        ax.set_ylabel('$\\hat{y}$')
        ax.set_title(f'$\\hat{{t}} = {t_value:.2f}$')

        ax.set_xlim(np.min(x_limts), np.max(x_limts))
        ax.set_ylim(np.min(y_limts), np.max(y_limts))
        sc = ax.scatter(x_plot+z[:, 0], y_plot+z[:, 1])
        ax.scatter(np.unique(x_plot), w_ad[::int(n_beam/n_space), frame])

        return ax

    n_frames = len(t_raw)
    ani = FuncAnimation(fig, update, frames=n_frames,
                        fargs=(x, y, x_mid, y_mid, nx, w_ad, x_plot, y_plot, t_raw, t_shaped, pinn, ax), interval=100, blit=False)

    file = f'{path}/sol_time_comparison.gif'
    ani.save(file, fps=5)


def plot_sol(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
             n_space: int, n_time: int, path: str, device):

    # y_plot squeezed for better visualization purposes, anyway is not encoded in the 1D solution, displacements not squeezed

    nx = n_space
    ny = n_space
    nt = n_time

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    t_raw = torch.unique(t, sorted=True)
    t_raw = t_raw.reshape(-1, 1)

    x_raw = x.reshape(nx, ny, nt)
    y_raw = y.reshape(nx, ny, nt)

    x = x_raw[:, :, 0]
    y = y_raw[:, :, 0]

    x = x.reshape(-1, 1).to(device)
    y = y.reshape(-1, 1).to(device)

    t_shaped = torch.ones_like(x).to(device)
    t = t_shaped*t_raw[0].to(device)

    output = f(pinn, x, y, t)

    x_plot = x.cpu().detach().numpy().reshape(nx, ny).reshape(-1)
    y_plot = 1/10*y.cpu().detach().numpy().reshape(nx, ny).reshape(-1)

    z0 = output.cpu().detach().numpy()
    norm = np.linalg.norm(z0, axis=1).reshape(-1)

    ax.scatter(x_plot+z0[:, 0], y_plot+z0[:, 1], c=norm, cmap='viridis')
    t_value = float(t_raw[0])

    ax.set_title(f'$\\hat{{t}} = {t_value:.2f}$')

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

        x_limts = np.array([0, 2])
        y_limts = np.array([-0.5, 0.5])
        t = t_shaped*t_raw[frame].to(device)

        output = f(pinn, x, y, t)

        z = output.cpu().detach().numpy()
        norm = np.linalg.norm(z, axis=1).reshape(-1)
        t_value = float(t[0])

        ax.clear()

        ax.set_xlabel('$\\hat{x}$')
        ax.set_ylabel('$\\hat{y}$')
        ax.set_title(f'$\\hat{{t}} = {t_value:.2f}$')

        ax.set_xlim(np.min(x_limts), np.max(x_limts))
        ax.set_ylim(np.min(y_limts), np.max(y_limts))
        ax.scatter(x_plot+z[:, 0], y_plot+z[:, 1], c=norm, cmap='viridis')

        return ax

    n_frames = len(t_raw)
    ani = FuncAnimation(fig, update, frames=n_frames,
                        fargs=(x, y, x_plot, y_plot, t_raw, t_shaped, pinn, ax), interval=100, blit=False)

    file = f'{path}/sol_time_comparison.gif'
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
    ax[1].set_ylabel('$\\overline{u}_\\text{y,an}-\\overline{u}_\\text{y,PINN}$')

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
