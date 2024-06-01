import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, FuncFormatter
import torch
from matplotlib.animation import FuncAnimation
from pinn import PINN, f
import numpy as np


def scatter_penalty_loss2D(x: torch.tensor, y: torch.tensor, n_train: int, factors: torch.tensor):
    x = x.reshape(n_train, n_train).detach().cpu().numpy()
    y = y.reshape(n_train, n_train).detach().cpu().numpy()
    factors = factors.reshape(n_train, n_train).detach().cpu().numpy()

    fig = plt.figure()
    plt.scatter(x, y, c=factors, cmap=viridis)
    plt.colorbar()

    plt.xlabel('$\\hat{x}$')
    plt.ylabel('$\\hat{y}$')

    fig.canvas.draw()

    image_np = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    return image_tensor


def scatter_penalty_loss3D(x: torch.tensor, y: torch.tensor, t: torch.tensor, n_train: int, factors: torch.tensor):
    nx = n_train-2
    ny = nx
    nt = n_train-1

    x = x.reshape(nx, ny, nt).detach().cpu().numpy()
    y = y.reshape(nx, ny, nt).detach().cpu().numpy()
    t = t.reshape(nx, ny, nt).detach().cpu().numpy()
    factors = factors.reshape(nx, ny, nt).detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, t, c=factors, cmap=viridis)

    ax.set_xlabel('$\\hat{x}$')
    ax.set_ylabel('$\\hat{y}$')
    ax.set_zlabel('$\\hat{t}$')

    cbar = fig.colorbar(sc, ax=ax)

    fig.canvas.draw()
    plt.tight_layout()

    image_np = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

    return image_tensor


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

    a_scatter = ax[0].scatter(X.reshape(-1)+z0[:, 0],
                              Y.reshape(-1)+z0[:, 1], c=norm_z0, cmap=cmap)
    ax[0].set_xlabel('$\\hat{x}$')
    ax[0].set_ylabel('$\\hat{y}$')
    ax[0].set_title('Analytical initial conditions')
    cbar1 = fig.colorbar(a_scatter, ax=ax[0], orientation='vertical')
    cbar1.set_label('$|\\mathbf{u}|$')

    p_scatter = ax[1].scatter(X.reshape(-1)+z[:, 0],
                              Y.reshape(-1)+z[:, 1], c=norm_z, cmap=cmap)
    ax[1].set_xlabel('$\\hat{x}$')
    ax[1].set_ylabel('$\\hat{y}$')
    ax[1].set_title('Predicted initial conditions')
    cbar2 = fig.colorbar(p_scatter, ax=ax[1], orientation='vertical')
    cbar2.set_label('$|\\mathbf{u}|$')

    plt.tight_layout()

    plt.savefig(f'{path}/init.png')


def plot_sol_comparison(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, w_ad: np.ndarray, n_train:
                        int, path: str, device):

    nx = n_train - 2
    ny = nx
    nt = n_train - 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    t_raw = torch.unique(t, sorted=True)
    t_raw = t_raw.reshape(-1, 1)

    x_raw = x.reshape(nx, ny, nt)
    y_raw = y.reshape(nx, ny, nt)

    x = x_raw[:, :, 0]
    y = y_raw[:, :, 0]

    x = x.reshape(-1, 1).to(device)
    y = y.reshape(-1, 1).to(device)

    t_shaped = torch.ones_like(x)
    t = t_shaped*t_raw[0].to(device)

    output = f(pinn, x, y, t)

    x_plot = x.cpu().detach().numpy().reshape(nx, ny).reshape(-1)
    y_plot = y.cpu().detach().numpy().reshape(nx, ny).reshape(-1)

    z0 = output.cpu().detach().numpy()
    norm = np.linalg.norm(z0, axis=1).reshape(-1)

    ax.scatter(x_plot+z0[:, 0], y_plot+z0[:, 1], c=norm, cmap='viridis')
    ax.scatter(np.unique(x_plot), w_ad[1:-1, 0])
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
        t = t_shaped*t_raw[frame]

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
        ax.scatter(np.unique(x_plot), w_ad[1:-1, frame])

        return ax

    n_frames = len(t_raw)
    ani = FuncAnimation(fig, update, frames=n_frames,
                        fargs=(x, y, x_plot, y_plot, t_raw, t_shaped, pinn, ax), interval=100, blit=False)

    file = f'{path}/sol_time_comparison.gif'
    ani.save(file, fps=5)


def plot_sol(pinn: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, n_train:
             int, path: str, device):

    nx = n_train - 2
    ny = nx
    nt = n_train - 1

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    t_raw = torch.unique(t, sorted=True)
    t_raw = t_raw.reshape(-1, 1)

    x_raw = x.reshape(nx, ny, nt)
    y_raw = y.reshape(nx, ny, nt)

    x = x_raw[:, :, 0]
    y = y_raw[:, :, 0]

    x = x.reshape(-1, 1).to(device)
    y = y.reshape(-1, 1).to(device)

    t_shaped = torch.ones_like(x)
    t = t_shaped*t_raw[0].to(device)

    output = f(pinn, x, y, t)

    x_plot = x.cpu().detach().numpy().reshape(nx, ny).reshape(-1)
    y_plot = y.cpu().detach().numpy().reshape(nx, ny).reshape(-1)

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
        y_limts = np.array([-0.25, 0.25])
        t = t_shaped*t_raw[frame]

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

    file = f'{path}/sol_time.gif'
    ani.save(file, fps=5)


def plot_midpoint_displ(pinn: PINN, t: torch.Tensor, n_train: int, uy_mid: np.ndarray, path: str, device):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    fig.suptitle('Midpoint displacement')

    t_raw = torch.unique(t, sorted=True)

    x = torch.tensor([0.5]).to(device).reshape(-1, 1)
    y = torch.tensor([0.]).to(device).reshape(-1, 1)

    uy_mid_PINN = []

    for t in t_raw:
        t = t.reshape(-1, 1)
        output = f(pinn, x, y, t)
        uy = output[0, 1].cpu().detach().numpy()
        uy_mid_PINN.append(uy)

    ax[0].plot(t_raw.cpu().detach().numpy(),
               np.array(uy_mid_PINN), color='blue')
    ax[0].set_title('Prediction from PINN')
    ax[0].set_xlabel('$\\hat{t}$')
    ax[0].set_ylabel('$\\hat{u}_y$')

    ax[1].plot(t_raw.cpu().detach().numpy(), uy_mid -
               np.array(uy_mid_PINN), color='red')
    ax[1].set_title('Deviation from analytical')
    ax[1].set_xlabel('$\\hat{t}$')
    ax[1].set_ylabel('$\\hat{u}_\\text{y,an}-\\hat{u}_\\text{y,PINN}$')

    plt.grid()
    plt.tight_layout()

    file = f'{path}/midpoint_time.png'
    plt.savefig(file)


def plot_energy(t: torch.tensor, en_k: torch.tensor, en_p: torch.tensor, en: torch.tensor, path):
    fig = plt.figure(figsize=(10, 8))
    plt.xlabel('t')

    t = t.detach().cpu().numpy()
    en = en.detach().cpu().numpy()
    en_k = en_k.detach().cpu().numpy()
    en_p = en_p.detach().cpu().numpy()

    plt.plot(t, en, label='Total energy')
    plt.plot(t, en_k, label='Kinetic energy')
    plt.plot(t, en_p, label='Potential energy')

    file = f'{path}/energy.png'
    plt.savefig(file)
