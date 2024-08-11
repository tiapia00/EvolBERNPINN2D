import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib.animation import FuncAnimation
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

def plot_energy(indicators_nn: dict, indicators_an: dict, t_nn: torch.tensor, t_beam: np.ndarray, path: str):
    ### Correct: 2 oscillations per period ###
    t_nn = torch.unique(t_nn, sorted=True).detach().cpu().numpy()

    T_nn = indicators_nn['T'].detach().cpu().numpy()
    Pi_nn = indicators_nn['Pi'].detach().cpu().numpy()

    T_an = indicators_an['T']
    Pi_an = indicators_an['V']

    plt.figure()
    plt.plot(t_nn, T_nn, label='$T_{{NN}}$')
    plt.plot(t_nn, Pi_nn, label='$\\Pi_{{NN}}$')
    #plt.plot(t_beam, T_an, label='$T_{{an}}$')
    #plt.plot(t_beam, Pi_an, label='$\\Pi_{{an}}$')
    plt.legend()

    file = f'{path}/energyfinal.png'
    plt.savefig(file)

def plot_centers(coord: torch.tensor, path: str):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    coord = coord.detach().cpu().numpy()

    ax.scatter(coord[:,0], coord[:,1], coord[:,2])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$t$')

    file = f'{path}/centers.png'
    plt.savefig(file)

def plot_deren(dPi: torch.tensor, dT: torch.tensor, t: torch.tensor, path: str):
    t = torch.unique(t, sorted=True).detach().cpu().numpy()

    plt.figure()
    plt.plot(t, dT.cpu().numpy(), label='dT')
    plt.plot(t, dPi.cpu().numpy(), label='dPi')
    plt.xlabel('t')
    plt.legend()
    plt.savefig()

    file = f'{path}/deren.png'
    plt.savefig(file)

def plot_fft(
        f: np.ndarray,
        modPI: np.ndarray,
        angPI: np.ndarray,
        modT: np.ndarray,
        angT: np.ndarray,
        path: str):
    plt.figure()

    plt.subplot(2,1,1)
    plt.plot(f, modPI, label='$|\\X_{{\\Pi}}|$') 
    plt.plot(f, modT, label='$|\\X_{{T}}|$') 
    
    plt.subplot(2,1,2)
    plt.plot(f, angPI, label='$\\varphi(X_{{\\Pi}})$')
    plt.plot(f, angT, label='$\\varphi(X_{{T}})$')

    plt.tight_layout()

    file = f'{path}/fft.png'
    plt.savefig(file)

