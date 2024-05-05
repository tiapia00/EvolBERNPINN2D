import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import integrate
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class Beam:   
  def __init__(self, length, E, rho, H, b, n_points):
      self.length = length # m
      self.E = E # Nm^2
      self.J = H**3*b/12
      self.rho = rho # kg/m
      self.H = H
      self.b = b
      self.A = self.H*self.b
      self.w : np.ndarray
      self.gamma : np.ndarray
      self.omega : np.ndarray
      self.xi = np.linspace(0, self.length, n_points)
      self.eps : np.ndarray
      self.phi : np.ndarray

  def update_freq(self):
    self.omega = (self.gamma**4/(self.rho*self.A)*self.E*self.J)**(1/2)

  def update_phi(self, phi:np.ndarray):
    self.phi = phi

  def return_modemat_eig(self, F:np.ndarray, gamma):
    return 1*np.sin(gamma*self.xi)+F[0]*np.cos(gamma*self.xi)+F[1]*np.sinh(gamma*self.xi)+F[2]*np.cosh(gamma*self.xi)

  def return_modemat(self):
    i = 0
    self.phi = np.zeros((len(self.xi), len(self.gamma)))
    for gamma in self.gamma:
        self.phi[:, i] = self.return_modemat_eig(F[i], gamma)
        i += 1
  def normalize_modeshapes(self):
    for i in np.arange(self.phi.shape[1]):
      M = integrate.simpson(y=self.phi[:,i]**2, x=self.xi)
      self.phi[:,i] = self.phi[:,i]/M

# Plots
  def plot_modes(self):
    # Notice that just the middle axis of the beam is plotted
    fig, axs = plt.subplots(self.phi.shape[1])
    self.normalize_modeshapes()
    for i in np.arange(self.phi.shape[1]):
      axs[i].plot(self.xi, self.phi[:,i], linewidth=4)
      axs[i].set_title(f"$\\omega = {self.omega[i]:.{3}f}$ rad/s")
      axs[i].set_xlabel("$\\xi$")
      axs[i].set_ylabel("$w$")
    plt.tight_layout()
    plt.show()

# Strains at z=-1
  def plot_strains(self):
    fig, axs = plt.subplots(self.phi.shape[1])
    for i in np.arange(self.phi.shape[1]):
      self.eps = np.gradient(np.gradient(self.phi[:-1,i], self.xi[:-1]), self.xi[:-1])
      axs[i].plot(self.xi[:-1], self.eps, linewidth=4)
      axs[i].set_title(f"$\\omega = {self.omega[i]:.{3}f}$ rad/s")
      axs[i].set_xlabel("$\\xi$")
      axs[i].set_ylabel("$\\varepsilon_1$")
    plt.tight_layout()
    plt.show()

  def calculate_solution(self, A, B, t:np.ndarray):
    self.w = np.zeros((self.phi.shape[0], len(t)))
    j = 0
    for t_s in t:
      w = np.zeros(self.phi.shape[0])
      i = 0
      for i in np.arange(self.phi.shape[1]):
        w_mode_i = self.phi[:,i]*(A[i]*np.cos(self.omega[i]*t_s)+B[i]*np.sin(self.omega[i]*t_s))
        w = w + w_mode_i
      self.w[:,j] = w
      j += 1

  def plot_sol(self, path):

    fig = plt.figure()
    ax = plt.axes()

    def drawframe(i):
        ax.clear()

        plt.title("Solution")
        ax.set_xlim(0, np.max(self.xi))
        ax.set_ylim(np.min(self.w), np.max(self.w))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$w$')
        ax.plot(self.xi, self.w[:, i], color='blue')
        return ax

    ani = animation.FuncAnimation(fig, drawframe, frames=self.w.shape[1], blit=False, repeat=True)
    
    file = f'{path}/sol_analytic.gif'
    ani.save(file, fps=60)

  class modal_appr:
    def __init__(self):
      self.m : np.ndarray
      self.k : np.ndarray
      self.idx : int
      self.xi_L : float
      self.F : float
      self.omega : float
      self.Q : np.ndarray

    def calculate_beam_mat(self):
      def calculate_m(self):
        m = []
        for i in np.arange(self.phi.shape[1]):
          phi = self.phi[:,i]
          m.append(integrate.simpson(y=phi**2, x=self.xi))
        m = 1/2*self.rho*np.array(m)
        m = np.diag(m)
        return m
      def calculate_k(self):
        k = []
        for i in np.arange(self.phi.shape[1]):
          phi = self.phi[:,i]
          ddw_phi = np.gradient(np.gradient(phi, self.xi), self.xi)
          k.append(integrate.simpson(y=ddw_phi**2, x=self.xi))
        k = 1/2*self.E*self.J*np.array(k)
        k = np.diag(k)
        return k
      self.m = calculate_m()
      self.k = calculate_k()

    def def_load(self, xi, F, omega):
    # Find index of nearest point
      nearest_index = np.abs(self.xi - xi).argmin()
      self.xi_L = self.xi[nearest_index]
      self.idx = nearest_index
      self.F = F
      self.omega = omega
    def calculate_Q(self):
      phi_1 = self.phi[self.idx,:]
      self.Q = self.F*phi_1

    def solve(self):
      def initial_cond_to_q(self):
        pass
      pass

class Prob_Solv_Modes:
  def __init__(self, my_beam : Beam):
    self.L = my_beam.length
    self.g_max : float
    self.xi = my_beam.xi
    self.gamma : np.ndarray

  def return_H(self, gm):
    L = self.L
    return np.array([[0, 1, 0, 1], [0, -1, 0, 1], [np.sin(gm*L), np.cos(gm*L), np.sinh(gm*L), np.cosh(gm*L)],
     [-np.sin(gm*L), -np.cos(gm*L), np.cosh(gm*L), np.sinh(gm*L)]])

  def return_det(self, H):
    return np.linalg.det(H)

  def update_gamma(self, my_beam : Beam):
    self.gamma = my_beam.gamma

  def pass_g_max(self, gm):
    self.g_max = gm

  def find_eig(self):
    gamma_loop = np.linspace(0, self.g_max, 100000) # Put high otherwise no convergence
    gm = []
    H = self.return_H(gamma_loop[0])
    deter = self.return_det(H)
    i = 0
    for gm_try in gamma_loop[1:]:
      H = self.return_H(gm_try)
      deter_i = self.return_det(H)
      if (deter > 0 and deter_i <0) or (deter < 0 and deter_i > 0):
        gm.append((gamma_loop[i-1]+gamma_loop[i])/2)
      deter = deter_i
      i += 1
    return np.array(gm)

  def find_F_eig(self, H):
  # delete first row and column
    #print(np.linalg.det(H))
    H = np.delete(H, 0, axis=0)
    N = H[:,0]
    h = np.delete(H, 0, axis=1)
    return np.linalg.solve(h, -N)

  def find_all_F(self, my_beam : Beam):
    F = []
    for i in np.arange(len(my_beam.gamma)):
      #print(self.return_H(my_beam.gamma[i]))
      F.append(self.find_F_eig(self.return_H(my_beam.gamma[i])))
    return F

  def return_modemat_eig(self, F:np.ndarray, gamma):
    return 1*np.sin(gamma*self.xi)+F[0]*np.cos(gamma*self.xi)+F[1]*np.sinh(gamma*self.xi)+F[2]*np.cosh(gamma*self.xi)

  def return_modemat(self, F):
    i = 0
    phi = np.zeros((len(self.xi), len(self.gamma)))
    for gamma in self.gamma:
        phi[:, i] = self.return_modemat_eig(F[i], gamma)
        i += 1
    return phi

class In_Cond:
  def __init__(self, my_beam : Beam):
    self.__w_0 : np.ndarray
    self.__w_dot_0 : np.ndarray
    self.phi = my_beam.phi
    self.L = my_beam.length
    self.omega = my_beam.omega
    self.xi = my_beam.xi

  def pass_init_cond(self, w_0, w_dot_0):
    self.__w_0 = w_0
    self.__w_dot_0 = w_dot_0

  def compute_coeff(self):
    A = []
    B = []
    for i in np.arange(self.phi.shape[1]):
        A.append(2/self.L*integrate.simpson(y=self.__w_0*self.phi[:, i], x=self.xi))
        B.append(2/(self.L*self.omega[i])*integrate.simpson(y=self.__w_dot_0*self.phi[:, i], x=self.xi))
    return A, B