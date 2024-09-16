class Parameters:
    def __init__(self):
        self.x_end = 500 
        self.y_end = 0.5
        self.t_end = 0.8
        self.n_space = 31 
        self.n_time = 43 
        self.dim_hidden = 2 
        self.n_hidden : int = 2 
        self.w0 = 0.3
        self.pinn_par = {
            'x_end': self.x_end,
            'y_end': self.y_end,
            't_end': self.t_end,
            'n_space': self.n_space,
            'n_time': self.n_time,
            'w0': self.w0,
            'dim_hidden': self.dim_hidden,
            'n_hidden_space': self.n_hidden,
            'lr': 1e-3,
            'epochs': 2000
        }
        self.beam_par = {
            'x_end': self.x_end,
            't_end': self.t_end,
            'h': self.y_end,
            'n_space': 10,
            'n_time': self.n_time,
            'w0': self.w0
        }
        self.mat_par = {
            'E': 68.0e3,
            'rho': 7e-9,
            'nu': 0.26
        }

    def to_matpar_PINN(self) -> float:

        E = self.mat_par['E']
        nu = self.mat_par['nu']
        lam = E*nu/(1+nu)/(1-2*nu)
        mu = E/2/(1+nu)

        return lam, mu


def get_params(par_nn: dict) -> tuple:
    values = []
    for _, value in par_nn.items():
        values.append(value)
    values = tuple(values)
    return values
