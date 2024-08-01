class Parameters:
    def __init__(self):
        self.x_end = 4
        self.y_end = 1e-1
        self.t_end = 0.1 
        self.n_space = (40, 40)
        self.n_time = 40
        self.dim_hidden = 20 
        self.dim_mult = (1,1)
        self.n_hidden_t : int = 5
        self.w0 = 0.3
        self.pinn_par = {
            'x_end': self.x_end,
            'y_end': self.y_end,
            't_end': self.t_end,
            'n_space': self.n_space,
            'n_time': self.n_time,
            'w0': self.w0,
            'dim_hidden_t': self.dim_hidden,
            'nlayers_t': self.n_hidden_t, 
            'lr': 8e-4,
            'epochs': 3000 
        }
        self.beam_par = {
            'x_end': self.x_end,
            't_end': self.t_end,
            'h': self.y_end,
            'w0': self.w0
        }
        self.mat_par = {
            'E': 68.0e9,
            'rho': 27.,
            'nu': 0.26
        }

    def to_matpar_PINN(self) -> float:

        E = self.mat_par['E']/1.e6
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
