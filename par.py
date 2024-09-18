class Parameters:
    def __init__(self):
        self.x_end = 500 
        self.y_end = 0.5
        self.t_end = 0.3
        self.n_space = 10
        self.n_time = 10
        self.dimhid = 3 
        self.dmodel = 1 
        self.nheads = 1 
        self.nblocks = 1 
        self.w0 = 0.3
        self.pinn_par = {
            'x_end': self.x_end,
            'y_end': self.y_end,
            't_end': self.t_end,
            'n_space': self.n_space,
            'n_time': self.n_time,
            'w0': self.w0,
            'dmodel': self.dmodel,
            'dimhid': self.dimhid,
            'nheads': self.nheads,
            'nblocks': self.nblocks,
            'lr': 1e-4,
            'epochs': int(4e3) 
        }
        self.beam_par = {
            'x_end': self.x_end,
            't_end': self.t_end,
            'h': self.y_end,
            'n_beam': 2000,
            'ntime': 1000,
            'w0': self.w0
        }
        self.mat_par = {
            'E': 68.0e3,
            'rho': 7e-9,
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
