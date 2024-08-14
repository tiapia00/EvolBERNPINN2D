class Parameters:
    def __init__(self):
        self.x_end = 5 
        self.y_end = 1e-1
        self.t_end = 2 
        self.n_space = (3, 3)
        self.n_time = 40
        self.dimhid = 2 
        self.dmodel = 4
        self.nheads = 2 
        self.nblocks = 3
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
            'lr_formin': 1e-3,
            'lr_formax': -1e-3,
            'epochs': int(1e3) 
        }
        self.beam_par = {
            'x_end': self.x_end,
            't_end': self.t_end,
            'h': self.y_end,
            'w0': self.w0
        }
        self.mat_par = {
            'E': 68.0e9,
            'rho': 270.,
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
