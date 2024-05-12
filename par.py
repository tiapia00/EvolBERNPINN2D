class Parameters:
    def __init__(self):
        self.x_end = 2
        self.y_end = 0.2
        self.t_end = 7
        self.n = 60
        self.hid_layers = 2
        self.neurons_per_layer = 70
        self.pinn_par = {
            'x_end' : self.x_end,
            'y_end' : self.y_end,
            't_end' : self.t_end,
            'n' : self.n,
            'hid_layers' : self.hid_layers,
            'neuron_per_layer' : self.neurons_per_layer,
            'lr' : 0.01,
            'epochs' : 1000,
            'weight_in' : 3,
            'weight_bound' : 1
            }
        self.nn_par = {
            'x_end' : self.x_end,
            't_end' : self.t_end,
            'n' : self.n,
            'hid_layers' : self.hid_layers,
            'neuron_per_layer' : self.neurons_per_layer,
            'lr' : 0.001,
            'epochs' : 1000,
            }   
        self.mat_par = {
            'E' : 68.0e9,
            'rho' : 2700.,
            'h' : self.y_end,
            'nu' : 0.26
            }
        
    def to_matpar_PINN(self) -> float:

        E = self.mat_par['E']/1.e6
        nu = self.mat_par['nu']
        lam = E*nu/(1+nu)/(1-2*nu)
        mu = E/2/(1+nu)

        return lam, mu
    
def get_params(par_nn : dict) -> tuple:
    values = []
    for _, value in par_nn.items():
        values.append(value)
    values = tuple(values)
    return values
        