import torch
import sys

class Simps_Cub:
    def __init__(self, n_train: int, step: float):
        self.n_train = n_train
        self.h = step
        self.w = self.initialize_weights()
        
    def initialize_weights(self) -> torch.tensor:
        
        try:
            w = torch.ones(self.n_train)
            w[0] = 1/3
            w[-1] = 1/3

            for i in range(1, w.size(0)-1):
                if i % 2 == 0:
                    w[i] = 2/3
                else:
                    w[i] = 4/3

            w_mesh = torch.zeros((self.n_train, self.n_train, self.n_train))
            for k in range(w_mesh.size(2)):
                for j in range(w_mesh.size(1)):
                    for i in range(w_mesh.size(0)):
                        w_mesh[i, j, k] = w[i]*w[j]*w[k]
                        
            if self.n_train % 2 == 0:
                raise ValueError("Even number of points")
                
        except ValueError as ve:
            print("Value error in integrator:", ve)
            sys.exit(1)
            
        return w_mesh
        
        
    def integrate(self, output: torch.tensor) -> float:
        """output: sampled values of the function to integrate,
            having built a mesh with weights, now we can just sum all points multiplied by the
            corresponding quadrature weights"""

        output = output.reshape(self.n_train, self.n_train, self.n_train)
        integral = 0

        for k in range(output.size(2)):
            for j in range(output.size(1)):
                for i in range(output.size(0)):
                    integral = integral + self.w[i, j, k]*output[i, j, k]
                        
        integral = self.h**3*integral 
        # notice the power of the step
                    
        return integral