# EvolBERNPINN2D
Implementation of PINN for the solution of an evolutive hyperbolic problem of a Bernoulli beam
## Usage
- all parameters regarding the model (material, NN configurations) are defined in par.py
In this way, consistent initialization of both models, unique database
- there's one standard NN that is used to pretrain the PINN, controlled by retrain_init bool
- main script: main.py