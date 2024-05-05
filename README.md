# EvolBERNPINN2D
Implementation of PINN for the solution of an evolutive hyperbolic problem of a Bernoulli beam
## Usage
- all parameters regarding the model (material, NN configurations) are defined in par.py
- there's one standard NN that is used to pretrain the PINN, controlled by retrain_init bool
- retrain both NNs every time parameters in par.py are changed
- script just write pictures in the various folders; don't expect to pop up during execution!
- main script to run: main.py