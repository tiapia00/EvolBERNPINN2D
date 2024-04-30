# EvolBERNPINN2D
Implementation of PINN for the solution of an evolutive hyperbolic problem of a Bernoulli beam
## Usage
- all parameters of material nn_init and PINN should be defined in the par.j2 file
In this way, consistent initialization of both models, unique database
## Known problems
- plots-> plot_solution not working
- initial conditions do not converge
- quadrature rule not really defined
## To be implemented
### short term
- ~separate curriculum training into its own function~
- implement plot in time
- ~material parameters definition directly in JSON file: improve consistency~
- check consistency in terms of measurement units for both models
### long term
- comparison with closed-form analytical solutions
- try to implement positional embeddings
## File status:
- Analytical_w works but it is obsolete
