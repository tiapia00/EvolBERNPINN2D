# Known problems
- plots-> plot_solution not working
- initial conditions do not converge
- quadrature rule not really defined
# To be implemented
## short term
- ~separate curriculum training into its own function~
- ~implement plot in time~
- ~material parameters definition directly in JSON file: improve consistency~
- check consistency in terms of measurement units for both models
- ~check df function for gradient calculation~
- graphs as a function of time to compare analytical and PINN solution (plot midpoint)
- implement shuffling of hyperparameters to select best combination
## long term
- comparison with closed-form analytical solutions
- try to implement positional embedding
- simulation after initial not very realistic: penalize loss till current time step is resolved