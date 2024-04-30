def to_matpar_PINN(E: float, nu: float) -> float:
    """ in JSON configuration file, E: Pa
        so, must be converted, because PINN accepts all in MPa: consistency already checked"""
    
    E = E/1.e6
    lam = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    
    return lam, mu