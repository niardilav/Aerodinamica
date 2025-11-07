import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Propiedades del fluido
mu = 1.78e-5 #0.05
rho_f = 1.22 #920
# Propiedades de la esfera
d = 5e-3
rho_s = 2500
gr = 9.81
tau_r = rho_s * d ** 2 / (18*mu)

def equation(V):
 
    Re = rho_f * V * d / mu

    if Re < 1000:
        CD = (24/Re) * (1+0.15*Re) ** 0.687
    else:
        CD = 0.44
    return V - 24*tau_r*gr*(1 - rho_f/rho_s) / (CD*Re)

# Suposicion inicial de velocidad (m/s)
V_guess = 10
V_solution = fsolve(equation, V_guess)[0]
Re = rho_f * V_solution * d / mu
print(f"Velocidad Terminal: {V_solution:.2f} m/s")
print(f"Re: {Re:.0f}")
print(f"Tiempo de Respuesta: {tau_r:0.4f} s")


