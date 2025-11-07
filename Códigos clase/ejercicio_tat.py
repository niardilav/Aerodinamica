import numpy as np
from scipy.integrate import quad

alpha = -4*np.pi/180

# Definir la función f(t)
int_1 = lambda t: ( (0.25*0.8) - 0.5*(0.5*(1-np.cos(t))) )*(np.cos(t) - 1)
int_2 = lambda t: 0.111*( 0.8 - 2*(0.5*(1-np.cos(t))) )*(np.cos(t) - 1)
int_3 = lambda t: (2/np.pi)*( (0.25*0.8) - 0.5*(0.5*(1-np.cos(t))) )*np.cos(t)
int_4 = lambda t: (2/np.pi)*0.111*( 0.8 - 2*(0.5*(1-np.cos(t))) )*np.cos(t)
int_5 = lambda t: (2/np.pi)*( (0.25*0.8) - 0.5*(0.5*(1-np.cos(t))) )*np.cos(2*t)
int_6 = lambda t: (2/np.pi)*0.111*( 0.8 - 2*(0.5*(1-np.cos(t))) )*np.cos(2*t)

# Realizar la integración: int_a^b f(t)
integral_1, err_1 = quad(int_1, 0, 1.37)
integral_2, err_2 = quad(int_2, 1.37, np.pi)
integral_3, err_3 = quad(int_3, 0, 1.37)
integral_4, err_4 = quad(int_4, 1.37, np.pi)
integral_5, err_5 = quad(int_5, 0, 1.37)
integral_6, err_6 = quad(int_6, 1.37, np.pi)

alpha_L0 = -(integral_1 + integral_2) / np.pi
A_1 = integral_3 + integral_4
A_2 = integral_5 + integral_6
cl = 2*np.pi*(alpha - alpha_L0)
c_m = (np.pi/4) * (A_2 - A_1)

x_cp = 0.25*( 1 + np.pi*(A_1 - A_2)/cl )

print(f"Angulo de ataque L=0 = {alpha_L0*180/np.pi:.2f}")
print(f"cl = {cl:.4f}")
print(f"c_m a c/4 = {c_m:.4f}")
print(f"x_cp/c = {x_cp:.4f}")

