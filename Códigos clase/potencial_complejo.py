import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Parametros
U = 1.0
m = 2
K = 2
R = 1
L = U * R**2
A = 1
n = 2
x_min, x_max = -2, 12
y_min, y_max = -4, 4
n_points = 100

# Grid
x = np.linspace(x_min, x_max, n_points)
y = np.linspace(y_min, y_max, n_points)
X, Y = np.meshgrid(x, y)

# Variable Compleja
Z = X + 1j * Y
f = A * Z ** n
#f = U*Z + L / Z

# Potencial de velocidad (parte real) y funcion de corriente (parte imaginaria)
phi = np.real(f)   # PHI
psi = np.imag(f)   # PSI

# Graficas
plt.figure(figsize=(8, 8))

# Graficar PHI
contour_phi = plt.contour(X, Y, phi, levels=np.arange(-5, 5.1, 0.1), colors='red', linestyles='dashed')
#plt.clabel(contour_phi, inline=1, fontsize=10, fmt=r"$\phi=$.1f")

# Graficar PSI
contour_psi = plt.contour(X, Y, psi, levels=np.arange(-5, 5.1, 0.1), colors='blue')
#plt.clabel(contour_psi, inline=1, fontsize=10, fmt=r"$\psi=$%.1f")

# Proxy para la leyenda
phi_proxy = Line2D([0], [0], color='red', linestyle='dashed', label=r'Líneas de potencial, $\phi$')
psi_proxy = Line2D([0], [0], color='blue', linestyle='solid', label=r'Líneas de corriente, $\psi$')

plt.legend(handles=[phi_proxy, psi_proxy], loc='upper right')


# Ejes
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect('equal')
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()
