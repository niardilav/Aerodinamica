import numpy as np
import matplotlib.pyplot as plt

# Coordenadas x, y
x = np.arange(-4.0, 4.01, 0.01)
y = np.arange(-4.0, 4.01, 0.01)

m = 1
a = 1

# Meshgrid
X, Y = np.meshgrid(x, y)

# Funcion de corriente
psi = -m * np.arctan( (2 * a * Y) / (X**2 + Y**2 - a**2) )

# Funcion potencial
phi =  (m / 2) * np.log( ((X + a)**2 + Y**2) / ((X - a)**2 + Y**2) )

# Lineas de corriente (blue)
plt.figure(figsize=(8, 6))
C1 = plt.contour(X, Y, psi, colors='blue')
plt.clabel(C1, inline=True, fontsize=8)

# Lineas de potencial (red)
C2 = plt.contour(X, Y, phi, levels = np.arange(-1,1,0.1), colors='red')

# Ejes
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.grid(True)
plt.show()
