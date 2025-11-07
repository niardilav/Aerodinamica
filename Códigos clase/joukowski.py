import numpy as np
import matplotlib.pyplot as plt

b = 1
a = 1.2*b
beta = 10 * np.pi/180
theta = np.linspace(0, 2 * np.pi, 360)
Z_c = b - a * np.exp(-1j* beta) # centro del circulo en el plano zeta
Z_circ = Z_c + a * np.exp(1j * theta)
#Z_circ = a * np.exp(1j * theta) # circulo centrado en el origen
z = Z_circ + b**2 / Z_circ # Transformaci´on de Joukowski
V_inf = 1
alpha = 0 * np.pi / 180
gamma = 4 * np.pi * a * V_inf * np.sin(alpha + beta)
# Generaci´on de la rejilla para los contornos
n = 500
chi, eta = np.meshgrid(np.linspace(-4.5 * b, 4.5 * b, n), np.linspace(-3.5 * b, 3.5 * b, n))
zeta = chi + 1j * eta # Coordenada compleja zeta
# Excluir puntos al interior del circulo
toll = 1e-4
mask = np.abs(zeta - Z_c) <= a - toll
zeta[mask] = np.nan
#
F = V_inf * zeta * np.exp(-1j * alpha) + V_inf * np.exp(1j * alpha) * a**2 / ( zeta - Z_c ) + 1j * gamma * np.log( zeta - Z_c ) / (2*np.pi)   # Potencial complejo
Z = zeta + b**2 / zeta # transformacion al plano z
#
plt.figure(1)# contornos en el plano zeta
plt.contour(zeta.real, zeta.imag, F.imag, levels=np.arange(-10, 10.25, 0.25), colors='blue')
plt.contour(zeta.real, zeta.imag, F.real, levels=np.arange(-10, 10.25, 0.25), colors='red')
plt.plot(Z_circ.real, Z_circ.imag, 'k', linewidth=2.5)# Cilindro
plt.xlim(-4.5*b, 4.5*b)
plt.ylim(-3.5*b, 3.5*b)
plt.figure(2)# contornos en el plano z
plt.contour(Z.real, Z.imag, F.imag, levels=np.arange(-10, 10.1, 0.1), colors='blue')
#plt.contour(Z.real, Z.imag, F.real, levels=np.arange(-10, 10.25, 0.25), colors='red')
plt.plot(z.real, z.imag, 'k', linewidth=2.5)# Placa horizontal
plt.xlim(-4.5*b, 4.5*b)
plt.ylim(-3.5*b, 3.5*b)
plt.show()
