import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Coordenadas del perfil: https://m-selig.ae.illinois.edu/ads/coord_database.html#E
# Activar la linea para leer las coordenadas del perfil
# Coordenadas del Extrados e Intrados como dos archivos separados
# verifique que los archivos esten en la misma carpeta del script
airfoil_intrados = np.loadtxt('e433_intrados.txt')
airfoil_extrados = np.loadtxt('e433_extrados.txt')
# Acomodar las coordenadas al formato requerido:
# Las coordenadas del perfil deben iniciar en el borde de fuga (TE) continuar
# por el intrados hasta el borde de ataque (LE) y regresar al TE por el extrados
# de tal manera que exista una sola posición en x para los punto del intrados y el extrados
# Intrados
Xd = airfoil_intrados[:, 0]
Yd = airfoil_intrados[:, 1]
X_int = np.flipud(Xd)
Y_int = np.flipud(Yd)
# Extrados
Xu = airfoil_extrados[:, 0]
Yu = airfoil_extrados[:, 1]
# Funcion de interpolacion
interp_func = interp1d(Xu, Yu, kind='linear', fill_value='extrapolate')
# Interpolacion para buscar los puntos Xd en el extrados
X_ext = Xd[1:]
Y_ext = interp_func(X_ext)
# Coordenadas sobre el perfil que serán los extremos de los paneles
XB = np.concatenate((X_int, X_ext))
YB = np.concatenate((Y_int, Y_ext))

# Ejemplo de coordenadas (XB, YB)
#XB = np.array([1.0, 0.933, 0.750, 0.500, 0.250, 0.067, 0.0, 0.067, 0.250, 0.500, 0.750, 0.933, 1.0], dtype=np.float64)
#YB = np.array([0.0, -0.005, -0.017, -0.033, -0.042, -0.033, 0.0, 0.041, 0.076, 0.072, 0.044, 0.013, 0.0], dtype=np.float64)

# Longitud de la cuerda (c)
c = XB[0] - np.min(XB)

# PARAMETRO DE ENTRADA : ANGULO DE ATAQUE
ALPHA = 7  # Angulo de ataque en grados
MP1 = XB.shape[0]  # m+1 puntos verdes
M = MP1 - 1  # m paneles o puntos de control azules
ALPHA1 = ALPHA * np.pi / 180  # Convertir a radianes

# Iniciaizar arreglos con precision np.float64
gamma = np.zeros(MP1, dtype=np.float64)
RHS = np.zeros(MP1, dtype=np.float64)

# Inicializar matrices de coeficientes con precision np.float64
# Misma notacion que la utilizada en la descripcion del metodo de lo paneles de vortice
CN1 = np.zeros((M, M), dtype=np.float64)
CN2 = np.zeros((M, M), dtype=np.float64)
CT1 = np.zeros((M, M), dtype=np.float64)
CT2 = np.zeros((M, M), dtype=np.float64)

# Inicializacion de otros coeficientes
AN = np.zeros((MP1, MP1), dtype=np.float64)
AT = np.zeros((M, MP1), dtype=np.float64)
THETA = np.zeros(M, dtype=np.float64)
X = np.zeros(M, dtype=np.float64)
Y = np.zeros(M, dtype=np.float64)
S = np.zeros(M, dtype=np.float64)
SX = np.zeros(M, dtype=np.float64)
SY = np.zeros(M, dtype=np.float64)
SINE = np.zeros(M, dtype=np.float64)
COSINE = np.zeros(M, dtype=np.float64)

# Calculo de los puntos de control y la longitud de los paneles
for i in range(M):
    ip1 = i + 1
    X[i] = 0.5 * (XB[i] + XB[ip1])  # Punto medio de XB
    Y[i] = 0.5 * (YB[i] + YB[ip1])  # Punto medio de YB
    SX[i] = XB[ip1] - XB[i]  # Longitud del panel en X
    SY[i] = YB[ip1] - YB[i]  # Longitud del panel en Y
    S[i] = np.sqrt(SX[i]**2 + SY[i]**2)  # Longitud del Panel
    THETA[i] = np.arctan2(SY[i], SX[i])  # Orientacion del Panel
    SINE[i] = np.sin(THETA[i])
    COSINE[i] = np.cos(THETA[i])
    RHS[i] = np.sin(THETA[i] - ALPHA1)  # Termino a la derecha en el sistema de ecuaciones

# Calculo de los coeficientes de influencia (CN1, CN2, CT1, CT2)
# XTEMP y YTEMP son variables auxiliares para calcular diferencias
for i in range(M):
    for j in range(M):
        if i == j:
            CN1[i, j] = -1.0
            CN2[i, j] = 1.0
            CT1[i, j] = np.pi / 2
            CT2[i, j] = np.pi / 2
        else:
            XTEMP = X[i] - XB[j]  # Differencia en X
            YTEMP = Y[i] - YB[j]  # Differencia en Y
            A = -XTEMP * COSINE[j] - YTEMP * SINE[j]
            B = XTEMP**2 + YTEMP**2
            TTEMP = THETA[i] - THETA[j]
            C = np.sin(TTEMP)
            D = np.cos(TTEMP)
            E = XTEMP * SINE[j] - YTEMP * COSINE[j]
            F = np.log(1 + S[j] * (S[j] + 2 * A) / B)
            G = np.arctan2(E * S[j], B + A * S[j])
            TTEMP -= THETA[j]
            P = XTEMP * np.sin(TTEMP) + YTEMP * np.cos(TTEMP)
            Q = XTEMP * np.cos(TTEMP) - YTEMP * np.sin(TTEMP)
            CN2[i, j] = D + Q * F / (2 * S[j]) - (A * C + D * E) * G / S[j]
            CN1[i, j] = D * F / 2 + C * G - CN2[i, j]
            CT2[i, j] = C + P * F / (2 * S[j]) + (A * D - C * E) * G / S[j]
            CT1[i, j] = C * F / 2 - D * G - CT2[i, j]

# Calculo de los coeficientes de influencia en las matrices AN y AT
for i in range(M):
    AN[i, 0] = CN1[i, 0]
    AN[i, MP1 - 1] = CN2[i, M - 1]
    AT[i, 0] = CT1[i, 0]
    AT[i, MP1 - 1] = CT2[i, M - 1]
    for j in range(1, M):
        AN[i, j] = CN1[i, j] + CN2[i, j - 1]
        AT[i, j] = CT1[i, j] + CT2[i, j - 1]

# Modificacion de la ultima fila de AN para satisfacer la condicion de Kutta
AN[MP1 - 1, 0] = 1.0
AN[MP1 - 1, MP1 - 1] = 1.0
AN[MP1 - 1, 1:M] = 0.0

# Termino de la derecha
RHS[:M] = np.sin(THETA - ALPHA1)
RHS[MP1 - 1] = 0.0

# Se resuelve el sistema para encontrar la distribucion de vortices en cada panel (gamma)
gamma = np.linalg.solve(AN, RHS)

# Calculo de la velocidad y del coeficiente de presion
V = np.cos(THETA - ALPHA1) + np.dot(AT, gamma)
CP = 1.0 - V**2

# Calculo de la circulacion del perfil alar
Gama_airfoil = np.dot(S, (gamma[:M] + gamma[1:MP1]) / 2)

# Coeficiente de sustentacion del perfil alar (Cl)
cl = 4 * np.pi * Gama_airfoil / c

# Coeficiente de momento respecto al centro aerodinamico (cmac)
# Se asume ubicado al cuarto de cuerda
XAC = np.min(XB) + c / 4
cmac = np.dot(S, CP * ((X - XAC) * COSINE + Y * SINE) / c**2)

# Resultados
print(f"Coeficiente de sustentacion (Cl): {cl:.4f}")
print(f"Coeficiente de momento en el ca (c_mca): {cmac:.4f}")
# Grafica de los paneles que aproximan al perfil
plt.figure(1)
plt.plot(XB, YB, 'k-', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Geometria perfil alar')
plt.xlim([0, 1])
plt.ylim([-0.5, 0.5])

# Grafica de Cp
plt.figure(2)
plt.plot(X, CP, 'b-', linewidth=2, label=fr'$\alpha$={ALPHA}°')
plt.gca().invert_yaxis()
plt.xlabel('x')
plt.ylabel(r'$C_p$')
plt.legend()
plt.title('Distribucion de presion sobre el perfil alar')
plt.grid(True)
plt.show()


