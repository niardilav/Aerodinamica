# -*- coding: utf-8 -*-
"""
Simulador 2D (Teoría de Perfil Delgado - TAT)
Usa la línea media analítica de un NACA de 4 cifras.
Autores: Nicolás Ardila y Juan González
"""

import numpy as np
from scipy import interpolate
from dataclasses import dataclass
from typing import Callable


# ===============================================================
# === GENERADOR DE CAMBER LINE PARA NACA DE 4 CIFRAS ============
# ===============================================================

def naca4_camber_line(m: float, p: float, c: float = 1.0, n: int = 200):
    """
    Genera la línea media analítica de un perfil NACA de 4 cifras.
    
    m: camber máximo (fracción de cuerda)
    p: ubicación del camber máximo (fracción de cuerda)
    c: cuerda (1.0 por defecto)
    n: resolución de la discretización
    """
    x = np.linspace(0, c, n)
    z = np.zeros_like(x)

    for i, xi in enumerate(x):
        xbar = xi / c
        if xbar < p:
            z[i] = m / p**2 * (2*p*xbar - xbar**2) * c
        else:
            z[i] = m / (1 - p)**2 * ((1 - 2*p) + 2*p*xbar - xbar**2) * c

    return x, z / c   # Normalizar a cuerda = 1


# ===============================================================
# === CÁLCULO DE COEFICIENTES TAT ===============================
# ===============================================================

@dataclass
class TATCoeffs:
    alpha_L0: float
    cm_c4: float
    cl_alpha: float = 2 * np.pi


def tat_from_camber(dzdx_fun: Callable[[np.ndarray], np.ndarray], n_quad: int = 801) -> TATCoeffs:
    theta = np.linspace(0.0, np.pi, n_quad)
    x = 0.5 * (1 - np.cos(theta))
    dzdx = dzdx_fun(x)

    alpha_L0 = - (1 / np.pi) * np.trapezoid(dzdx * (1 - np.cos(theta)), theta)
    A1 = (2 / np.pi) * np.trapezoid(dzdx * np.cos(theta), theta)
    A2 = (2 / np.pi) * np.trapezoid(dzdx * np.cos(2 * theta), theta)

    cm_c4 = - (np.pi / 4) * (A1 - 0.5 * A2)

    return TATCoeffs(alpha_L0, cm_c4)


# ===============================================================
# ======================= MAIN ==================================
# ===============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # === Definir perfil NACA ===
    # Ejemplo: NACA 2415 → m=0.02, p=0.4
    m = 0.02
    p = 0.4

    x, zc = naca4_camber_line(m, p, n=400)

    # Spline y derivada
    spline = interpolate.CubicSpline(x, zc, bc_type='natural')
    dzdx_fun = spline.derivative()

    # === Calcular coeficientes TAT ===
    tat = tat_from_camber(dzdx_fun)

    alpha_L0_deg = np.rad2deg(tat.alpha_L0)
    print(f"Ángulo de cero sustentación (α_L0) = {alpha_L0_deg:.3f}°")
    print(f"Momento aerodinámico en 1/4 de cuerda, Cm_c/4 = {tat.cm_c4:.5f}")

    # === Graficar C_L vs α ===
    alpha_deg = np.linspace(-14, 14, 200)
    alpha_rad = np.deg2rad(alpha_deg)
    CL = tat.cl_alpha * (alpha_rad - tat.alpha_L0)

    plt.figure(figsize=(7,5))
    plt.plot(alpha_deg, CL, label="C_L(α) – TAT", linewidth=2)
    plt.axvline(alpha_L0_deg, color='r', linestyle='--', label=f"α_L0 = {alpha_L0_deg:.2f}°")

    plt.xlabel("Ángulo de ataque α (°)")
    plt.ylabel("Coeficiente de sustentación C_L")
    plt.title("Curva C_L(α) – Teoría de Perfil Delgado (NACA analítico)")
    plt.grid()
    plt.legend()
    plt.show()
