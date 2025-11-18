# -*- coding: utf-8 -*-
"""
Simulador 2D (Teoría de Perfil Delgado - TAT)
Utiliza la línea media del perfil (x, zc) de un archivo .dat
Encuentra:
 - AoA de cero sustentación
 - Cl
 - Cm a 1/4 de cuerda (centro aerodinámico, según la teoría TAT)
Basado en el código utilizado para el proyecto
Autores: Nicolás Ardila y Juan González
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

# ===============================================================
# === LECTURA DE LÍNEA MEDIA (.dat) =============================
# ===============================================================

def load_camber_line(path: str, units_mm: bool = True, flip_vertical: bool = True) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Lee un archivo .dat con la línea media del perfil: columnas X, Y.
    Si units_mm=True, convierte a metros.
    Si flip_vertical=True, invierte el perfil por el eje vertical (zc -> -zc).
    Devuelve:
        x (array), zc (array), zc'(x) (función derivada)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = data[:, 0]
    zc = data[:, 1]

    if units_mm:
        x = x / 1000.0
        zc = zc / 1000.0

    # === Flip vertical ===
    if flip_vertical:
        zc = -zc

    # Normaliza cuerda a [0,1]
    x = (x - x.min()) / (x.max() - x.min())
    spline = interpolate.CubicSpline(x, zc, bc_type='natural')
    dzdx_fun = spline.derivative()

    return x, zc, dzdx_fun



# ===============================================================
# === CÁLCULO DE COEFICIENTES TAT ===============================
# ===============================================================

@dataclass
class TATCoeffs:
    alpha_L0: float
    cm_c4: float
    cl_alpha: float = 2 * np.pi


def tat_from_camber(dzdx_fun: Callable[[np.ndarray], np.ndarray], n_quad: int = 801) -> TATCoeffs:
    """
    Calcula α_L0 y c_m,c/4 según Teoría de Perfil Delgado.
    """
    theta = np.linspace(0.0, np.pi, n_quad)
    x = 0.5 * (1 - np.cos(theta))
    dzdx = dzdx_fun(x)

    alpha_L0 = - (1 / np.pi) * np.trapezoid(dzdx * (1 - np.cos(theta)), theta)
    A1 = (2 / np.pi) * np.trapezoid(dzdx * np.cos(theta), theta)
    A2 = (2 / np.pi) * np.trapezoid(dzdx * np.cos(2 * theta), theta)
    cm_c4 = - (np.pi / 4) * (A1 - 0.5 * A2)

    return TATCoeffs(alpha_L0, cm_c4)

def tat_from_camber(dzdx_fun: Callable[[np.ndarray], np.ndarray], n_quad: int = 801) -> TATCoeffs:
    """
    Calcula α_L0 y c_m,c/4 según Teoría de Perfil Delgado.
    """
    theta = np.linspace(0.0, np.pi, n_quad)
    x = 0.5 * (1 - np.cos(theta))
    dzdx = dzdx_fun(x)

    # === alpha_L0 usando TU versión original (la correcta para tu código) ===
    alpha_L0 = - (1 / np.pi) * np.trapezoid(dzdx * (1 - np.cos(theta)), theta)

    # === coeficientes TAT A1 y A2 ===
    A1 = (2 / np.pi) * np.trapezoid(dzdx * np.cos(theta), theta)
    A2 = (2 / np.pi) * np.trapezoid(dzdx * np.cos(2 * theta), theta)

    cm_c4 = - (np.pi / 4) * (A1 - 0.5 * A2)

    return TATCoeffs(alpha_L0, cm_c4)


# ===============================================================
# ======================= CALCULO ===============================
# ===============================================================

# ===============================================================
# ======================= CALCULO ===============================
# ===============================================================

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt
    

    # === Archivo con la línea media ===
    wing_file = "ala_linea_media.dat"

    # === Cargar línea media ===
    _, _, dzdx_w = load_camber_line(wing_file)

    # === Calcular coeficientes TAT ===
    tat_w = tat_from_camber(dzdx_w)

    # ===========================================================
    # 1) Ángulo de cero sustentación α_L0 (en grados)
    # ===========================================================
    alpha_L0_deg = np.rad2deg(tat_w.alpha_L0)
    print(f"Ángulo de cero sustentación (α_L0) = {alpha_L0_deg:.3f}°")

    # ===========================================================
    # 2) Graficar C_L vs α para α ∈ [-14°, +14°]
    # ===========================================================
    alpha_deg = np.linspace(-14, 14, 200)
    alpha_rad = np.deg2rad(alpha_deg)

    # C_L = 2π (α - α_L0)
    CL = 2 * np.pi * (alpha_rad - tat_w.alpha_L0)

    plt.figure(figsize=(7,5))
    plt.plot(alpha_deg, CL, label="C_L(α) teoría TAT")
    plt.axvline(alpha_L0_deg, color='r', linestyle='--',
                label=f"α_L0 = {alpha_L0_deg:.2f}°")
    plt.xlabel("Ángulo de ataque α (°)")
    plt.ylabel("Coeficiente de sustentación C_L")
    plt.title("Curva C_L(α) según Teoría de Perfil Delgado")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===========================================================
    # 3) Momento en 1/4 de cuerda
    # ===========================================================
    print(f"Momento aerodinámico en 1/4 de cuerda, Cm_c/4 = {tat_w.cm_c4:.5f}")
