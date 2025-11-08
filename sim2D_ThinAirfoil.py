# -*- coding: utf-8 -*-
"""
Simulador 2D (Teoría de Perfil Delgado - TAT)
Utiliza la línea media del perfil (x, zc) de un archivo .dat
Resuelve el sistema de 6EDOS de la 2da ley de newton para encontrar la trayectoria
Autor: 
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

# ===============================================================
# === LECTURA DE LÍNEA MEDIA (.dat) =============================
# ===============================================================

def load_camber_line(path: str, units_mm: bool = True) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """
    Lee un archivo .dat con la línea media del perfil: columnas X, Y.
    Si units_mm=True, convierte a metros.
    Devuelve:
        x (array), zc (array), zc'(x) (función derivada)
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    x = data[:, 0]
    zc = data[:, 1]

    if units_mm:
        x = x / 1000.0
        zc = zc / 1000.0

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


# ===============================================================
# === ESTRUCTURAS DE DATOS DEL AVIÓN =============================
# ===============================================================

@dataclass
class SurfaceAero:
    S: float
    c: float
    x_c4: float
    tat: TATCoeffs
    CD0: float = 0.01
    k: float = 0.05
    alpha_inc: float = 0.0  # rad


@dataclass
class Aircraft2D:
    m: float
    Iz: float
    rho: float
    wing: SurfaceAero
    tail: SurfaceAero
    downwash: float = 0.0   # rad


# ===============================================================
# === MODELO AERODINÁMICO Y ECUACIONES DE MOVIMIENTO ============
# ===============================================================

def lift_drag(q: float, surf: SurfaceAero, alpha: float):
    CL = surf.tat.cl_alpha * (alpha - surf.tat.alpha_L0)
    CD = surf.CD0 + surf.k * CL**2
    L = q * surf.S * CL
    D = q * surf.S * CD
    return CL, L, D


def aero_forces(ac: Aircraft2D, u: float, v: float, theta: float):
    V = np.hypot(u, v)
    if V < 1e-6:
        return 0, 0, 0

    q = 0.5 * ac.rho * V**2
    gamma = np.arctan2(v, u)
    alpha_body = theta - gamma
    t_hat = np.array([u, v]) / V
    n_hat = np.array([-t_hat[1], t_hat[0]])

    # --- Ala ---
    alpha_w = alpha_body + ac.wing.alpha_inc
    _, Lw, Dw = lift_drag(q, ac.wing, alpha_w)
    Fw = -Dw * t_hat + Lw * n_hat
    Mw = q * ac.wing.S * ac.wing.c * ac.wing.tat.cm_c4 + ac.wing.x_c4 * Fw[1]

    # --- Estabilizador ---
    alpha_t = alpha_body + ac.tail.alpha_inc - ac.downwash
    _, Lt, Dt = lift_drag(q, ac.tail, alpha_t)
    Ft = -Dt * t_hat + Lt * n_hat
    Mt = q * ac.tail.S * ac.tail.c * ac.tail.tat.cm_c4 + ac.tail.x_c4 * Ft[1]

    Fx = Fw[0] + Ft[0]
    Fy = Fw[1] + Ft[1]
    Mz = Mw + Mt

    return Fx, Fy, Mz


def rhs(t, Y, ac: Aircraft2D, g=9.81):
    x, y, theta, u, v, w = Y
    Fx, Fy, Mz = aero_forces(ac, u, v, theta)

    udot = Fx / ac.m
    vdot = (Fy - ac.m * g) / ac.m
    wdot = Mz / ac.Iz
    return [u, v, w, udot, vdot, wdot]


# ===============================================================
# === SIMULACIÓN ===============================================
# ===============================================================

def simulate(ac: Aircraft2D, t_final, dt, x0, y0, u0, v0, theta0_deg, w0_deg_s):
    theta0 = np.deg2rad(theta0_deg)
    w0 = np.deg2rad(w0_deg_s)
    Y0 = [x0, y0, theta0, u0, v0, w0]

    eps_t = 1e-3  # s

    def event_y_zero(t, Y):
        if t < eps_t:
            return 1.0
        return Y[1]
    event_y_zero.terminal = True
    event_y_zero.direction = -1

    t_eval = np.arange(0.0, t_final + dt, dt)

    sol = solve_ivp(lambda t, y: rhs(t, y, ac),
                    [0.0, t_final], Y0,
                    t_eval=t_eval,
                    events=event_y_zero,
                    dense_output=True,
                    rtol=1e-7, atol=1e-9)

    # Si hubo impacto, t_hit es el tiempo final real
    if sol.t_events[0].size > 0:
        t_hit = float(sol.t_events[0][0])
        y_hit = sol.sol(t_hit)
        mask = sol.t <= t_hit + 1e-12
        t_out = np.append(sol.t[mask], t_hit) if sol.t[mask][-1] < t_hit else sol.t[mask]
        Y_out = sol.y[:, :mask.sum()]
        if t_out[-1] < t_hit:
            Y_out = np.column_stack([Y_out, y_hit])
        print(f"\nSimulación detenida: y(t)=0 en t = {t_hit:.3f} s")
        return t_out, Y_out, t_hit

    # Si no se cruzó el suelo, usar t_final nominal
    return sol.t, sol.y, t_final

    


# ===============================================================
# === CALCULO ===========================================
# ===============================================================

if __name__ == "__main__":

    # Archivos con línea media (X,Y en mm)

    wing_file = "ala_linea_media.dat"
    tail_file = "estab_linea_media.dat"

    # === Carga de TAT desde línea media ===
    _, _, dzdx_w = load_camber_line(wing_file)
    _, _, dzdx_t = load_camber_line(tail_file)

    tat_w = tat_from_camber(dzdx_w)
    tat_t = tat_from_camber(dzdx_t)

    # === Definición del avión ===
    wing = SurfaceAero(S=0.075, c=0.1, x_c4=+0.075, tat=tat_w, CD0=0.012, k=0.06)
    tail = SurfaceAero(S=0.144, c=0.06, x_c4=-0.45, tat=tat_t, CD0=0.010, k=0.06)
    ac = Aircraft2D(m=0.026, Iz=0.080400, rho=1.225, wing=wing, tail=tail)

    # === Simulación ===
    T, Y, t_hit = simulate(ac, x0=0, y0=2, t_final=40.0, dt=0.1,
                    u0=6.0, v0=0, theta0_deg=4.0, w0_deg_s=0.0)

    xf, yf, thetaf, uf, vf, wf = Y[:, -1]
    print("\n=== Coeficientes TAT ===")
    print(f"Ala:   αL0 = {np.rad2deg(tat_w.alpha_L0):.3f}°, cm_c/4 = {tat_w.cm_c4:.4f}")
    print(f"Estab: αL0 = {np.rad2deg(tat_t.alpha_L0):.3f}°, cm_c/4 = {tat_t.cm_c4:.4f}")

    print("\n=== Estado final ===")
    print(f"x={xf:.2f} m, y={yf:.2f} m, θ={np.rad2deg(thetaf):.2f}°")
    print(f"u={uf:.2f} m/s, v={vf:.2f} m/s, ω={np.rad2deg(wf):.3f}°/s")
    print(f"Tiempo de caída: {t_hit:.3f} s")



# ===========================================================
# === GRÁFICAS DE RESULTADOS ================================
# ===========================================================
import matplotlib.pyplot as plt

# Igualar longitud de los arreglos
if len(T) > Y.shape[1]:
    T = T[:Y.shape[1]]
elif len(T) < Y.shape[1]:
    Y = Y[:, :len(T)]

theta_deg = np.rad2deg(Y[2, :])

plt.figure(figsize=(8, 5))
plt.plot(Y[0, :], Y[1, :], 'b', linewidth=2)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Trayectoria del avión')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()

plt.figure(figsize=(8, 4))
plt.plot(T, theta_deg, 'r', linewidth=2)
plt.xlabel('Tiempo [s]')
plt.ylabel(r'$\theta$ [°]')
plt.title('Orientación del avión en el tiempo')
plt.grid(True)
plt.tight_layout()

plt.show()


