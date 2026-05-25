"""
Agro-Risk Pro — Griegas Climáticas
Delta, Gamma, Vega, Theta adaptadas a derivados HDD paramétricos
Materia: Programación para Economía y Finanzas

ANALOGÍA CON OPCIONES FINANCIERAS:
  Opción financiera  →  Derivado climático HDD
  ─────────────────     ──────────────────────
  Precio del activo  →  HDD acumulado (índice climático)
  Strike             →  Strike HDD (umbral de activación)
  Volatilidad (σ)    →  Volatilidad de temperatura diaria
  Tiempo (T)         →  Días restantes en la temporada
  Tasa libre riesgo  →  Tasa de descuento USD

GRIEGAS CLIMÁTICAS:
  Delta_clim  = ∂Prima / ∂HDD_base
                Sensibilidad al nivel de frío esperado
  Gamma_clim  = ∂²Prima / ∂HDD_base²
                Convexidad: cómo acelera Delta cuando el HDD se acerca al strike
  Vega_clim   = ∂Prima / ∂σ_temp
                Cuánto sube la prima si la volatilidad climática sube 1°C
  Theta_clim  = ∂Prima / ∂t  (negativo)
                Cuánto pierde valor la cobertura por cada día que pasa
  Rho_clim    = ∂Prima / ∂ENSO
                Sensibilidad al índice El Niño/La Niña

MÉTODO: Monte Carlo diferencial (bump-and-reprice)
  Prima(X + ε) - Prima(X - ε)
  ─────────────────────────── = Griega respecto a X
            2ε
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

C_FONDO  = "#0f1117"; C_PANEL = "#1a1d27"; C_TEXTO  = "#e8eaf0"
C_SUB    = "#8b90a0"; C_AZUL  = "#4f8ef7"; C_VERDE  = "#2ecc71"
C_NARANJA= "#f39c12"; C_ROJO  = "#e74c3c"; C_GRID   = "#2a2d3a"
C_MORADO = "#9b59b6"; C_CIAN  = "#1abc9c"

plt.rcParams.update({
    "figure.facecolor": C_FONDO, "axes.facecolor": C_PANEL,
    "axes.edgecolor":   C_GRID,  "axes.labelcolor": C_TEXTO,
    "axes.titlecolor":  C_TEXTO, "xtick.color":     C_SUB,
    "ytick.color":      C_SUB,   "text.color":      C_TEXTO,
    "grid.color":       C_GRID,  "grid.linewidth":  0.5, "font.size": 10,
})

# ── Parámetros base ──────────────────────────────────────────────────
UMBRAL     = 10.0
STRIKE     = 38.0
CAP        = 65.0
TICK       = 250.0
N_DIAS     = 90
N_SIM      = 20_000
TASA_DESC  = 0.06

# ── Calibración con datos históricos ────────────────────────────────
df = pd.read_csv("outputs/temperatura_sabana_10años.csv",
                 parse_dates=["fecha"])

df["dia_año"] = df["fecha"].dt.dayofyear.clip(1, 365)
cal = df.groupby("dia_año")["t_promedio_c"].agg(
    media="mean", sigma="std"
).reindex(range(1,366)).interpolate()

df_s   = df.sort_values("fecha")
resid  = (df_s["t_promedio_c"].values
          - cal.loc[df_s["dia_año"].values, "media"].values)
AR1    = np.corrcoef(resid[:-1], resid[1:])[0,1]
SIGMA_BASE = float(np.std(resid) * np.sqrt(1 - AR1**2))

dias_idx  = np.array(pd.date_range("2025-01-01", periods=N_DIAS, freq="D").dayofyear).clip(1,365)
MU_VEC    = cal.loc[dias_idx, "media"].values


# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN CORE: precio del derivado dado un conjunto de parámetros
# ═══════════════════════════════════════════════════════════════════
def precio_mc(mu_shift=0.0, sigma=None, enso_adj=0.0,
              dias=None, seed=42):
    """
    Calcula prima justa del derivado HDD por Monte Carlo.

    Parámetros
    ----------
    mu_shift  : desplazamiento de temperatura base (°C) — para Delta/Gamma
    sigma     : volatilidad diaria (°C) — para Vega (default = SIGMA_BASE)
    enso_adj  : ajuste por índice ENSO (°C de enfriamiento adicional) — Rho
    dias      : días restantes en temporada (int) — para Theta
    seed      : semilla reproducible

    Retorna
    -------
    float : prima esperada (USD)
    """
    if sigma is None:
        sigma = SIGMA_BASE
    if dias is None:
        dias = N_DIAS

    np.random.seed(seed)
    mu_local = MU_VEC[:dias] + mu_shift + enso_adj

    eps    = np.random.normal(0, sigma, (N_SIM, dias))
    T_sim  = np.zeros((N_SIM, dias))
    e_prev = np.zeros(N_SIM)

    for t in range(dias):
        T_sim[:, t] = mu_local[t] + AR1 * e_prev + eps[:, t]
        e_prev      = T_sim[:, t] - mu_local[t]

    hdd_total = np.maximum(0.0, UMBRAL - T_sim).sum(axis=1)
    payoff    = np.minimum(
        np.maximum(0.0, hdd_total - STRIKE),
        CAP - STRIKE
    ) * TICK

    factor_desc = 1 / (1 + TASA_DESC) ** (dias / 365)
    return float(payoff.mean() * factor_desc)


# ═══════════════════════════════════════════════════════════════════
# CÁLCULO DE GRIEGAS (bump-and-reprice)
# ═══════════════════════════════════════════════════════════════════
print("=" * 65)
print("GRIEGAS CLIMÁTICAS — Derivado HDD Call Spread")
print(f"Strike={STRIKE} | Cap={CAP} | Tick=USD {TICK} | Días={N_DIAS}")
print(f"σ_base={SIGMA_BASE:.4f}°C | AR(1)={AR1:.4f}")
print("=" * 65)

P0 = precio_mc()
print(f"\nPrima base:  USD {P0:,.2f}")

# ── DELTA climático ──────────────────────────────────────────────────
# ∂Prima/∂T_base  —  pero invertido: +1°C en temperatura → menos HDD
# Bump: ±0.5°C en la temperatura media
eps_delta = 0.5
P_d_up    = precio_mc(mu_shift=+eps_delta)
P_d_dn    = precio_mc(mu_shift=-eps_delta)
delta     = (P_d_up - P_d_dn) / (2 * eps_delta)
# Interpretación: delta respecto a HDD (bajamos T → más HDD → prima sube)
delta_hdd = (P_d_dn - P_d_up) / (2 * eps_delta)

print(f"\n{'─'*65}")
print("DELTA CLIMÁTICO  (∂Prima / ∂T_base)")
print(f"{'─'*65}")
print(f"  Prima con T+0.5°C:   USD {P_d_up:,.2f}")
print(f"  Prima con T-0.5°C:   USD {P_d_dn:,.2f}")
print(f"  Delta (respecto a T):   USD {delta:+.2f} por °C de temperatura")
print(f"  Delta (respecto a HDD): USD {delta_hdd:+.2f} por °C·día de HDD adicional")
print(f"\n  → Si la temperatura baja 1°C, la prima sube USD {abs(delta):.2f}")
print(f"  → El exportador pierde esa diferencia si no tiene cobertura")

# ── GAMMA climático ──────────────────────────────────────────────────
# ∂²Prima / ∂T²  —  convexidad de la prima
# Segunda derivada centrada: (P+ - 2P0 + P-) / ε²
gamma = (P_d_up - 2*P0 + P_d_dn) / (eps_delta**2)

print(f"\n{'─'*65}")
print("GAMMA CLIMÁTICO  (∂²Prima / ∂T²)")
print(f"{'─'*65}")
print(f"  Gamma: {gamma:+.4f} USD / °C²")
delta_en_strike = precio_mc(mu_shift=-(STRIKE/N_DIAS - SIGMA_BASE))
print(f"  Prima cerca del strike: USD {delta_en_strike:,.2f}")
if gamma < 0:
    print(f"\n  → Gamma negativo: Delta se REDUCE cuando T sube más (efecto cap)")
    print(f"     La prima crece con fuerza cerca del strike y se aplana al llegar al cap")
else:
    print(f"\n  → Gamma positivo: Delta se ACELERA conforme T baja (convexidad)")
    print(f"     La cobertura gana valor cada vez más rápido cerca del strike")

# ── VEGA climático ───────────────────────────────────────────────────
# ∂Prima / ∂σ_temp  —  sensibilidad a la volatilidad del clima
eps_vega  = 0.5   # bump de ±0.5°C en volatilidad diaria
P_v_up    = precio_mc(sigma=SIGMA_BASE + eps_vega)
P_v_dn    = precio_mc(sigma=max(0.1, SIGMA_BASE - eps_vega))
vega      = (P_v_up - P_v_dn) / (2 * eps_vega)

print(f"\n{'─'*65}")
print("VEGA CLIMÁTICO  (∂Prima / ∂σ_temperatura)")
print(f"{'─'*65}")
print(f"  σ base:               {SIGMA_BASE:.4f}°C/día")
print(f"  Prima con σ+0.5°C:   USD {P_v_up:,.2f}")
print(f"  Prima con σ-0.5°C:   USD {P_v_dn:,.2f}")
print(f"  Vega:                 USD {vega:+.2f} por °C de volatilidad diaria")
print(f"\n  → Si la volatilidad del clima sube 1°C (ej: año de El Niño intenso),")
print(f"    la prima sube USD {abs(vega):.2f}")
print(f"  → La aseguradora debe cobrar más en años de mayor incertidumbre climática")

# Vega en rango completo de sigmas
sigmas_rango = np.linspace(0.5, 5.0, 15)
precios_vega = [precio_mc(sigma=s, seed=42) for s in sigmas_rango]

# ── THETA climático ──────────────────────────────────────────────────
# ∂Prima / ∂t  —  decaimiento temporal (negativo para opciones largas)
dias_rango = [90, 75, 60, 45, 30, 15, 7, 3, 1]
precios_theta = [precio_mc(dias=d, seed=42) for d in dias_rango]
theta_7d = (precio_mc(dias=8) - precio_mc(dias=6)) / (-2)  # aprox

print(f"\n{'─'*65}")
print("THETA CLIMÁTICO  (∂Prima / ∂días_restantes)")
print(f"{'─'*65}")
for d, p in zip(dias_rango, precios_theta):
    barra = "█" * int(p / 100)
    print(f"  {d:>2} días restantes: USD {p:>8,.2f}  {barra}")
print(f"\n  Theta (con 7 días restantes): USD {theta_7d:+.2f}/día")
print(f"\n  → La prima decae conforme se acerca el vencimiento")
print(f"    (si no hay HDD acumulado, el tiempo juega contra el comprador)")

# ── RHO_ENSO climático ───────────────────────────────────────────────
# Sensibilidad al índice ENSO (El Niño = temperaturas más altas → menos HDD)
eps_enso  = 0.5
P_e_up    = precio_mc(enso_adj=-eps_enso)  # El Niño: +0.5°C → menos HDD
P_e_dn    = precio_mc(enso_adj=+eps_enso)  # La Niña: -0.5°C → más HDD
rho_enso  = (P_e_up - P_e_dn) / (2 * eps_enso)

print(f"\n{'─'*65}")
print("RHO_ENSO  (∂Prima / ∂ENSO)")
print(f"{'─'*65}")
print(f"  Prima escenario El Niño  (ENSO+0.5, T+0.5°C): USD {P_e_up:,.2f}")
print(f"  Prima escenario La Niña  (ENSO-0.5, T-0.5°C): USD {P_e_dn:,.2f}")
print(f"  Rho_ENSO: USD {rho_enso:+.2f} por unidad de índice ENSO")
print(f"\n  → En año de El Niño la prima baja USD {abs(rho_enso):.2f}")
print(f"    En año de La Niña la prima sube USD {abs(rho_enso):.2f}")
print(f"    El exportador debería comprar cobertura anticipando La Niña")

# ── Tabla resumen griegas ────────────────────────────────────────────
print(f"\n{'='*65}")
print("TABLA RESUMEN DE GRIEGAS CLIMÁTICAS")
print(f"{'='*65}")
print(f"  {'Griega':<18} {'Valor':>12}  {'Interpretación'}")
print(f"  {'─'*60}")
print(f"  {'Prima base':<18} {'USD '+f'{P0:,.2f}':>12}  Precio justo del contrato")
print(f"  {'Delta (vs T)':<18} {'USD '+f'{delta:+.2f}':>12}  Por cada °C que sube T")
print(f"  {'Delta (vs HDD)':<18} {'USD '+f'{delta_hdd:+.2f}':>12}  Por cada °C·día adicional")
print(f"  {'Gamma':<18} {f'{gamma:+.4f}':>12}  Convexidad de la prima")
print(f"  {'Vega':<18} {'USD '+f'{vega:+.2f}':>12}  Por cada °C de σ adicional")
print(f"  {'Theta (7d)':<18} {'USD '+f'{theta_7d:+.2f}':>12}  Por día que pasa (7d restantes)")
print(f"  {'Rho_ENSO':<18} {'USD '+f'{rho_enso:+.2f}':>12}  Por unidad índice ENSO")


# ═══════════════════════════════════════════════════════════════════
# SUPERFICIE DE PRECIO: PRIMA vs (T_base, σ)
# ═══════════════════════════════════════════════════════════════════
print("\nCalculando superficie de precio (puede tardar 30s)...")
t_shifts = np.linspace(-3, 3, 12)
sigmas_s = np.linspace(0.5, 4.5, 12)
superficie = np.zeros((len(sigmas_s), len(t_shifts)))

for i, s in enumerate(sigmas_s):
    for j, t in enumerate(t_shifts):
        superficie[i, j] = precio_mc(mu_shift=t, sigma=s, seed=42)

print("✅ Superficie calculada")


# ═══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN — 6 PANELES
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16), facecolor=C_FONDO)
fig.suptitle("Agro-Risk Pro  ·  Griegas Climáticas — Sensibilidades del Derivado HDD",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.34,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

# ── Panel A: Delta — Prima vs desplazamiento temperatura ────────────
ax1 = fig.add_subplot(gs[0, 0])
t_bumps  = np.linspace(-4, 4, 25)
p_delta  = [precio_mc(mu_shift=t, seed=42) for t in t_bumps]

ax1.plot(t_bumps, p_delta, color=C_AZUL, lw=2.5, marker="o", ms=4)
ax1.fill_between(t_bumps, p_delta, alpha=0.15, color=C_AZUL)
ax1.axvline(0, color=C_SUBTEXTO if False else C_SUB, lw=1, ls="--", alpha=0.6)
ax1.axhline(P0, color=C_NARANJA, lw=1.2, ls="--",
            label=f"Prima base USD {P0:,.0f}")

# Línea tangente en 0 (Delta)
x_tan = np.array([-2, 2])
ax1.plot(x_tan, P0 + delta * x_tan, color=C_ROJO, lw=1.5,
         ls=":", label=f"Tangente (Δ={delta:+.0f})")

ax1.axvspan(-4, 0, alpha=0.05, color=C_ROJO,   label="T baja → más HDD")
ax1.axvspan(0,  4, alpha=0.05, color=C_VERDE,  label="T sube → menos HDD")
ax1.set_xlabel("Desplazamiento T base (°C)")
ax1.set_ylabel("Prima (USD)")
ax1.set_title("A  Delta — Prima vs Temperatura Base", fontsize=10, pad=8)
ax1.legend(fontsize=8, framealpha=0.2); ax1.grid(alpha=0.3)

# Anotar convexidad (Gamma)
ax1.annotate(f"Gamma = {gamma:+.3f}\nConvexidad {'positiva' if gamma>=0 else 'negativa'}",
             xy=(0, P0), xytext=(1.5, P0*0.6 if P0 > 0 else 200),
             fontsize=8, color=C_TEXTO,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL,
                       edgecolor=C_AZUL, alpha=0.8),
             arrowprops=dict(arrowstyle="->", color=C_AZUL, lw=1))

# ── Panel B: Vega — Prima vs Volatilidad ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(sigmas_rango, precios_vega, color=C_MORADO, lw=2.5, marker="s", ms=4)
ax2.fill_between(sigmas_rango, precios_vega, alpha=0.15, color=C_MORADO)
ax2.axvline(SIGMA_BASE, color=C_NARANJA, lw=1.5, ls="--",
            label=f"σ actual {SIGMA_BASE:.2f}°C")
ax2.axhline(P0, color=C_NARANJA, lw=1, ls=":", alpha=0.6,
            label=f"Prima base USD {P0:,.0f}")

# Banda de σ El Niño
sigma_enso = SIGMA_BASE * 1.5
ax2.axvspan(sigma_enso, sigmas_rango[-1], alpha=0.07, color=C_ROJO,
            label="Zona El Niño/alta incertidumbre")

# Etiqueta Vega
idx_base = np.argmin(np.abs(sigmas_rango - SIGMA_BASE))
ax2.annotate(f"Vega = USD {vega:+.1f}/°C",
             xy=(SIGMA_BASE, precios_vega[idx_base]),
             xytext=(SIGMA_BASE + 0.8, precios_vega[idx_base] * 0.7 if precios_vega[idx_base] > 0 else 200),
             fontsize=8.5, color=C_MORADO,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL,
                       edgecolor=C_MORADO, alpha=0.8),
             arrowprops=dict(arrowstyle="->", color=C_MORADO, lw=1))

ax2.set_xlabel("Volatilidad temperatura diaria (°C)")
ax2.set_ylabel("Prima (USD)")
ax2.set_title("B  Vega — Prima vs Volatilidad Climática", fontsize=10, pad=8)
ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(alpha=0.3)

# ── Panel C: Theta — Decaimiento temporal ───────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(dias_rango, precios_theta, color=C_VERDE, lw=2.5,
         marker="D", ms=5)
ax3.fill_between(dias_rango, precios_theta, alpha=0.15, color=C_VERDE)
# Marcar días críticos
for d, p in zip(dias_rango, precios_theta):
    ax3.annotate(f"  {p:,.0f}", xy=(d, p), fontsize=7.5,
                 color=C_TEXTO, va="center")
ax3.axvline(30, color=C_NARANJA, lw=1.2, ls="--", alpha=0.7,
            label="30 días (un mes)")
ax3.axvline(7,  color=C_ROJO,    lw=1.2, ls=":",  alpha=0.7,
            label="7 días (última semana)")
ax3.set_xlabel("Días restantes en temporada")
ax3.set_ylabel("Prima (USD)")
ax3.set_title("C  Theta — Decaimiento del Valor en el Tiempo", fontsize=10, pad=8)
ax3.invert_xaxis()
ax3.legend(fontsize=8, framealpha=0.2); ax3.grid(alpha=0.3)

# ── Panel D: Rho ENSO ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
enso_rango  = np.linspace(-2, 2, 17)
p_enso_rng  = [precio_mc(enso_adj=-e*0.4, seed=42) for e in enso_rango]
# ENSO + → El Niño → T sube → menos HDD → prima baja

ax4.plot(enso_rango, p_enso_rng, color=C_CIAN, lw=2.5, marker="^", ms=4)
ax4.fill_between(enso_rango, p_enso_rng, alpha=0.15, color=C_CIAN)
ax4.axvline(0,    color=C_SUB,    lw=1, ls="--", alpha=0.5, label="ENSO neutro")
ax4.axvline(-1.5, color=C_AZUL,   lw=1.2, ls=":",  label="La Niña fuerte")
ax4.axvline( 1.5, color=C_ROJO,   lw=1.2, ls=":",  label="El Niño fuerte")
ax4.axhline(P0,   color=C_NARANJA,lw=1, ls="--", alpha=0.5,
            label=f"Prima base USD {P0:,.0f}")
ax4.axvspan(-2, 0, alpha=0.06, color=C_AZUL)
ax4.axvspan( 0, 2, alpha=0.06, color=C_ROJO)

ax4.text(-1.5, max(p_enso_rng)*0.85, "La Niña\nmás frío\nprima SUBE",
         ha="center", fontsize=8, color=C_AZUL)
ax4.text( 1.5, max(p_enso_rng)*0.85, "El Niño\nmás calor\nprima BAJA",
         ha="center", fontsize=8, color=C_ROJO)

ax4.set_xlabel("Índice ENSO")
ax4.set_ylabel("Prima (USD)")
ax4.set_title("D  Rho_ENSO — Sensibilidad al Fenómeno del Niño/Niña", fontsize=10, pad=8)
ax4.legend(fontsize=8, framealpha=0.2); ax4.grid(alpha=0.3)

# ── Panel E: Superficie de precio Prima(T_shift, σ) ─────────────────
ax5 = fig.add_subplot(gs[2, 0])
im = ax5.imshow(superficie, aspect="auto", cmap="RdYlGn",
                origin="lower",
                extent=[t_shifts[0], t_shifts[-1],
                        sigmas_s[0],  sigmas_s[-1]])
ax5.axvline(0,          color="white", lw=1.5, ls="--", alpha=0.8,
            label="T base actual")
ax5.axhline(SIGMA_BASE, color="white", lw=1.5, ls=":",  alpha=0.8,
            label=f"σ actual {SIGMA_BASE:.2f}°C")
ax5.scatter([0], [SIGMA_BASE], color="white", s=80, zorder=5,
            label=f"Punto base  USD {P0:,.0f}")
cb = fig.colorbar(im, ax=ax5, fraction=0.04, pad=0.02)
cb.set_label("Prima (USD)", fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)
ax5.set_xlabel("Desplazamiento T base (°C)")
ax5.set_ylabel("Volatilidad σ (°C/día)")
ax5.set_title("E  Superficie de Precio Prima(ΔT, σ)", fontsize=10, pad=8)
ax5.legend(fontsize=8, framealpha=0.4); 

# ── Panel F: Todas las griegas normalizadas ──────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
griegas_nombres = ["Delta\n(vs T, USD/°C)",
                   "Delta\n(vs HDD)",
                   "Vega\n(USD/°C σ)",
                   "Theta\n(USD/día·7d)",
                   "Rho_ENSO\n(USD/idx)"]
griegas_vals    = [delta, delta_hdd, vega, theta_7d, rho_enso]
colores_g       = [C_AZUL, C_CIAN, C_MORADO, C_VERDE, C_NARANJA]

bars = ax6.barh(griegas_nombres, griegas_vals,
                color=colores_g, alpha=0.80, height=0.55)
ax6.axvline(0, color=C_TEXTO, lw=1.2, alpha=0.7)

for bar, val in zip(bars, griegas_vals):
    xoff = 8 if val >= 0 else -8
    ha   = "left" if val >= 0 else "right"
    ax6.text(val + xoff, bar.get_y() + bar.get_height()/2,
             f"{val:+.1f}", va="center", ha=ha, fontsize=9,
             fontweight="bold",
             color=bar.get_facecolor())

ax6.set_xlabel("Valor de la griega (USD o adimensional)")
ax6.set_title("F  Resumen Griegas Climáticas (valores absolutos)", fontsize=10, pad=8)
ax6.grid(axis="x", alpha=0.3)

plt.savefig("outputs/fig8_griegas.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()

# Exportar tabla de griegas
pd.DataFrame({
    "griega":        griegas_nombres,
    "valor":         griegas_vals,
    "descripcion":   [
        "USD por °C que sube T base",
        "USD por °C·día adicional de HDD",
        "USD por °C de volatilidad diaria",
        "USD por día transcurrido (7d restantes)",
        "USD por unidad de índice ENSO"
    ]
}).to_csv("outputs/11_griegas_climaticas.csv", index=False)

print(f"\n✅ fig8_griegas.png guardada")
print(f"✅ 11_griegas_climaticas.csv exportado")
