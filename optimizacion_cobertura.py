"""
Agro-Risk Pro — Optimización del Contrato de Cobertura
Minimizar varianza de ingresos netos del exportador
con scipy.optimize

PROBLEMA DE OPTIMIZACIÓN:
  Variables de decisión:  x = [strike_HDD, tick_USD]
  Función objetivo:       min Var( Ventas(t) - Prima(x) + Payoff(x,t) )
  
  donde:
    Ventas(t)     = ingreso semanal correlacionado con temperatura
    Prima(x)      = E[Payoff(x,t)] × (1 + carga_seguro)
    Payoff(x,t)   = max(0, HDD(t) - strike) × tick  [sin cap por ahora]

  Restricciones:
    strike ∈ [10, 200]      válido para temporada de 90 días
    tick   ∈ [10, 2000]     rango económicamente razonable
    Prima  ≤ 0.15 × Ventas  la cobertura no puede costar más del 15% del ingreso

INTUICIÓN ECONÓMICA:
  Sin cobertura: Ingreso = Ventas(t)  → varianza alta (depende del clima)
  Con cobertura: Ingreso = Ventas(t) - Prima + Payoff(t)
  El contrato óptimo es el que más "aplaniza" la distribución de ingresos.
  En el límite perfecto, la varianza llegaría a cero (cobertura perfecta).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, differential_evolution
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

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# 1. GENERAR ESCENARIOS HISTÓRICOS COMPLETOS
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("outputs/temperatura_sabana_10años.csv",
                 parse_dates=["fecha"])

UMBRAL    = 10.0
CARGA_SEG = 0.20
R_BASE    = 16_000    # USD/semana ingreso base flores

# Datos semanales
df["hdd"]  = np.maximum(0.0, UMBRAL - df["t_promedio_c"])
df["anio"] = df["fecha"].dt.year
df["mes"]  = df["fecha"].dt.month

sem = df.set_index("fecha").resample("W").agg(
    hdd_sem     = ("hdd",           "sum"),
    t_min_media = ("t_minima_c",    "mean"),
    heladas     = ("evento_helada", "sum"),
    enso        = ("enso_index",    "mean"),
).reset_index()

# Ventas correlacionadas con temperatura (del script de regresión)
BETA_HDD    = -85
BETA_HELADA = -1_800
BETA_ENSO   = -600
SIGMA_RUIDO = 1_200
AR1_V       = 0.45

ruido  = np.zeros(len(sem))
eps    = np.random.normal(0, SIGMA_RUIDO, len(sem))
ruido[0] = eps[0]
for i in range(1, len(sem)):
    ruido[i] = AR1_V * ruido[i-1] + eps[i]

sem["ventas_usd"] = (
    R_BASE
    + BETA_HDD    * sem["hdd_sem"]
    + BETA_HELADA * sem["heladas"]
    + BETA_ENSO   * sem["enso"].clip(lower=0)
    + ruido
).clip(lower=0)

N = len(sem)
HDD   = sem["hdd_sem"].values
VENTAS = sem["ventas_usd"].values

print("=" * 65)
print("OPTIMIZACIÓN DEL CONTRATO DE COBERTURA")
print(f"Semanas de datos: {N}  |  Umbral HDD: {UMBRAL}°C")
print(f"Ventas base: USD {R_BASE:,}/semana")
print(f"Var(Ventas sin cobertura): USD² {np.var(VENTAS):,.0f}")
print(f"Std(Ventas sin cobertura): USD  {np.std(VENTAS):,.0f}")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# 2. FUNCIÓN OBJETIVO
# ═══════════════════════════════════════════════════════════════════
def calcular_ingresos_netos(strike, tick, hdd_vec, ventas_vec, carga=CARGA_SEG):
    """
    Calcula ingresos netos semana a semana dado un contrato (strike, tick).
    
    Returns: array de ingresos netos
    """
    payoff    = np.maximum(0.0, hdd_vec - strike) * tick
    prima     = payoff.mean() * (1 + carga)
    ingresos  = ventas_vec - prima + payoff
    return ingresos, prima


def objetivo(x, hdd_vec=HDD, ventas_vec=VENTAS):
    """
    Función a minimizar: varianza de ingresos netos.
    x = [strike, tick]
    Penalización si la prima supera el 15% del ingreso medio.
    """
    strike, tick = x[0], x[1]

    # Restricciones suaves (penalización interior)
    if strike < 0 or tick < 0:
        return 1e12

    ingresos, prima = calcular_ingresos_netos(strike, tick, hdd_vec, ventas_vec)

    varianza = np.var(ingresos)

    # Penalización: prima no debe superar 15% del ingreso medio
    ingreso_medio = ventas_vec.mean()
    if prima > 0.15 * ingreso_medio:
        penalizacion = 1e8 * (prima - 0.15 * ingreso_medio)**2
        varianza += penalizacion

    return varianza


# ═══════════════════════════════════════════════════════════════════
# 3. OPTIMIZACIÓN — MÚLTIPLES MÉTODOS
# ═══════════════════════════════════════════════════════════════════
bounds = [(0.1, 200.0), (1.0, 2000.0)]

# ── Método 1: Evolución diferencial (global, no necesita gradiente) ──
print("\n[1/3] Evolución diferencial (búsqueda global)...")
res_de = differential_evolution(
    objetivo,
    bounds       = bounds,
    seed         = 42,
    maxiter      = 500,
    tol          = 1e-8,
    popsize      = 20,
    mutation     = (0.5, 1.5),
    recombination= 0.9,
    workers      = 1,
    polish       = True,
    disp         = False
)
strike_de, tick_de = res_de.x
print(f"   Strike* = {strike_de:.2f}  |  Tick* = {tick_de:.2f}")
print(f"   Var* = {res_de.fun:,.0f}  |  Éxito: {res_de.success}")

# ── Método 2: Nelder-Mead desde múltiples puntos iniciales ──────────
print("\n[2/3] Nelder-Mead (múltiples arranques)...")
starts = [
    [10, 50], [20, 100], [50, 200], [80, 500],
    [5, 1000], [100, 50], [30, 300], [15, 750]
]
resultados_nm = []
for x0 in starts:
    r = minimize(
        objetivo, x0,
        method  = "Nelder-Mead",
        options = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10000}
    )
    if r.success or r.fun < 1e10:
        resultados_nm.append((r.fun, r.x[0], r.x[1]))

resultados_nm.sort(key=lambda x: x[0])
var_nm, strike_nm, tick_nm = resultados_nm[0]
print(f"   Strike* = {strike_nm:.2f}  |  Tick* = {tick_nm:.2f}")
print(f"   Var* = {var_nm:,.0f}  |  Arranques: {len(starts)}")

# ── Método 3: L-BFGS-B (gradiente numérico) ─────────────────────────
print("\n[3/3] L-BFGS-B (gradiente numérico)...")
res_lbfgs = minimize(
    objetivo,
    x0      = [strike_de, tick_de],
    method  = "L-BFGS-B",
    bounds  = [(0.1, 200), (1, 2000)],
    options = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000}
)
strike_lb, tick_lb = res_lbfgs.x
print(f"   Strike* = {strike_lb:.2f}  |  Tick* = {tick_lb:.2f}")
print(f"   Var* = {res_lbfgs.fun:,.0f}")

# ── Seleccionar el mejor resultado ───────────────────────────────────
mejor = min(
    [(res_de.fun, strike_de, tick_de, "Evolución Diferencial"),
     (var_nm,     strike_nm, tick_nm, "Nelder-Mead"),
     (res_lbfgs.fun, strike_lb, tick_lb, "L-BFGS-B")],
    key=lambda x: x[0]
)
var_opt, STRIKE_OPT, TICK_OPT, metodo_opt = mejor


# ═══════════════════════════════════════════════════════════════════
# 4. COMPARACIÓN: SIN COBERTURA vs HEURÍSTICA vs ÓPTIMO
# ═══════════════════════════════════════════════════════════════════
# Contrato heurístico (valores del proyecto antes de optimizar)
STRIKE_HEUR = 38.0
TICK_HEUR   = 250.0

ing_netos_opt,  prima_opt  = calcular_ingresos_netos(STRIKE_OPT,  TICK_OPT,  HDD, VENTAS)
ing_netos_heur, prima_heur = calcular_ingresos_netos(STRIKE_HEUR, TICK_HEUR, HDD, VENTAS)

var_sin    = np.var(VENTAS)
var_heur   = np.var(ing_netos_heur)
var_opt_r  = np.var(ing_netos_opt)

reduccion_heur = (1 - var_heur / var_sin) * 100
reduccion_opt  = (1 - var_opt_r / var_sin) * 100
mejora_vs_heur = (1 - var_opt_r / var_heur) * 100

print(f"\n{'='*65}")
print("COMPARACIÓN DE CONTRATOS")
print(f"{'='*65}")
print(f"\n  {'Métrica':<30} {'Sin cob.':>12} {'Heurístico':>12} {'Óptimo':>12}")
print(f"  {'─'*66}")
print(f"  {'Strike HDD':<30} {'—':>12} {STRIKE_HEUR:>12.1f} {STRIKE_OPT:>12.2f}")
print(f"  {'Tick (USD/°C·día)':<30} {'—':>12} {TICK_HEUR:>12.1f} {TICK_OPT:>12.2f}")
print(f"  {'Prima anual (USD)':<30} {'0':>12} {prima_heur*52:>12,.0f} {prima_opt*52:>12,.0f}")
print(f"  {'Prima sem. (USD)':<30} {'0':>12} {prima_heur:>12,.0f} {prima_opt:>12,.0f}")
print(f"  {'Ingreso medio (USD/sem)':<30} {VENTAS.mean():>12,.0f} {ing_netos_heur.mean():>12,.0f} {ing_netos_opt.mean():>12,.0f}")
print(f"  {'Std ingresos (USD)':<30} {np.std(VENTAS):>12,.0f} {np.std(ing_netos_heur):>12,.0f} {np.std(ing_netos_opt):>12,.0f}")
print(f"  {'Var ingresos (USD²)':<30} {var_sin:>12,.0f} {var_heur:>12,.0f} {var_opt_r:>12,.0f}")
print(f"  {'Reducción varianza':<30} {'0%':>12} {reduccion_heur:>11.1f}% {reduccion_opt:>11.1f}%")
print(f"  {'VaR 5% (USD/sem)':<30} {np.percentile(VENTAS,5):>12,.0f} {np.percentile(ing_netos_heur,5):>12,.0f} {np.percentile(ing_netos_opt,5):>12,.0f}")
print(f"  {'CVaR 5% (USD/sem)':<30} {VENTAS[VENTAS<=np.percentile(VENTAS,5)].mean():>12,.0f} {ing_netos_heur[ing_netos_heur<=np.percentile(ing_netos_heur,5)].mean():>12,.0f} {ing_netos_opt[ing_netos_opt<=np.percentile(ing_netos_opt,5)].mean():>12,.0f}")

print(f"\n  Método ganador:           {metodo_opt}")
print(f"  Mejora vs heurístico:     {mejora_vs_heur:.1f}% menos varianza")
print(f"  Reducción varianza total: {reduccion_opt:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# 5. SUPERFICIE DE VARIANZA (strike × tick)
# ═══════════════════════════════════════════════════════════════════
print("\nCalculando superficie de optimización...")
strikes_g = np.linspace(1, 180, 35)
ticks_g   = np.linspace(1, 1500, 35)
S_g, T_g  = np.meshgrid(strikes_g, ticks_g)
Z_g       = np.vectorize(lambda s, t: objetivo([s, t]))(S_g, T_g)
Z_log     = np.log1p(Z_g)   # escala log para visualizar mejor
print("✅ Superficie lista")


# ═══════════════════════════════════════════════════════════════════
# 6. ANÁLISIS DE SENSIBILIDAD ALREDEDOR DEL ÓPTIMO
# ═══════════════════════════════════════════════════════════════════
# ¿Cuánto cambia la varianza si el contrato se aleja del óptimo?
delta_strike = np.linspace(-30, 30, 61)
delta_tick   = np.linspace(-200, 200, 61)

var_strike_sens = [objetivo([STRIKE_OPT + ds, TICK_OPT]) for ds in delta_strike]
var_tick_sens   = [objetivo([STRIKE_OPT, TICK_OPT + dt]) for dt in delta_tick]


# ═══════════════════════════════════════════════════════════════════
# 7. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16), facecolor=C_FONDO)
fig.suptitle(
    f"Agro-Risk Pro  ·  Optimización scipy.optimize — Strike*={STRIKE_OPT:.1f}  Tick*={TICK_OPT:.1f}",
    fontsize=14, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

# ── Panel A: Comparación distribuciones de ingreso ──────────────────
ax1 = fig.add_subplot(gs[0, :])
kw = dict(density=True, alpha=0.55, edgecolor="none", bins=60)
ax1.hist(VENTAS,          **kw, color=C_ROJO,    label=f"Sin cobertura   σ={np.std(VENTAS):,.0f}")
ax1.hist(ing_netos_heur,  **kw, color=C_NARANJA, label=f"Heurístico      σ={np.std(ing_netos_heur):,.0f}")
ax1.hist(ing_netos_opt,   **kw, color=C_VERDE,   label=f"Óptimo          σ={np.std(ing_netos_opt):,.0f}")

for arr, col, ls in [
    (VENTAS,         C_ROJO,    "-"),
    (ing_netos_heur, C_NARANJA, "--"),
    (ing_netos_opt,  C_VERDE,   ":")
]:
    mu_k, sig_k = arr.mean(), arr.std()
    x_k = np.linspace(arr.min(), arr.max(), 300)
    ax1.plot(x_k, stats.norm.pdf(x_k, mu_k, sig_k),
             color=col, lw=2, ls=ls)

ax1.set_xlabel("Ingreso semanal (USD)")
ax1.set_ylabel("Densidad")
ax1.set_title(
    f"A  Distribución de Ingresos: Sin cobertura vs Heurístico vs Óptimo",
    fontsize=10, pad=8
)
ax1.legend(fontsize=9, framealpha=0.2)
ax1.grid(alpha=0.2)

# Anotación de reducción de varianza
ax1.annotate(
    f"Reducción varianza\nheurístico: {reduccion_heur:.1f}%\nóptimo: {reduccion_opt:.1f}%",
    xy=(VENTAS.min(), ax1.get_ylim()[1] * 0.5),
    fontsize=9, color=C_TEXTO,
    bbox=dict(boxstyle="round,pad=0.4", facecolor=C_PANEL,
              edgecolor=C_VERDE, alpha=0.9)
)

# ── Panel B: Superficie de varianza (mapa de calor) ─────────────────
ax2 = fig.add_subplot(gs[1, 0])
im  = ax2.contourf(S_g, T_g, Z_log, levels=30, cmap="RdYlGn_r")
ax2.contour(S_g, T_g, Z_log, levels=10,
            colors="white", linewidths=0.4, alpha=0.3)
ax2.scatter(STRIKE_OPT,  TICK_OPT,  color="white",   s=120, zorder=5,
            marker="*",  label=f"Óptimo ({STRIKE_OPT:.1f}, {TICK_OPT:.1f})")
ax2.scatter(STRIKE_HEUR, TICK_HEUR, color=C_NARANJA, s=80,  zorder=5,
            marker="D",  label=f"Heurístico ({STRIKE_HEUR}, {TICK_HEUR})")
cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
cb.set_label("log(1 + Varianza)", fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)
ax2.set_xlabel("Strike HDD (°C·día)")
ax2.set_ylabel("Tick (USD/°C·día)")
ax2.set_title("B  Superficie de Varianza — Verde = óptimo", fontsize=10, pad=8)
ax2.legend(fontsize=8, framealpha=0.3)

# ── Panel C: Convergencia — varianza vs strike (tick fijo) ──────────
ax3 = fig.add_subplot(gs[1, 1])
ax3b = ax3.twinx()
ax3.plot(delta_strike + STRIKE_OPT, var_strike_sens,
         color=C_AZUL, lw=2, label="Var vs strike")
ax3.fill_between(delta_strike + STRIKE_OPT, var_strike_sens,
                 alpha=0.12, color=C_AZUL)
ax3b.plot(delta_tick + TICK_OPT, var_tick_sens,
          color=C_MORADO, lw=2, ls="--", label="Var vs tick")

ax3.axvline(STRIKE_OPT, color=C_VERDE, lw=1.5, ls=":",
            label=f"Strike* {STRIKE_OPT:.1f}")
ax3b.axvline(TICK_OPT,   color=C_CIAN,  lw=1.2, ls=":",
             label=f"Tick* {TICK_OPT:.1f}")

ax3.set_xlabel("Valor del parámetro")
ax3.set_ylabel("Varianza (Strike fijo)", color=C_AZUL)
ax3b.set_ylabel("Varianza (Tick fijo)", color=C_MORADO)
ax3b.tick_params(colors=C_MORADO)
ax3.set_title("C  Sensibilidad: Varianza vs Strike y Tick", fontsize=10, pad=8)
lns = ax3.get_lines() + ax3b.get_lines()
ax3.legend(lns, [l.get_label() for l in lns], fontsize=8, framealpha=0.2)
ax3.grid(alpha=0.3)

# ── Panel D: Serie temporal de ingresos ─────────────────────────────
ax4 = fig.add_subplot(gs[2, :])
fechas_plot = sem["fecha"].values

ax4.fill_between(fechas_plot, VENTAS,
                 alpha=0.15, color=C_ROJO)
ax4.plot(fechas_plot, VENTAS,
         color=C_ROJO, lw=0.8, alpha=0.6,
         label=f"Sin cobertura  μ={VENTAS.mean():,.0f}  σ={np.std(VENTAS):,.0f}")
ax4.plot(fechas_plot, ing_netos_heur,
         color=C_NARANJA, lw=1, alpha=0.8,
         label=f"Heurístico  μ={ing_netos_heur.mean():,.0f}  σ={np.std(ing_netos_heur):,.0f}")
ax4.plot(fechas_plot, ing_netos_opt,
         color=C_VERDE, lw=1.5,
         label=f"Óptimo  μ={ing_netos_opt.mean():,.0f}  σ={np.std(ing_netos_opt):,.0f}")

# Bandas ±1σ del óptimo
mu_o  = ing_netos_opt.mean()
sig_o = ing_netos_opt.std()
ax4.axhline(mu_o,        color=C_VERDE, lw=1,   ls="--", alpha=0.5)
ax4.axhline(mu_o + sig_o,color=C_VERDE, lw=0.8, ls=":",  alpha=0.4)
ax4.axhline(mu_o - sig_o,color=C_VERDE, lw=0.8, ls=":",  alpha=0.4)
ax4.fill_between(fechas_plot, mu_o - sig_o, mu_o + sig_o,
                 alpha=0.05, color=C_VERDE)

ax4.set_ylabel("Ingreso semanal (USD)")
ax4.set_xlabel("Fecha")
ax4.set_title(
    "D  Serie Temporal de Ingresos — El contrato óptimo aplana la distribución",
    fontsize=10, pad=8
)
ax4.legend(fontsize=9, framealpha=0.2, ncol=3)
ax4.grid(alpha=0.2)

plt.savefig("outputs/fig9_optimizacion.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()

# ── Exportar resultados ───────────────────────────────────────────
resultados_df = pd.DataFrame({
    "contrato":        ["Sin cobertura", "Heurístico", "Óptimo"],
    "strike":          [None, STRIKE_HEUR, STRIKE_OPT],
    "tick_usd":        [None, TICK_HEUR,   TICK_OPT],
    "prima_semanal":   [0, prima_heur, prima_opt],
    "ingreso_medio":   [VENTAS.mean(), ing_netos_heur.mean(), ing_netos_opt.mean()],
    "std_ingresos":    [np.std(VENTAS), np.std(ing_netos_heur), np.std(ing_netos_opt)],
    "var_ingresos":    [var_sin, var_heur, var_opt_r],
    "reduccion_var_pct":[0, reduccion_heur, reduccion_opt],
    "var_5pct":        [np.percentile(VENTAS,5),
                        np.percentile(ing_netos_heur,5),
                        np.percentile(ing_netos_opt,5)],
})
resultados_df.to_csv("outputs/12_optimizacion.csv", index=False)

print(f"\n{'='*65}")
print("RESUMEN EJECUTIVO")
print(f"{'='*65}")
print(f"  Método ganador:    {metodo_opt}")
print(f"  Strike óptimo:     {STRIKE_OPT:.2f} °C·día")
print(f"  Tick óptimo:       USD {TICK_OPT:.2f} / °C·día")
print(f"  Prima semanal:     USD {prima_opt:,.2f}")
print(f"  Std sin cob.:      USD {np.std(VENTAS):,.0f}/semana")
print(f"  Std con óptimo:    USD {np.std(ing_netos_opt):,.0f}/semana")
print(f"  Reducción varianza:{reduccion_opt:.1f}%")
print(f"  Mejora vs heurís.: {mejora_vs_heur:.1f}%")
print(f"\n✅ fig9_optimizacion.png guardada")
print(f"✅ 12_optimizacion.csv exportado")
