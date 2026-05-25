"""
Agro-Risk Pro — Regresión Lineal: Temperatura vs Rendimiento Exportador
Materia: Programación para Economía y Finanzas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── Paleta ───────────────────────────────────────────────────────────
C_FONDO   = "#0f1117"; C_PANEL  = "#1a1d27"; C_TEXTO  = "#e8eaf0"
C_SUB     = "#8b90a0"; C_AZUL   = "#4f8ef7"; C_VERDE  = "#2ecc71"
C_NARANJA = "#f39c12"; C_ROJO   = "#e74c3c"; C_GRID   = "#2a2d3a"
C_MORADO  = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor": C_FONDO, "axes.facecolor": C_PANEL,
    "axes.edgecolor":   C_GRID,  "axes.labelcolor": C_TEXTO,
    "axes.titlecolor":  C_TEXTO, "xtick.color":     C_SUB,
    "ytick.color":      C_SUB,   "text.color":      C_TEXTO,
    "grid.color":       C_GRID,  "grid.linewidth":  0.5,
    "font.size": 10,
})

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS CLIMÁTICOS
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("outputs/temperatura_sabana_10años.csv",
                 parse_dates=["fecha"])

# Usar temperatura mínima diaria como variable explicativa
# Agregar a semanas (escala operativa del exportador)
df_sem = df.set_index("fecha").resample("W").agg(
    t_min_media   = ("t_minima_c",  "mean"),
    t_min_minima  = ("t_minima_c",  "min"),
    t_prom_media  = ("t_promedio_c","mean"),
    hdd_semana    = ("hdd",         "sum"),
    heladas       = ("evento_helada","sum"),
    precip_total  = ("precipitacion_mm","sum"),
    enso          = ("enso_index",  "mean"),
).reset_index()

n = len(df_sem)  # ~522 semanas

# ═══════════════════════════════════════════════════════════════════
# 2. SIMULAR RENDIMIENTOS DE EXPORTACIÓN DE FLORES
# ═══════════════════════════════════════════════════════════════════
"""
Modelo económico del exportador de Cundinamarca:
  - Ingreso base: USD 16,000/semana (aprox. exportadora mediana)
  - Correlación negativa con temperatura mínima:
      * Cada grado bajo el umbral de 10°C reduce el rendimiento
      * Heladas destruyen entre 15-40% de la cosecha semanal
      * El Niño (ENSO > 0.5) reduce producción por sequía
  - Componentes del modelo:
      R(t) = R_base + β₁·HDD_t + β₂·heladas_t + β₃·ENSO_t + ε_t
"""

R_BASE       = 16_000    # USD/semana ingreso base
BETA_HDD     = -85       # USD por grado-día de frío acumulado
BETA_HELADA  = -1_800    # USD por día de helada en la semana
BETA_ENSO    = -600      # USD por unidad de índice El Niño
SIGMA_RUIDO  = 1_200     # USD desviación estándar del ruido operativo

# Componente estructural
rendimiento_base = (
    R_BASE
    + BETA_HDD    * df_sem["hdd_semana"]
    + BETA_HELADA * df_sem["heladas"]
    + BETA_ENSO   * df_sem["enso"].clip(lower=0)  # solo El Niño positivo
)

# Ruido AR(1) para autocorrelación operativa
ruido = np.zeros(n)
eps   = np.random.normal(0, SIGMA_RUIDO, n)
ruido[0] = eps[0]
for i in range(1, n):
    ruido[i] = 0.45 * ruido[i-1] + eps[i]

df_sem["rendimiento_usd"] = (rendimiento_base + ruido).clip(lower=0).round(2)

# Variable de pérdida relativa (%)
df_sem["perdida_pct"] = ((R_BASE - df_sem["rendimiento_usd"]) / R_BASE * 100).round(2)

# Déficit de temperatura respecto al umbral (solo valores negativos = frío)
T_UMBRAL = 10.0
df_sem["deficit_temp"] = (df_sem["t_min_media"] - T_UMBRAL)  # negativo = más frío

print("="*65)
print("ESTADÍSTICAS DE RENDIMIENTO SEMANAL — Exportadora de Flores")
print("="*65)
print(f"  Semanas simuladas:          {n}")
print(f"  Ingreso base (R_base):      USD {R_BASE:,.0f}/semana")
print(f"  Ingreso medio real:         USD {df_sem['rendimiento_usd'].mean():,.0f}/semana")
print(f"  Ingreso mínimo:             USD {df_sem['rendimiento_usd'].min():,.0f}/semana")
print(f"  Desv. estándar:             USD {df_sem['rendimiento_usd'].std():,.0f}/semana")
print(f"  Pérdida promedio:           {df_sem['perdida_pct'].mean():.1f}%")
print(f"  Pérdida máxima en semana:   {df_sem['perdida_pct'].max():.1f}%")
print(f"  Semanas con pérdida > 20%:  {(df_sem['perdida_pct'] > 20).sum()}")


# ═══════════════════════════════════════════════════════════════════
# 3. REGRESIÓN LINEAL SIMPLE: rendimiento ~ déficit temperatura
# ═══════════════════════════════════════════════════════════════════
X_simple = df_sem["t_min_media"].values
Y        = df_sem["rendimiento_usd"].values

# scipy — regresión rápida
slope_s, intercept_s, r_value, p_value, std_err = stats.linregress(X_simple, Y)

# statsmodels — regresión completa con tabla de resultados
X_sm = sm.add_constant(df_sem["t_min_media"])
modelo_simple = sm.OLS(Y, X_sm).fit()

print("\n" + "="*65)
print("REGRESIÓN LINEAL SIMPLE")
print("  Rendimiento = β₀ + β₁ · T_mín_media_semana + ε")
print("="*65)
print(modelo_simple.summary())

print(f"\n  INTERPRETACIÓN ECONÓMICA:")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Por cada 1°C que SUBE la T mínima:  +USD {slope_s:,.0f}/semana")
print(f"  Por cada 1°C que BAJA la T mínima:  -USD {abs(slope_s):,.0f}/semana")
print(f"  R² = {r_value**2:.4f}  → el {r_value**2*100:.1f}% de la varianza del")
print(f"        rendimiento es explicada por la temperatura")
print(f"  p-value = {p_value:.2e}  → {'SIGNIFICATIVO' if p_value < 0.05 else 'NO significativo'} al 5%")


# ═══════════════════════════════════════════════════════════════════
# 4. REGRESIÓN MÚLTIPLE: rendimiento ~ T + HDD + heladas + ENSO
# ═══════════════════════════════════════════════════════════════════
X_multi = sm.add_constant(df_sem[["t_min_media","hdd_semana","heladas","enso"]])
modelo_multi = sm.OLS(Y, X_multi).fit()

print("\n" + "="*65)
print("REGRESIÓN MÚLTIPLE")
print("  Rendimiento = β₀ + β₁·T + β₂·HDD + β₃·Heladas + β₄·ENSO + ε")
print("="*65)
print(modelo_multi.summary())

coefs = modelo_multi.params
print(f"\n  COEFICIENTES RECUPERADOS vs VERDADEROS:")
print(f"  {'Variable':<20} {'β estimado':>14} {'β verdadero':>14}")
print(f"  {'─'*50}")
print(f"  {'HDD semanal':<20} {coefs['hdd_semana']:>14,.1f} {BETA_HDD:>14,.0f}")
print(f"  {'Días helada':<20} {coefs['heladas']:>14,.1f} {BETA_HELADA:>14,.0f}")
print(f"  {'Índice ENSO':<20} {coefs['enso']:>14,.1f} {BETA_ENSO:>14,.0f}")


# ═══════════════════════════════════════════════════════════════════
# 5. CUANTIFICACIÓN DEL RIESGO FINANCIERO
# ═══════════════════════════════════════════════════════════════════
perdida_por_grado_anual = abs(slope_s) * 52  # semanas/año

# VaR 5% — pérdida máxima esperada en semana mala
var_5pct = np.percentile(df_sem["rendimiento_usd"], 5)
perdida_var = R_BASE - var_5pct

# Expected Shortfall (CVaR) — promedio de las peores semanas
cvar = R_BASE - df_sem[df_sem["rendimiento_usd"] <= var_5pct]["rendimiento_usd"].mean()

print("\n" + "="*65)
print("MÉTRICAS DE RIESGO FINANCIERO")
print("="*65)
print(f"  Pérdida por 1°C de caída T (semanal):  USD {abs(slope_s):,.0f}")
print(f"  Pérdida por 1°C de caída T (anual):    USD {perdida_por_grado_anual:,.0f}")
print(f"  VaR 5% semanal:                        USD {perdida_var:,.0f}")
print(f"  CVaR 5% semanal (Expected Shortfall):  USD {cvar:,.0f}")
print(f"  Prima justa derivado (aprox 4%/año):   USD {perdida_por_grado_anual * 0.04:,.0f}")


# ═══════════════════════════════════════════════════════════════════
# 6. FIGURAS
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14), facecolor=C_FONDO)
fig.suptitle("Agro-Risk Pro  ·  Temperatura vs Rendimiento de Exportación de Flores",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.32,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

# ── Panel A: Serie temporal dual ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1b = ax1.twinx()

ax1.fill_between(df_sem["fecha"], df_sem["rendimiento_usd"],
                 alpha=0.25, color=C_VERDE)
ax1.plot(df_sem["fecha"], df_sem["rendimiento_usd"],
         color=C_VERDE, lw=0.8, label="Rendimiento USD/semana")
ax1.axhline(R_BASE, color=C_VERDE, lw=1, ls="--", alpha=0.6,
            label=f"Base USD {R_BASE:,}")

ax1b.plot(df_sem["fecha"], df_sem["t_min_media"],
          color=C_AZUL, lw=0.9, alpha=0.8, label="T mínima media (°C)")
ax1b.axhline(T_UMBRAL, color=C_NARANJA, lw=1, ls=":", alpha=0.7)

ax1.set_ylabel("Rendimiento (USD/semana)", color=C_VERDE, fontsize=10)
ax1b.set_ylabel("T mínima media (°C)", color=C_AZUL, fontsize=10)
ax1b.tick_params(colors=C_AZUL)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, fontsize=8.5, framealpha=0.2, loc="lower left", ncol=3)
ax1.set_title("A  Serie Temporal: Rendimiento de Exportación y Temperatura 2015–2024",
              fontsize=10, pad=8)
ax1.grid(alpha=0.2)

# ── Panel B: Scatter regresión simple ───────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])

# Colorear puntos por intensidad de pérdida
perdida = df_sem["perdida_pct"].clip(0, 50)
sc = ax2.scatter(df_sem["t_min_media"], df_sem["rendimiento_usd"],
                 c=perdida, cmap="RdYlGn_r", s=12, alpha=0.65, zorder=2)

# Línea de regresión
x_line = np.linspace(df_sem["t_min_media"].min(), df_sem["t_min_media"].max(), 200)
y_line = intercept_s + slope_s * x_line
ax2.plot(x_line, y_line, color=C_ROJO, lw=2, zorder=3,
         label=f"ŷ = {intercept_s:,.0f} + {slope_s:,.0f}·T")

# Banda de confianza 95%
pred = modelo_simple.get_prediction(sm.add_constant(x_line))
ci   = pred.conf_int(alpha=0.05)
ax2.fill_between(x_line, ci[:,0], ci[:,1], alpha=0.15, color=C_ROJO,
                 label="IC 95%")

ax2.axvline(T_UMBRAL, color=C_NARANJA, lw=1.2, ls="--", alpha=0.7,
            label=f"Umbral {T_UMBRAL}°C")
ax2.axvline(4.0, color=C_ROJO, lw=1, ls=":", alpha=0.6, label="Riesgo helada 4°C")

cb2 = fig.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
cb2.set_label("Pérdida (%)", fontsize=8)
plt.setp(cb2.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)

ax2.set_xlabel("T mínima media semanal (°C)"); ax2.set_ylabel("Rendimiento (USD)")
ax2.set_title(f"B  Scatter + Regresión  (R²={r_value**2:.3f})", fontsize=10, pad=8)
ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(alpha=0.3)

# ── Panel C: Residuos ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
residuos = modelo_simple.resid
ax3.scatter(df_sem["t_min_media"], residuos,
            s=8, color=C_AZUL, alpha=0.5)
ax3.axhline(0, color=C_NARANJA, lw=1.5, ls="--")
ax3.axhline( 2*residuos.std(), color=C_ROJO, lw=1, ls=":", alpha=0.7, label="±2σ")
ax3.axhline(-2*residuos.std(), color=C_ROJO, lw=1, ls=":", alpha=0.7)
ax3.set_xlabel("T mínima media semanal (°C)"); ax3.set_ylabel("Residuo (USD)")
ax3.set_title("C  Residuos vs Temperatura", fontsize=10, pad=8)
ax3.legend(fontsize=8, framealpha=0.2); ax3.grid(alpha=0.3)

# ── Panel D: Pérdida marginal por grado ──────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
temp_escenarios = np.arange(-2, T_UMBRAL + 0.1, 0.5)
perdida_marginal = [(T_UMBRAL - t) * abs(slope_s) for t in temp_escenarios]
colores_barra = [C_ROJO if t < 2 else C_NARANJA if t < 4 else C_AZUL
                 for t in temp_escenarios]
bars = ax4.bar(temp_escenarios, perdida_marginal, width=0.4,
               color=colores_barra, alpha=0.85, edgecolor="none")
ax4.axvline(4.0, color=C_NARANJA, lw=1.2, ls="--", alpha=0.7, label="Riesgo 4°C")
ax4.axvline(2.0, color=C_ROJO,    lw=1.2, ls=":",  alpha=0.7, label="Destructiva 2°C")
for bar, val in zip(bars, perdida_marginal):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
             f"${val:,.0f}", ha="center", fontsize=7, rotation=45,
             color=C_TEXTO)
ax4.set_xlabel("Temperatura mínima (°C)"); ax4.set_ylabel("Pérdida vs base (USD/semana)")
ax4.set_title("D  Pérdida Marginal por Temperatura (β₁ × ΔT)", fontsize=10, pad=8)
ax4.legend(fontsize=8, framealpha=0.2); ax4.grid(axis="y", alpha=0.3)

# ── Panel E: Distribución rendimientos + VaR ────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.hist(df_sem["rendimiento_usd"], bins=50, color=C_AZUL,
         alpha=0.65, edgecolor="none", label="Distribución rendimientos")
ax5.axvline(R_BASE,   color=C_VERDE,   lw=2,   ls="-",  label=f"Base ${R_BASE:,}")
ax5.axvline(var_5pct, color=C_NARANJA, lw=1.8, ls="--",
            label=f"VaR 5%  ${var_5pct:,.0f}")
ax5.axvline(df_sem["rendimiento_usd"].mean() - cvar, color=C_ROJO, lw=1.5, ls=":",
            label=f"CVaR 5%  −${cvar:,.0f}")
# Zona de pérdida
ax5.axvspan(0, var_5pct, alpha=0.12, color=C_ROJO)
ax5.set_xlabel("Rendimiento (USD/semana)"); ax5.set_ylabel("Frecuencia")
ax5.set_title("E  Distribución Rendimientos y Métricas de Riesgo", fontsize=10, pad=8)
ax5.legend(fontsize=8, framealpha=0.2); ax5.grid(axis="y", alpha=0.3)

plt.savefig("outputs/fig3_regresion.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()
print("\n✅ fig3_regresion.png guardada")

# Exportar tabla semanal
df_sem.to_csv("outputs/06_rendimientos_semanales.csv", index=False)
print("✅ 06_rendimientos_semanales.csv exportado")
