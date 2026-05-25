"""
Agro-Risk Pro — Simulación Monte Carlo
10,000 escenarios climáticos · 3 meses futuros
Materia: Programación para Economía y Finanzas

MODELO:
  T(t) = μ_estacional(t) + σ_hist × Z(t)   donde Z ~ N(0,1) con AR(1)
  HDD(t) = max(0, 10 - T(t))
  Payoff = min(max(0, HDD_90dias - Strike), Cap - Strike) × Tick
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
C_MORADO = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor": C_FONDO, "axes.facecolor": C_PANEL,
    "axes.edgecolor":   C_GRID,  "axes.labelcolor": C_TEXTO,
    "axes.titlecolor":  C_TEXTO, "xtick.color":     C_SUB,
    "ytick.color":      C_SUB,   "text.color":      C_TEXTO,
    "grid.color":       C_GRID,  "grid.linewidth":  0.5, "font.size": 10,
})

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# 1. CALIBRACIÓN CON DATOS HISTÓRICOS
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv("outputs/temperatura_sabana_10años.csv",
                 parse_dates=["fecha"])

UMBRAL    = 10.0
STRIKE    = 38.0
CAP       = 65.0
TICK      = 250.0
PRIMA     = 580.0
N_SIM     = 10_000
N_DIAS    = 90       # 3 meses
HOY       = pd.Timestamp("2025-01-01")
MESES_SIM = [1, 2, 3]  # calibrado al HDD trimestral histórico   # enero–marzo: temporada fría

# Media y sigma diaria por día del año
df["dia_año"] = df["fecha"].dt.dayofyear.clip(1, 365)
calibracion   = df.groupby("dia_año")["t_promedio_c"].agg(
    media = "mean",
    sigma = "std"
).reindex(range(1, 366)).interpolate()

# Autocorrelación AR(1) de los residuos
df_sorted  = df.sort_values("fecha")
residuos   = df_sorted["t_promedio_c"].values - \
             calibracion.loc[df_sorted["dia_año"].values, "media"].values
ar1_coef   = np.corrcoef(residuos[:-1], residuos[1:])[0, 1]
sigma_ruido= np.std(residuos) * np.sqrt(1 - ar1_coef**2)

# Días futuros a simular
fechas_sim  = pd.date_range(start=HOY, periods=N_DIAS, freq="D")
dias_año    = fechas_sim.dayofyear
mu_vec      = calibracion.loc[dias_año, "media"].values
sigma_vec   = calibracion.loc[dias_año, "sigma"].values

print("=" * 65)
print(f"CALIBRACIÓN DEL MODELO (datos 2015–2024)")
print("=" * 65)
print(f"  AR(1) coeficiente:     {ar1_coef:.4f}")
print(f"  σ ruido diario:        {sigma_ruido:.4f}°C")
print(f"  T media ene–mar:       {mu_vec.mean():.2f}°C")
print(f"  T sigma promedio:      {sigma_vec.mean():.2f}°C")
print(f"  Período simulado:      {HOY.date()} → {fechas_sim[-1].date()}")
print(f"  Días simulados:        {N_DIAS}")
print(f"  Escenarios:            {N_SIM:,}")
print(f"\nContrato valuado:")
print(f"  Umbral HDD:  {UMBRAL}°C   Strike: {STRIKE}   Cap: {CAP}")
print(f"  Tick: USD {TICK}   Prima pagada: USD {PRIMA:,.2f}")


# ═══════════════════════════════════════════════════════════════════
# 2. SIMULACIÓN MONTE CARLO
# ═══════════════════════════════════════════════════════════════════
# Matriz temperatura: N_SIM filas × N_DIAS columnas
# Proceso AR(1): T(t) = mu(t) + rho*e(t-1) + eps(t)

eps     = np.random.normal(0, sigma_ruido, (N_SIM, N_DIAS))
T_sim   = np.zeros((N_SIM, N_DIAS))
e_prev  = np.zeros(N_SIM)

for t in range(N_DIAS):
    T_sim[:, t] = mu_vec[t] + ar1_coef * e_prev + eps[:, t]
    e_prev      = T_sim[:, t] - mu_vec[t]

# HDD diario por escenario
HDD_diario = np.maximum(0.0, UMBRAL - T_sim)         # (N_SIM × N_DIAS)

# HDD acumulado al final de los 90 días
HDD_total  = HDD_diario.sum(axis=1)                  # (N_SIM,)

# Payoff del derivado para cada escenario
payoff_bruto = np.maximum(0.0, HDD_total - STRIKE)
payoff_cap   = np.minimum(payoff_bruto, CAP - STRIKE)
payoff_usd   = payoff_cap * TICK                      # (N_SIM,)

# P&L neto del exportador (payoff recibido - prima pagada)
pnl_neto     = payoff_usd - PRIMA

print(f"\n{'='*65}")
print("RESULTADOS DE LA SIMULACIÓN")
print(f"{'='*65}")
print(f"\n  HDD acumulado (90 días):")
print(f"    Media:       {HDD_total.mean():.2f} °C·día")
print(f"    Mediana:     {np.median(HDD_total):.2f} °C·día")
print(f"    Desv. std:   {HDD_total.std():.2f} °C·día")
print(f"    P5/P95:      {np.percentile(HDD_total,5):.1f} / {np.percentile(HDD_total,95):.1f}")
print(f"    Min/Max:     {HDD_total.min():.1f} / {HDD_total.max():.1f}")
print(f"    Strike {STRIKE}: {(HDD_total >= STRIKE).mean()*100:.1f}% de escenarios lo superan")
print(f"    Cap {CAP}:     {(HDD_total >= CAP).mean()*100:.1f}% de escenarios lo superan")

print(f"\n  Payoff del derivado:")
print(f"    P(activación):        {(payoff_usd > 0).mean()*100:.1f}%")
print(f"    Valor esperado E[P]:  USD {payoff_usd.mean():,.2f}")
print(f"    Mediana payoff:       USD {np.median(payoff_usd):,.2f}")
print(f"    Payoff máx posible:   USD {(CAP-STRIKE)*TICK:,.0f}")
print(f"    Payoff P95:           USD {np.percentile(payoff_usd,95):,.2f}")

print(f"\n  P&L neto (payoff − prima USD {PRIMA:,.0f}):")
print(f"    E[P&L]:               USD {pnl_neto.mean():,.2f}")
print(f"    P(P&L > 0):           {(pnl_neto > 0).mean()*100:.1f}%")
print(f"    VaR 95% (pérdida máx):{np.percentile(pnl_neto,5):,.2f}")
print(f"    CVaR 95%:             {pnl_neto[pnl_neto <= np.percentile(pnl_neto,5)].mean():,.2f}")


# ═══════════════════════════════════════════════════════════════════
# 3. ANÁLISIS POR ESCENARIO (fan chart)
# ═══════════════════════════════════════════════════════════════════
HDD_acum = HDD_diario.cumsum(axis=1)   # acumulado día a día

p_bandas = [5, 10, 25, 50, 75, 90, 95]
fandata  = {p: np.percentile(HDD_acum, p, axis=0) for p in p_bandas}

# Temperatura: 50 trayectorias para el fan chart
idx_muestra = np.random.choice(N_SIM, 50, replace=False)

# Escenarios extremos
idx_max_hdd = np.argmax(HDD_total)
idx_min_hdd = np.argmin(HDD_total)
idx_med_hdd = np.argmin(np.abs(HDD_total - np.median(HDD_total)))


# ═══════════════════════════════════════════════════════════════════
# 4. VISUALIZACIÓN — 6 PANELES
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 16), facecolor=C_FONDO)
fig.suptitle(
    f"Agro-Risk Pro  ·  Monte Carlo {N_SIM:,} escenarios · "
    f"{HOY.strftime('%d %b %Y')} + 90 días",
    fontsize=14, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

# ── Panel A: Fan chart HDD acumulado ────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
dias_eje = np.arange(N_DIAS)

ax1.fill_between(dias_eje, fandata[5],  fandata[95],
                 alpha=0.12, color=C_AZUL, label="P5–P95")
ax1.fill_between(dias_eje, fandata[10], fandata[90],
                 alpha=0.18, color=C_AZUL, label="P10–P90")
ax1.fill_between(dias_eje, fandata[25], fandata[75],
                 alpha=0.28, color=C_AZUL, label="P25–P75")
ax1.plot(dias_eje, fandata[50], color=C_AZUL, lw=2.5,
         label=f"Mediana  HDD={fandata[50][-1]:.0f}")

# Escenarios extremos
ax1.plot(dias_eje, HDD_acum[idx_max_hdd],
         color=C_ROJO, lw=1.2, ls="--", alpha=0.8,
         label=f"Escenario máx  HDD={HDD_total[idx_max_hdd]:.0f}")
ax1.plot(dias_eje, HDD_acum[idx_min_hdd],
         color=C_VERDE, lw=1.2, ls="--", alpha=0.8,
         label=f"Escenario mín  HDD={HDD_total[idx_min_hdd]:.0f}")

ax1.axhline(STRIKE, color=C_NARANJA, lw=1.8, ls="--",
            label=f"Strike {STRIKE:.0f}")
ax1.axhline(CAP,    color=C_ROJO,    lw=1.2, ls=":",
            label=f"Cap {CAP:.0f}")
ax1.axhspan(STRIKE, CAP, alpha=0.06, color=C_ROJO, label="Zona pago")

# Etiquetas en eje x: fechas
xticks = np.arange(0, N_DIAS, 15)
xlabs  = [fechas_sim[i].strftime("%d %b") for i in xticks]
ax1.set_xticks(xticks); ax1.set_xticklabels(xlabs, fontsize=9)
ax1.set_ylabel("HDD acumulado (°C·día)"); ax1.set_xlabel("Fecha")
ax1.set_title("A  Fan Chart HDD Acumulado — 10,000 Escenarios", fontsize=10, pad=8)
ax1.legend(fontsize=8.5, framealpha=0.2, ncol=4, loc="upper left")
ax1.grid(alpha=0.2)

# ── Panel B: Histograma HDD final + distribución ─────────────────
ax2 = fig.add_subplot(gs[1, 0])
n_bins = 80
cnt, bins, patches = ax2.hist(HDD_total, bins=n_bins,
                               color=C_AZUL, alpha=0.65,
                               edgecolor="none", density=True)

# Colorear zona de activación
for patch, left in zip(patches, bins[:-1]):
    if left >= STRIKE:
        patch.set_facecolor(C_ROJO)
        patch.set_alpha(0.80)
    elif left >= STRIKE * 0.9:
        patch.set_facecolor(C_NARANJA)
        patch.set_alpha(0.75)

# Curva normal ajustada
mu_hdd, sig_hdd = HDD_total.mean(), HDD_total.std()
x_norm = np.linspace(HDD_total.min(), HDD_total.max(), 300)
ax2.plot(x_norm, stats.norm.pdf(x_norm, mu_hdd, sig_hdd),
         color=C_VERDE, lw=2, label=f"Normal ajustada\nμ={mu_hdd:.1f}  σ={sig_hdd:.1f}")

ax2.axvline(STRIKE,               color=C_NARANJA, lw=1.8, ls="--",
            label=f"Strike {STRIKE:.0f}")
ax2.axvline(CAP,                  color=C_ROJO,    lw=1.5, ls=":",
            label=f"Cap {CAP:.0f}")
ax2.axvline(np.median(HDD_total), color=C_AZUL,    lw=1.5,
            label=f"Mediana {np.median(HDD_total):.1f}")

ax2.set_xlabel("HDD acumulado 90 días (°C·día)")
ax2.set_ylabel("Densidad")
ax2.set_title("B  Distribución HDD Final — Rojo = derivado activo", fontsize=10, pad=8)
ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(axis="y", alpha=0.3)

# ── Panel C: Histograma PAYOFF ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
p_activa = (payoff_usd > 0).mean() * 100
p_cero   = 100 - p_activa

# Barra de cero separada
ax3.bar(0, p_cero / 100 * N_SIM / (N_SIM / n_bins),
        width=(CAP-STRIKE)*TICK / n_bins * 2,
        color=C_AZUL, alpha=0.55, label=f"Sin pago  ({p_cero:.1f}%)")

payoff_pos = payoff_usd[payoff_usd > 0]
if len(payoff_pos) > 0:
    ax3.hist(payoff_pos, bins=50, color=C_ROJO, alpha=0.75,
             edgecolor="none", density=False,
             label=f"Con pago  ({p_activa:.1f}%)")

ax3.axvline(payoff_usd.mean(), color=C_VERDE, lw=2,
            label=f"E[Payoff]  USD {payoff_usd.mean():,.0f}")
ax3.axvline(PRIMA, color=C_NARANJA, lw=1.8, ls="--",
            label=f"Prima  USD {PRIMA:,.0f}")

ax3.set_xlabel("Payoff (USD)")
ax3.set_ylabel("Frecuencia")
ax3.set_title("C  Histograma Payoff del Derivado", fontsize=10, pad=8)
ax3.legend(fontsize=8, framealpha=0.2); ax3.grid(axis="y", alpha=0.3)

# ── Panel D: P&L neto (payoff - prima) ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
var95  = np.percentile(pnl_neto, 5)
cvar95 = pnl_neto[pnl_neto <= var95].mean()

cnt4, bins4, patches4 = ax4.hist(pnl_neto, bins=80,
                                  color=C_AZUL, alpha=0.65,
                                  edgecolor="none", density=False)
# Colorear pérdidas de rojo
for patch, left in zip(patches4, bins4[:-1]):
    if left < 0:
        patch.set_facecolor(C_ROJO)
        patch.set_alpha(0.70)
    else:
        patch.set_facecolor(C_VERDE)
        patch.set_alpha(0.65)

ax4.axvline(0,               color=C_TEXTO,   lw=1.5, ls="-",
            label="Break-even")
ax4.axvline(pnl_neto.mean(), color=C_VERDE,   lw=2,
            label=f"E[P&L]  USD {pnl_neto.mean():,.0f}")
ax4.axvline(var95,           color=C_NARANJA, lw=1.8, ls="--",
            label=f"VaR 95%  USD {var95:,.0f}")
ax4.axvline(cvar95,          color=C_ROJO,    lw=1.5, ls=":",
            label=f"CVaR 95%  USD {cvar95:,.0f}")

ax4.set_xlabel("P&L neto (USD)")
ax4.set_ylabel("Frecuencia")
ax4.set_title("D  P&L Neto del Exportador (Payoff − Prima)", fontsize=10, pad=8)
ax4.legend(fontsize=8, framealpha=0.2); ax4.grid(axis="y", alpha=0.3)

# ── Panel E: Fan chart TEMPERATURA ──────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
T_p  = {p: np.percentile(T_sim, p, axis=0) for p in [5, 25, 50, 75, 95]}

ax5.fill_between(dias_eje, T_p[5],  T_p[95],
                 alpha=0.12, color=C_MORADO)
ax5.fill_between(dias_eje, T_p[25], T_p[75],
                 alpha=0.25, color=C_MORADO)
ax5.plot(dias_eje, T_p[50],  color=C_MORADO, lw=2,
         label=f"Mediana T {T_p[50].mean():.1f}°C")
ax5.plot(dias_eje, mu_vec,   color=C_NARANJA, lw=1.5,
         ls="--", label="Media histórica")
ax5.plot(dias_eje, T_sim[idx_max_hdd],
         color=C_ROJO, lw=0.9, alpha=0.7,
         label=f"Escenario máx HDD")
ax5.plot(dias_eje, T_sim[idx_min_hdd],
         color=C_VERDE, lw=0.9, alpha=0.7,
         label=f"Escenario mín HDD")
ax5.axhline(UMBRAL, color=C_AZUL, lw=1.5, ls="--",
            label=f"Umbral HDD {UMBRAL}°C")

ax5.set_xticks(xticks); ax5.set_xticklabels(xlabs, fontsize=9)
ax5.set_ylabel("Temperatura promedio (°C)")
ax5.set_xlabel("Fecha")
ax5.set_title("E  Fan Chart Temperatura Simulada", fontsize=10, pad=8)
ax5.legend(fontsize=8, framealpha=0.2); ax5.grid(alpha=0.2)

plt.savefig("outputs/fig7_montecarlo.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()

# ── Exportar resultados ──────────────────────────────────────────
resultados_mc = pd.DataFrame({
    "escenario":      np.arange(1, N_SIM + 1),
    "hdd_90dias":     HDD_total.round(2),
    "payoff_usd":     payoff_usd.round(2),
    "pnl_neto_usd":   pnl_neto.round(2),
    "activado":       (payoff_usd > 0).astype(int),
    "cap_alcanzado":  (HDD_total >= CAP).astype(int),
})
resultados_mc.to_csv("outputs/10_montecarlo_resultados.csv",
                     index=False)

print(f"\n{'='*65}")
print("RESUMEN EJECUTIVO PARA EL MÓDULO IA")
print(f"{'='*65}")
print(f"  Escenarios simulados:      {N_SIM:,}")
print(f"  HDD esperado (90 días):    {HDD_total.mean():.1f} °C·día  "
      f"(IC95: {np.percentile(HDD_total,2.5):.0f}–{np.percentile(HDD_total,97.5):.0f})")
print(f"  P(activar cobertura):      {(payoff_usd>0).mean()*100:.1f}%")
print(f"  Valor esperado cobertura:  USD {payoff_usd.mean():,.2f}")
print(f"  Prima pagada:              USD {PRIMA:,.2f}")
print(f"  E[P&L] neto:               USD {pnl_neto.mean():,.2f}")
print(f"  P(recuperar prima):        {(pnl_neto>0).mean()*100:.1f}%")
print(f"  VaR 95%:                   USD {var95:,.0f}")
print(f"  CVaR 95%:                  USD {cvar95:,.0f}")
print(f"\n✅ fig7_montecarlo.png guardada")
print(f"✅ 10_montecarlo_resultados.csv exportado ({N_SIM:,} filas)")
