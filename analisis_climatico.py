"""
Agro-Risk Pro — Análisis Estadístico y Visualización Climática
Sabana de Bogotá 2015–2024
Materia: Programación para Economía y Finanzas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Paleta Agro-Risk Pro ─────────────────────────────────────────────
C_FONDO    = "#0f1117"
C_PANEL    = "#1a1d27"
C_TEXTO    = "#e8eaf0"
C_SUB      = "#8b90a0"
C_AZUL     = "#4f8ef7"
C_VERDE    = "#2ecc71"
C_NARANJA  = "#f39c12"
C_ROJO     = "#e74c3c"
C_GRID     = "#2a2d3a"

plt.rcParams.update({
    "figure.facecolor": C_FONDO, "axes.facecolor": C_PANEL,
    "axes.edgecolor":   C_GRID,  "axes.labelcolor": C_TEXTO,
    "axes.titlecolor":  C_TEXTO, "xtick.color":     C_SUB,
    "ytick.color":      C_SUB,   "text.color":      C_TEXTO,
    "grid.color":       C_GRID,  "grid.linewidth":  0.6,
    "font.size": 10,
})

UMBRAL_RIESGO = 4.0
UMBRAL_FROST  = 2.0
MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# ── Cargar datos ─────────────────────────────────────────────────────
df = pd.read_csv("outputs/temperatura_sabana_10años.csv",
                 parse_dates=["fecha"])

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 1 — ESTADÍSTICAS POR MES
# ═══════════════════════════════════════════════════════════════════
stats = df.groupby("mes")["t_minima_c"].agg(
    media   = "mean",
    std     = "std",
    p5      = lambda x: np.percentile(x, 5),
    p25     = lambda x: np.percentile(x, 25),
    mediana = "median",
    p75     = lambda x: np.percentile(x, 75),
    p95     = lambda x: np.percentile(x, 95),
    n_dias  = "count"
).round(2)
stats.index = MESES

riesgo = df[df["t_minima_c"] < UMBRAL_RIESGO].groupby("mes").size().reindex(range(1,13), fill_value=0)
frost  = df[df["t_minima_c"] < UMBRAL_FROST ].groupby("mes").size().reindex(range(1,13), fill_value=0)
stats["dias_riesgo"]     = riesgo.values
stats["dias_destructiva"]= frost.values
stats["prob_pct"]        = (riesgo.values / stats["n_dias"].values * 100).round(1)

total_riesgo = (df["t_minima_c"] < UMBRAL_RIESGO).sum()
total_frost  = (df["t_minima_c"] < UMBRAL_FROST ).sum()

print("="*65)
print("ESTADÍSTICAS T_MÍNIMA POR MES — Sabana de Bogotá 2015–2024")
print("="*65)
print(stats.to_string())
print(f"\nDías T_min < {UMBRAL_RIESGO}°C (riesgo):        {total_riesgo:,}  ({total_riesgo/len(df)*100:.1f}%)")
print(f"Días T_min < {UMBRAL_FROST}°C  (destructiva):   {total_frost:,}   ({total_frost/len(df)*100:.1f}%)")
print(f"\nMes más crítico: {stats['dias_riesgo'].idxmax()}  ({stats['dias_riesgo'].max()} días en 10 años)")

# ═══════════════════════════════════════════════════════════════════
# FIGURA 1 — DASHBOARD ESTADÍSTICO (4 paneles)
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12), facecolor=C_FONDO)
fig.suptitle("Agro-Risk Pro  ·  Análisis Estadístico Climático — Sabana de Bogotá 2015–2024",
             fontsize=14, fontweight="bold", y=0.99)
gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.30,
                        left=0.07, right=0.97, top=0.93, bottom=0.07)
x   = np.arange(12)

# Panel A — Boxplot
ax1 = fig.add_subplot(gs[0, 0])
bp  = ax1.boxplot(
    [df[df["mes"]==m]["t_minima_c"].values for m in range(1,13)],
    patch_artist=True,
    medianprops   = dict(color=C_NARANJA, linewidth=2),
    whiskerprops  = dict(color=C_SUB, linewidth=1),
    capprops      = dict(color=C_SUB, linewidth=1.5),
    flierprops    = dict(marker=".", color=C_ROJO, alpha=0.3, markersize=3),
    boxprops      = dict(facecolor=C_AZUL+"44", edgecolor=C_AZUL, linewidth=1.2)
)
ax1.axhline(UMBRAL_RIESGO, color=C_NARANJA, lw=1.5, ls="--", label=f"Riesgo {UMBRAL_RIESGO}°C")
ax1.axhline(UMBRAL_FROST,  color=C_ROJO,    lw=1.5, ls=":",  label=f"Destructiva {UMBRAL_FROST}°C")
ax1.set_xticks(range(1,13)); ax1.set_xticklabels(MESES, fontsize=8.5)
ax1.set_title("A  Distribución T Mínima por Mes", fontsize=10, pad=8)
ax1.set_ylabel("°C"); ax1.legend(fontsize=8, framealpha=0.2); ax1.grid(axis="y", alpha=0.4)

# Panel B — Barras días de helada
ax2  = fig.add_subplot(gs[0, 1])
bw   = 0.38
b1   = ax2.bar(x-bw/2, stats["dias_riesgo"],     bw, color=C_NARANJA, alpha=0.85, label=f"Riesgo (<{UMBRAL_RIESGO}°C)")
b2   = ax2.bar(x+bw/2, stats["dias_destructiva"], bw, color=C_ROJO,    alpha=0.85, label=f"Destructiva (<{UMBRAL_FROST}°C)")
for bar, col in [(b1, C_NARANJA), (b2, C_ROJO)]:
    for b in bar:
        h = b.get_height()
        if h > 0:
            ax2.text(b.get_x()+b.get_width()/2, h+0.4, str(int(h)),
                     ha="center", fontsize=7.5, color=col)
ax2.set_xticks(x); ax2.set_xticklabels(MESES, fontsize=8.5)
ax2.set_title("B  Días de Helada por Mes (10 años)", fontsize=10, pad=8)
ax2.set_ylabel("Número de días"); ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(axis="y", alpha=0.4)

# Panel C — Banda de percentiles
ax3 = fig.add_subplot(gs[1, 0])
ax3.fill_between(x, stats["p5"],  stats["p95"], alpha=0.15, color=C_AZUL, label="P5–P95")
ax3.fill_between(x, stats["p25"], stats["p75"], alpha=0.30, color=C_AZUL, label="P25–P75")
ax3.plot(x, stats["mediana"], color=C_AZUL,   lw=2,   label="Mediana")
ax3.plot(x, stats["media"],   color=C_VERDE,  lw=1.5, ls="--", label="Media")
ax3.axhline(UMBRAL_RIESGO, color=C_NARANJA, lw=1.2, ls="--", alpha=0.7)
ax3.axhline(UMBRAL_FROST,  color=C_ROJO,    lw=1.2, ls=":",  alpha=0.7)
ax3.set_xticks(x); ax3.set_xticklabels(MESES, fontsize=8.5)
ax3.set_title("C  Percentiles T Mínima Mensual", fontsize=10, pad=8)
ax3.set_ylabel("°C"); ax3.legend(fontsize=8, framealpha=0.2, ncol=2); ax3.grid(axis="y", alpha=0.4)

# Panel D — Mapa de calor año × mes
ax4   = fig.add_subplot(gs[1, 1])
años  = sorted(df["anio"].unique())
mat   = np.zeros((len(años), 12))
for i, año in enumerate(años):
    for j, mes in enumerate(range(1,13)):
        mask    = (df["anio"]==año) & (df["mes"]==mes)
        total   = mask.sum()
        mat[i,j]= (df[mask]["t_minima_c"] < UMBRAL_RIESGO).sum() / total * 100 if total else 0
im = ax4.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=mat.max())
ax4.set_xticks(range(12)); ax4.set_xticklabels(MESES, fontsize=8.5)
ax4.set_yticks(range(len(años))); ax4.set_yticklabels(años, fontsize=8.5)
ax4.set_title("D  Probabilidad de Riesgo (%) Año × Mes", fontsize=10, pad=8)
cb = fig.colorbar(im, ax=ax4, fraction=0.03, pad=0.03)
cb.set_label("% días con riesgo", fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)

plt.savefig("outputs/fig1_estadisticas.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()
print("\n✅ fig1_estadisticas.png guardada")


# ═══════════════════════════════════════════════════════════════════
# FIGURA 2 — SCATTER DE ANOMALÍAS
# ═══════════════════════════════════════════════════════════════════
fig2, (ax_main, ax_bar) = plt.subplots(
    2, 1, figsize=(18, 11), facecolor=C_FONDO,
    gridspec_kw={"height_ratios":[3,1], "hspace":0.08}
)
fig2.suptitle("Agro-Risk Pro  ·  Scatter de Anomalías y Eventos de Helada 2015–2024",
              fontsize=13, fontweight="bold", y=0.99)

# Puntos normales
nm = df["t_minima_c"] >= UMBRAL_RIESGO
ax_main.scatter(df.loc[nm,"fecha"], df.loc[nm,"t_minima_c"],
                s=1.2, color=C_SUB, alpha=0.2, zorder=1, label="Normal (≥4°C)")

# Riesgo
rm = (df["t_minima_c"] >= UMBRAL_FROST) & (df["t_minima_c"] < UMBRAL_RIESGO)
ax_main.scatter(df.loc[rm,"fecha"], df.loc[rm,"t_minima_c"],
                s=10, color=C_NARANJA, alpha=0.85, zorder=3,
                label=f"Riesgo 2–4°C  (n={rm.sum():,})")

# Helada destructiva
fm = df["t_minima_c"] < UMBRAL_FROST
ax_main.scatter(df.loc[fm,"fecha"], df.loc[fm,"t_minima_c"],
                s=22, color=C_ROJO, alpha=0.9, zorder=4,
                label=f"Helada destructiva <2°C  (n={fm.sum():,})")

ax_main.axhline(UMBRAL_RIESGO, color=C_NARANJA, lw=1.2, ls="--", alpha=0.7)
ax_main.axhline(UMBRAL_FROST,  color=C_ROJO,    lw=1.2, ls=":",  alpha=0.7)
ax_main.axhline(df["t_minima_c"].mean(), color=C_VERDE, lw=1, alpha=0.5,
                label=f"Media {df['t_minima_c'].mean():.1f}°C")

# Anotar mínima absoluta
fila = df.loc[df["t_minima_c"].idxmin()]
ax_main.annotate(
    f"Mínima absoluta\n{fila['t_minima_c']:.1f}°C  ({fila['fecha'].strftime('%b %Y')})",
    xy=(fila["fecha"], fila["t_minima_c"]),
    xytext=(50, 30), textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color=C_ROJO, lw=1.2),
    fontsize=8.5, color=C_ROJO,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=C_PANEL, edgecolor=C_ROJO, alpha=0.85)
)

# Zonas críticas sombreadas
for año in años:
    for ini, fin in [("02-01","04-30"),("10-01","12-31")]:
        ax_main.axvspan(pd.Timestamp(f"{año}-{ini}"), pd.Timestamp(f"{año}-{fin}"),
                        alpha=0.04, color=C_NARANJA, zorder=0)

ax_main.set_xlim(df["fecha"].min(), df["fecha"].max())
ax_main.set_ylabel("Temperatura Mínima (°C)", fontsize=11)
ax_main.legend(fontsize=9, framealpha=0.25, markerscale=2.5, loc="upper left", ncol=2)
ax_main.grid(alpha=0.2); ax_main.tick_params(labelbottom=False)

# Panel inferior — conteo mensual
def conteo_mensual(mask):
    sub = df[mask].copy()
    grp = sub.groupby(sub["fecha"].dt.to_period("M")).size()
    return grp.index.to_timestamp(), grp.values

ri, rv = conteo_mensual(rm | fm)
fi, fv = conteo_mensual(fm)
ax_bar.bar(ri, rv, width=25, color=C_NARANJA, alpha=0.7, label="Días riesgo/mes")
ax_bar.bar(fi, fv, width=25, color=C_ROJO,    alpha=0.9, label="Días destructiva/mes")
ax_bar.set_xlim(df["fecha"].min(), df["fecha"].max())
ax_bar.set_ylabel("Días/mes", fontsize=9)
ax_bar.set_xlabel("Fecha", fontsize=10)
ax_bar.legend(fontsize=8, framealpha=0.25, loc="upper right")
ax_bar.grid(axis="y", alpha=0.25)

plt.savefig("outputs/fig2_scatter_anomalias.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()
print("✅ fig2_scatter_anomalias.png guardada")
print("\nAnálisis completado.")
