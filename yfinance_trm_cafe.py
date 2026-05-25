"""
Agro-Risk Pro — Descarga TRM y Café con yfinance + Unificación con datos climáticos
Materia: Programación para Economía y Finanzas

CÓMO CORRER ESTE ARCHIVO EN TU MÁQUINA:
  pip install yfinance pandas matplotlib
  python yfinance_trm_cafe.py

TICKERS USADOS:
  USDCOP=X  → Tipo de cambio USD/COP (TRM aproximada Yahoo Finance)
  KC=F      → Futuros de Café Arábica en NY (ICE, cotizado en USD/libra)

NOTA ACADÉMICA:
  Yahoo Finance entrega el precio "spot" de divisas, no la TRM oficial
  del Banco de la República. Para uso académico es suficiente. Para
  producción usar: https://www.banrep.gov.co/es/estadisticas/trm
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── Paleta ───────────────────────────────────────────────────────────
C_FONDO  = "#0f1117"; C_PANEL = "#1a1d27"; C_TEXTO  = "#e8eaf0"
C_SUB    = "#8b90a0"; C_AZUL  = "#4f8ef7"; C_VERDE  = "#2ecc71"
C_NARANJA= "#f39c12"; C_ROJO  = "#e74c3c"; C_GRID   = "#2a2d3a"
C_MORADO = "#9b59b6"

plt.rcParams.update({
    "figure.facecolor": C_FONDO, "axes.facecolor": C_PANEL,
    "axes.edgecolor":   C_GRID,  "axes.labelcolor": C_TEXTO,
    "axes.titlecolor":  C_TEXTO, "xtick.color":     C_SUB,
    "ytick.color":      C_SUB,   "text.color":      C_TEXTO,
    "grid.color":       C_GRID,  "grid.linewidth":  0.5,
    "font.size": 10,
})

# ═══════════════════════════════════════════════════════════════════
# BLOQUE 1 — DESCARGA CON YFINANCE
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("DESCARGANDO DATOS CON YFINANCE...")
print("=" * 60)

PERIODO  = "5y"       # últimos 5 años
INTERVALO = "1wk"     # frecuencia semanal (alinea con datos climáticos)

# ── 1A. TRM USD/COP ──────────────────────────────────────────────────
"""
yfinance usa USDCOP=X para el tipo de cambio spot.
La columna 'Close' da el precio de cierre del período.
Para frecuencia diaria usar interval="1d".
"""
print("\n[1/2] Descargando TRM USD/COP...")
trm_raw = yf.download(
    tickers  = "USDCOP=X",
    period   = PERIODO,
    interval = INTERVALO,
    auto_adjust = True,
    progress = False
)
# Limpiar MultiIndex si aparece
if isinstance(trm_raw.columns, pd.MultiIndex):
    trm_raw.columns = trm_raw.columns.get_level_values(0)

trm = trm_raw[["Close"]].copy()
trm.columns = ["trm_usdcop"]
trm.index   = pd.to_datetime(trm.index).tz_localize(None)
trm         = trm.dropna()
print(f"   ✅ {len(trm)} semanas descargadas")
print(f"   Rango: {trm.index[0].date()} → {trm.index[-1].date()}")
print(f"   TRM actual aprox: {trm['trm_usdcop'].iloc[-1]:,.0f} COP/USD")

# ── 1B. Café Arábica NY (KC=F) ───────────────────────────────────────
"""
KC=F es el contrato de futuros del café Arábica en ICE New York.
Cotiza en USD/libra (USc/lb — centavos). Multiplicar ×100 para
convertir a USD/libra completos o ×22.04 para USD/kg.
"""
print("\n[2/2] Descargando Café Arábica NY (KC=F)...")
cafe_raw = yf.download(
    tickers  = "KC=F",
    period   = PERIODO,
    interval = INTERVALO,
    auto_adjust = True,
    progress = False
)
if isinstance(cafe_raw.columns, pd.MultiIndex):
    cafe_raw.columns = cafe_raw.columns.get_level_values(0)

cafe = cafe_raw[["Close","Volume"]].copy()
cafe.columns = ["cafe_usd_libra", "cafe_volumen"]
cafe.index   = pd.to_datetime(cafe.index).tz_localize(None)
cafe         = cafe.dropna(subset=["cafe_usd_libra"])

# Conversiones útiles
cafe["cafe_usd_kg"]  = (cafe["cafe_usd_libra"] / 100 * 2.20462).round(4)
cafe["cafe_cop_kg"]  = None  # se calculará al unir con TRM
print(f"   ✅ {len(cafe)} semanas descargadas")
print(f"   Rango: {cafe.index[0].date()} → {cafe.index[-1].date()}")
print(f"   Precio actual aprox: {cafe['cafe_usd_libra'].iloc[-1]:.2f} USc/lb")


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 2 — CARGAR DATOS CLIMÁTICOS Y RENDIMIENTOS
# ═══════════════════════════════════════════════════════════════════
print("\n[3/3] Cargando datos climáticos y rendimientos simulados...")

clima = pd.read_csv("temperatura_sabana_10años.csv", parse_dates=["fecha"])
rend  = pd.read_csv("06_rendimientos_semanales.csv",  parse_dates=["fecha"])

# Agregar clima a frecuencia semanal
clima_sem = clima.set_index("fecha").resample("W").agg(
    t_min_media      = ("t_minima_c",    "mean"),
    t_prom_media     = ("t_promedio_c",  "mean"),
    hdd_semana       = ("hdd",           "sum"),
    heladas          = ("evento_helada", "sum"),
    precipitacion_mm = ("precipitacion_mm","sum"),
    enso_index       = ("enso_index",    "mean"),
).reset_index().rename(columns={"fecha":"fecha"})

# Usar solo los últimos 5 años para alinear con yfinance
fecha_corte = clima_sem["fecha"].max() - pd.DateOffset(years=5)
clima_5y    = clima_sem[clima_sem["fecha"] >= fecha_corte].copy()
rend_5y     = rend[rend["fecha"] >= fecha_corte].copy()

print(f"   ✅ Clima: {len(clima_5y)} semanas  |  Rendimientos: {len(rend_5y)} semanas")


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 3 — UNIFICACIÓN DE DATAFRAMES
# ═══════════════════════════════════════════════════════════════════
"""
ESTRATEGIA DE MERGE:
  - Todos los datos se alinean a frecuencia semanal
  - Se usa merge_asof (merge por proximidad temporal) para
    manejar diferencias de 1-2 días entre fuentes
  - Dirección: backward (usa el precio más reciente disponible)
  - Tolerancia: 3 días calendario
"""
print("\nUnificando DataFrames...")

# Estandarizar índice temporal
trm_m  = trm.reset_index().rename(columns={"Date":"fecha","Datetime":"fecha","Price Date":"fecha"})
# Asegurar nombre correcto de columna fecha
if "index" in trm_m.columns:
    trm_m = trm_m.rename(columns={"index":"fecha"})
trm_m["fecha"] = pd.to_datetime(trm_m["fecha"]).dt.tz_localize(None)

cafe_m = cafe.reset_index().rename(columns={"Date":"fecha","Datetime":"fecha"})
if "index" in cafe_m.columns:
    cafe_m = cafe_m.rename(columns={"index":"fecha"})
cafe_m["fecha"] = pd.to_datetime(cafe_m["fecha"]).dt.tz_localize(None)

# Ordenar para merge_asof
clima_5y = clima_5y.sort_values("fecha")
trm_m    = trm_m.sort_values("fecha")
cafe_m   = cafe_m.sort_values("fecha")
rend_5y  = rend_5y.sort_values("fecha")

# Merge 1: clima + TRM
df_uni = pd.merge_asof(
    clima_5y, trm_m[["fecha","trm_usdcop"]],
    on="fecha", direction="backward", tolerance=pd.Timedelta("3d")
)

# Merge 2: + café
df_uni = pd.merge_asof(
    df_uni, cafe_m[["fecha","cafe_usd_libra","cafe_usd_kg"]],
    on="fecha", direction="backward", tolerance=pd.Timedelta("3d")
)

# Merge 3: + rendimientos simulados
df_uni = pd.merge_asof(
    df_uni, rend_5y[["fecha","rendimiento_usd","perdida_pct"]],
    on="fecha", direction="nearest", tolerance=pd.Timedelta("4d")
)

# Calcular precio café en COP usando TRM descargada
df_uni["cafe_cop_kg"] = (df_uni["cafe_usd_kg"] * df_uni["trm_usdcop"]).round(0)

# Calcular rendimiento en COP
df_uni["rendimiento_cop"] = (df_uni["rendimiento_usd"] * df_uni["trm_usdcop"]).round(0)

# Columna de año-semana para referencia
df_uni["anio_semana"] = df_uni["fecha"].dt.strftime("%Y-W%U")

# Manejo de NaN: rellenar con forward fill para gaps pequeños
df_uni[["trm_usdcop","cafe_usd_libra","cafe_usd_kg","cafe_cop_kg"]] = (
    df_uni[["trm_usdcop","cafe_usd_libra","cafe_usd_kg","cafe_cop_kg"]]
    .fillna(method="ffill")
)

print(f"\n{'='*60}")
print(f"DATAFRAME UNIFICADO — Agro-Risk Pro")
print(f"{'='*60}")
print(f"  Dimensiones:   {df_uni.shape[0]} filas × {df_uni.shape[1]} columnas")
print(f"  Período:       {df_uni['fecha'].min().date()} → {df_uni['fecha'].max().date()}")
print(f"  Nulos TRM:     {df_uni['trm_usdcop'].isna().sum()}")
print(f"  Nulos Café:    {df_uni['cafe_usd_libra'].isna().sum()}")
print(f"\nColumnas disponibles:")
for col in df_uni.columns:
    dtype = str(df_uni[col].dtype)
    ejemplo = df_uni[col].dropna().iloc[-1] if not df_uni[col].dropna().empty else "N/A"
    if isinstance(ejemplo, float):
        ejemplo = f"{ejemplo:,.2f}"
    print(f"  {col:<28} {dtype:<12} ej: {ejemplo}")

# Estadísticas clave
print(f"\n{'='*60}")
print(f"ESTADÍSTICAS CLAVE (últimas 5 años)")
print(f"{'='*60}")
print(f"  TRM media:            {df_uni['trm_usdcop'].mean():,.0f} COP/USD")
print(f"  TRM rango:            {df_uni['trm_usdcop'].min():,.0f} – {df_uni['trm_usdcop'].max():,.0f}")
print(f"  Café medio NY:        {df_uni['cafe_usd_libra'].mean():.2f} USc/lb")
print(f"  Café medio COP/kg:    {df_uni['cafe_cop_kg'].mean():,.0f}")
print(f"  Correlación T–Café:   {df_uni[['t_min_media','cafe_usd_libra']].corr().iloc[0,1]:.3f}")
print(f"  Correlación TRM–Café: {df_uni[['trm_usdcop','cafe_usd_libra']].corr().iloc[0,1]:.3f}")

# Exportar
df_uni.to_csv("07_dataset_unificado.csv", index=False)
print(f"\n✅ 07_dataset_unificado.csv exportado ({df_uni.shape[0]} filas)")


# ═══════════════════════════════════════════════════════════════════
# BLOQUE 4 — VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14), facecolor=C_FONDO)
fig.suptitle("Agro-Risk Pro  ·  TRM, Café NY y Clima Unificados — Últimos 5 Años",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.32,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

# ── Panel A: TRM + Café en eje dual ─────────────────────────────────
ax1  = fig.add_subplot(gs[0, :])
ax1b = ax1.twinx()
l1,  = ax1.plot(df_uni["fecha"], df_uni["trm_usdcop"],
                color=C_VERDE, lw=1.4, label="TRM COP/USD")
ax1.fill_between(df_uni["fecha"], df_uni["trm_usdcop"],
                 alpha=0.12, color=C_VERDE)
l2,  = ax1b.plot(df_uni["fecha"], df_uni["cafe_usd_libra"],
                 color=C_NARANJA, lw=1.4, label="Café NY (USc/lb)")
ax1.set_ylabel("TRM (COP/USD)", color=C_VERDE, fontsize=10)
ax1b.set_ylabel("Café (USc/lb)", color=C_NARANJA, fontsize=10)
ax1b.tick_params(colors=C_NARANJA)
ax1.set_title("A  TRM USD/COP y Café Arábica NY (semanal)", fontsize=10, pad=8)
ax1.legend(handles=[l1,l2], fontsize=9, framealpha=0.2, loc="upper left")
ax1.grid(alpha=0.2)

# ── Panel B: Scatter TRM vs Café ────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
sc  = ax2.scatter(df_uni["trm_usdcop"], df_uni["cafe_usd_libra"],
                  c=df_uni["t_min_media"], cmap="coolwarm_r",
                  s=18, alpha=0.7)
# Línea de tendencia
mask = df_uni[["trm_usdcop","cafe_usd_libra"]].dropna()
if len(mask) > 5:
    from scipy import stats as sp
    sl, ic, rv, pv, _ = sp.linregress(mask["trm_usdcop"], mask["cafe_usd_libra"])
    xr = np.linspace(mask["trm_usdcop"].min(), mask["trm_usdcop"].max(), 100)
    ax2.plot(xr, ic + sl*xr, color=C_ROJO, lw=1.8,
             label=f"R²={rv**2:.3f}  β={sl:.4f}")
cb = fig.colorbar(sc, ax=ax2, fraction=0.04, pad=0.02)
cb.set_label("T mín °C", fontsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)
ax2.set_xlabel("TRM (COP/USD)"); ax2.set_ylabel("Café NY (USc/lb)")
ax2.set_title("B  Correlación TRM vs Café  (color = temperatura)", fontsize=10, pad=8)
ax2.legend(fontsize=8.5, framealpha=0.2); ax2.grid(alpha=0.3)

# ── Panel C: Precio café en COP/kg ──────────────────────────────────
ax3  = fig.add_subplot(gs[1, 1])
ax3b = ax3.twinx()
ax3.plot(df_uni["fecha"], df_uni["cafe_cop_kg"],
         color=C_MORADO, lw=1.4, label="Café (COP/kg)")
ax3.fill_between(df_uni["fecha"], df_uni["cafe_cop_kg"],
                 alpha=0.15, color=C_MORADO)
ax3b.plot(df_uni["fecha"], df_uni["t_min_media"],
          color=C_AZUL, lw=0.9, alpha=0.7, label="T mín (°C)")
ax3b.axhline(4.0, color=C_NARANJA, lw=1, ls="--", alpha=0.6)
ax3.set_ylabel("Café (COP/kg)", color=C_MORADO, fontsize=10)
ax3b.set_ylabel("T mínima (°C)", color=C_AZUL, fontsize=10)
ax3b.tick_params(colors=C_AZUL)
lns = ax3.get_lines() + ax3b.get_lines()
ax3.legend(lns, [l.get_label() for l in lns], fontsize=8.5, framealpha=0.2)
ax3.set_title("C  Precio Café en COP/kg vs Temperatura", fontsize=10, pad=8)
ax3.grid(alpha=0.2)

# ── Panel D: Rendimiento exportador vs TRM ──────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
sub = df_uni.dropna(subset=["rendimiento_usd","trm_usdcop"])
sc4 = ax4.scatter(sub["trm_usdcop"], sub["rendimiento_usd"],
                  c=sub["hdd_semana"], cmap="YlOrRd",
                  s=16, alpha=0.7)
cb4 = fig.colorbar(sc4, ax=ax4, fraction=0.04, pad=0.02)
cb4.set_label("HDD semana", fontsize=8)
plt.setp(cb4.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)
ax4.set_xlabel("TRM (COP/USD)"); ax4.set_ylabel("Rendimiento (USD/semana)")
ax4.set_title("D  Rendimiento Exportador vs TRM  (color = frío)", fontsize=10, pad=8)
ax4.grid(alpha=0.3)

# ── Panel E: Matriz de correlación ──────────────────────────────────
ax5   = fig.add_subplot(gs[2, 1])
cols  = ["t_min_media","hdd_semana","trm_usdcop",
         "cafe_usd_libra","cafe_cop_kg","rendimiento_usd"]
labs  = ["T mínima","HDD","TRM","Café NY","Café COP","Rendimiento"]
corr  = df_uni[cols].dropna().corr()
im5   = ax5.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
ax5.set_xticks(range(len(labs))); ax5.set_xticklabels(labs, rotation=35,
               ha="right", fontsize=8.5)
ax5.set_yticks(range(len(labs))); ax5.set_yticklabels(labs, fontsize=8.5)
for i in range(len(labs)):
    for j in range(len(labs)):
        val = corr.values[i, j]
        ax5.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=7.5,
                 color="white" if abs(val) > 0.5 else C_TEXTO)
cb5 = fig.colorbar(im5, ax=ax5, fraction=0.04, pad=0.02)
plt.setp(cb5.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)
ax5.set_title("E  Matriz de Correlación — Variables del Modelo", fontsize=10, pad=8)

plt.savefig("fig4_trm_cafe_clima.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()
print("✅ fig4_trm_cafe_clima.png guardada")
