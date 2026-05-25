"""
Agro-Risk Pro — Burn Analysis: Valuación de Opción Call Climática HDD
Prima justa por simulación histórica
Materia: Programación para Economía y Finanzas

DERIVADO VALUADO — Call Spread HDD:
  Payoff = Tick × min( max(0, HDD_realizado - Strike), Cap - Strike )
  Strike calibrado al percentil 40 del HDD histórico (dentro del dinero)
  Cap     calibrado al percentil 85 (protege el rango más probable de pérdida)

NOTA SOBRE EL UMBRAL:
  Usamos umbral 10°C (no 18°C) porque la Sabana de Bogotá rara vez
  supera 18°C — con ese umbral TODOS los días tienen HDD elevado y
  la señal de riesgo se pierde. El umbral 10°C es el estándar
  agronómico para cultivos de clima frío andino.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# ── Cargar datos ─────────────────────────────────────────────────────
df = pd.read_csv("outputs/temperatura_sabana_10años.csv",
                 parse_dates=["fecha"])

UMBRAL_HDD = 10.0   # °C agronómico para Cundinamarca

df["hdd"]  = np.maximum(0.0, UMBRAL_HDD - df["t_promedio_c"])
df["anio"] = df["fecha"].dt.year
df["mes"]  = df["fecha"].dt.month

# Temporada completa (año calendario — exposición total del exportador)
temporada = (
    df.groupby("anio")
    .agg(
        hdd_anual    = ("hdd",           "sum"),
        t_min_media  = ("t_minima_c",    "mean"),
        dias_helada  = ("evento_helada", "sum"),
        dias_total   = ("hdd",           "count"),
    )
    .round(2).reset_index()
)

# ── Calibrar strike y cap automáticamente ────────────────────────────
p40 = np.percentile(temporada["hdd_anual"], 40)
p85 = np.percentile(temporada["hdd_anual"], 85)

STRIKE_HDD  = round(p40 / 10) * 10      # redondeado a decena
CAP_HDD     = round(p85 / 10) * 10
TICK        = 250.0                      # USD por °C·día sobre el strike
NOCIONAL    = temporada["hdd_anual"].mean() * TICK
TASA_DESC   = 0.06
CARGA_SEG   = 0.20

print("=" * 65)
print("BURN ANALYSIS — OPCIÓN CALL CLIMÁTICA HDD")
print(f"Umbral agronómico: {UMBRAL_HDD}°C  (estándar cultivos andinos)")
print(f"Strike:  {STRIKE_HDD:.0f} °C·día  (P40 histórico)")
print(f"Cap:     {CAP_HDD:.0f} °C·día  (P85 histórico)")
print(f"Tick:    USD {TICK}/°C·día")
print(f"Nocional estimado: USD {NOCIONAL:,.0f}")
print("=" * 65)

print(f"\nHDD anual por año (umbral {UMBRAL_HDD}°C):")
for _, r in temporada.iterrows():
    barra = "█" * int(r["hdd_anual"] / 5)
    print(f"  {int(r['anio'])}: {r['hdd_anual']:>7.1f} °C·día  {barra}")
print(f"\n  P40 (Strike):  {STRIKE_HDD:.0f}")
print(f"  P85 (Cap):     {CAP_HDD:.0f}")
print(f"  Media:         {temporada['hdd_anual'].mean():.1f}")
print(f"  Min/Max:       {temporada['hdd_anual'].min():.1f} / {temporada['hdd_anual'].max():.1f}")


# ═══════════════════════════════════════════════════════════════════
# BURN ANALYSIS — PAYOFF HISTÓRICO
# ═══════════════════════════════════════════════════════════════════
n_años = len(temporada)
año_max = temporada["anio"].max()
factores = np.array([1/(1+TASA_DESC)**(año_max - a)
                     for a in temporada["anio"]])

temporada["hdd_sobre_strike"] = np.maximum(0.0, temporada["hdd_anual"] - STRIKE_HDD)
temporada["hdd_efectivo"]     = np.minimum(temporada["hdd_sobre_strike"],
                                            CAP_HDD - STRIKE_HDD)
temporada["payoff_usd"]       = (temporada["hdd_efectivo"] * TICK).round(2)
temporada["payoff_vp"]        = (temporada["payoff_usd"] * factores).round(2)
temporada["activado"]         = (temporada["payoff_usd"] > 0).astype(int)

prima_pura     = temporada["payoff_usd"].mean()
prima_pura_vp  = temporada["payoff_vp"].mean()
prima_comerc   = prima_pura_vp * (1 + CARGA_SEG)
tasa_prima_pct = prima_comerc / NOCIONAL * 100
activaciones   = temporada["activado"].sum()
payoff_cond    = temporada.loc[temporada["activado"]==1, "payoff_usd"].mean() \
                 if activaciones > 0 else 0
payoff_max_pos = (CAP_HDD - STRIKE_HDD) * TICK

print(f"\n{'='*65}")
print("RESULTADOS DEL BURN ANALYSIS")
print(f"{'='*65}")
print(f"  Años analizados:              {n_años}")
print(f"  Años con activación:          {activaciones}  ({activaciones/n_años*100:.0f}%)")
print(f"  Payoff promedio bruto:        USD {prima_pura:>10,.2f}")
print(f"  Prima pura (VP descontado):   USD {prima_pura_vp:>10,.2f}")
print(f"  Prima comercial (+20%):       USD {prima_comerc:>10,.2f}")
print(f"  Prima como % del nocional:    {tasa_prima_pct:.2f}%")
print(f"  Payoff condicional medio:     USD {payoff_cond:>10,.2f}")
print(f"  Payoff máximo posible:        USD {payoff_max_pos:>10,.2f}")
if payoff_cond > 0 and prima_comerc > 0:
    print(f"  Ratio beneficio/prima:        {payoff_cond/prima_comerc:.1f}×")

print(f"\n{'─'*65}")
print("DETALLE POR AÑO:")
for _, r in temporada.iterrows():
    tag  = " ◄ ACTIVADO" if r["activado"] else ""
    barra= "█" * int(r["hdd_efectivo"] / 5) if r["activado"] else ""
    print(f"  {int(r['anio'])}: HDD={r['hdd_anual']:>6.0f}  "
          f"Sobre strike={r['hdd_sobre_strike']:>5.0f}  "
          f"Payoff=USD {r['payoff_usd']:>9,.0f}  {barra}{tag}")


# ═══════════════════════════════════════════════════════════════════
# SENSIBILIDAD STRIKE vs PRIMA
# ═══════════════════════════════════════════════════════════════════
hdd_min = temporada["hdd_anual"].min()
hdd_max = temporada["hdd_anual"].max()
strikes_rango = np.linspace(hdd_min * 0.85, hdd_max * 0.95, 20)
rows_sens = []
for s in strikes_rango:
    cap_s = s + (CAP_HDD - STRIKE_HDD)
    pf = np.minimum(np.maximum(0, temporada["hdd_anual"] - s),
                    cap_s - s) * TICK
    rows_sens.append({
        "strike": round(s, 1),
        "prima_pura": round(pf.mean(), 2),
        "prob_act_pct": round((pf > 0).mean() * 100, 1),
        "payoff_cond": round(pf[pf > 0].mean(), 2) if (pf > 0).any() else 0.0
    })
df_sens = pd.DataFrame(rows_sens)

print(f"\n{'='*65}")
print("SENSIBILIDAD: STRIKE vs PRIMA PURA vs P(ACTIVACIÓN)")
print(f"{'='*65}")
print(df_sens.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════
np.random.seed(42)
N_BOOT = 10_000
payoffs_arr  = temporada["payoff_usd"].values
primas_boot  = np.array([
    payoffs_arr[np.random.choice(n_años, n_años, replace=True)].mean()
    for _ in range(N_BOOT)
])
ic95_lo = np.percentile(primas_boot, 2.5)
ic95_hi = np.percentile(primas_boot, 97.5)
ic80_lo = np.percentile(primas_boot, 10.0)
ic80_hi = np.percentile(primas_boot, 90.0)

print(f"\n{'='*65}")
print("INCERTIDUMBRE (Bootstrap 10,000 muestras)")
print(f"{'='*65}")
print(f"  Prima boot media:   USD {primas_boot.mean():,.2f}")
print(f"  Desv. estándar:     USD {primas_boot.std():,.2f}")
print(f"  IC 80%:  USD {ic80_lo:,.0f} – USD {ic80_hi:,.0f}")
print(f"  IC 95%:  USD {ic95_lo:,.0f} – USD {ic95_hi:,.0f}")
coef_var = primas_boot.std()/primas_boot.mean()*100 if primas_boot.mean() > 0 else 0
print(f"  Coef. variación:    {coef_var:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 15), facecolor=C_FONDO)
fig.suptitle("Agro-Risk Pro  ·  Burn Analysis — Opción Call HDD Climática (Umbral 10°C)",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, hspace=0.52, wspace=0.32,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

# ── Panel A: HDD anual vs Strike / Cap ──────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
colores_b = [C_ROJO if a else C_AZUL for a in temporada["activado"]]
bars = ax1.bar(temporada["anio"], temporada["hdd_anual"],
               color=colores_b, alpha=0.80, width=0.65, zorder=2)

ax1.axhline(STRIKE_HDD, color=C_NARANJA, lw=2, ls="--", zorder=3,
            label=f"Strike {STRIKE_HDD:.0f} °C·día  (P40)")
ax1.axhline(CAP_HDD, color=C_ROJO, lw=1.5, ls=":", zorder=3,
            label=f"Cap {CAP_HDD:.0f} °C·día  (P85)")
ax1.axhspan(STRIKE_HDD, CAP_HDD, alpha=0.07, color=C_ROJO, label="Zona de pago")
ax1.axhline(temporada["hdd_anual"].mean(), color=C_VERDE, lw=1.2,
            ls="-", alpha=0.6, label=f"Media {temporada['hdd_anual'].mean():.0f}")

for bar, row in zip(bars, temporada.itertuples()):
    yoff = bar.get_height() + 2
    ax1.text(bar.get_x()+bar.get_width()/2, yoff,
             f"{row.hdd_anual:.0f}", ha="center", fontsize=8.5, color=C_TEXTO)
    if row.activado:
        ax1.text(bar.get_x()+bar.get_width()/2,
                 (STRIKE_HDD + bar.get_height())/2,
                 f"USD\n{row.payoff_usd:,.0f}",
                 ha="center", va="center", fontsize=8,
                 fontweight="bold", color="white")

ax1.set_xticks(temporada["anio"])
ax1.set_ylabel("HDD anual acumulado (°C·día)")
ax1.set_title("A  HDD Anual vs Strike/Cap — Rojo = derivado activado", fontsize=10, pad=8)
ax1.legend(fontsize=9, framealpha=0.2, loc="upper right")
ax1.grid(axis="y", alpha=0.25)

# ── Panel B: Payoffs históricos + prima ─────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
c_p = [C_ROJO if p > 0 else C_AZUL+"55" for p in temporada["payoff_usd"]]
ax2.bar(temporada["anio"], temporada["payoff_usd"],
        color=c_p, alpha=0.85, width=0.65)
ax2.axhline(prima_pura,    color=C_VERDE,   lw=2,   ls="-",
            label=f"Prima pura  USD {prima_pura:,.0f}")
ax2.axhline(prima_pura_vp, color=C_MORADO,  lw=1.5, ls=":",
            label=f"Prima VP  USD {prima_pura_vp:,.0f}")
ax2.axhline(prima_comerc,  color=C_NARANJA, lw=1.8, ls="--",
            label=f"Prima comercial  USD {prima_comerc:,.0f}")
if ic95_hi > 0:
    ax2.fill_between(temporada["anio"],
                     [ic95_lo]*n_años, [ic95_hi]*n_años,
                     alpha=0.08, color=C_VERDE, label="IC 95% boot")
ax2.set_xticks(temporada["anio"])
ax2.set_ylabel("Payoff (USD)")
ax2.set_title("B  Payoffs Históricos y Estimación de Prima", fontsize=10, pad=8)
ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(axis="y", alpha=0.3)

# ── Panel C: Sensibilidad strike–prima ──────────────────────────────
ax3  = fig.add_subplot(gs[1, 1])
ax3b = ax3.twinx()
ax3.plot(df_sens["strike"], df_sens["prima_pura"],
         color=C_AZUL, lw=2, marker="o", ms=4, label="Prima pura (USD)")
ax3.fill_between(df_sens["strike"], df_sens["prima_pura"],
                 alpha=0.12, color=C_AZUL)
ax3.axvline(STRIKE_HDD, color=C_NARANJA, lw=1.5, ls="--",
            label=f"Strike actual {STRIKE_HDD:.0f}")
ax3b.plot(df_sens["strike"], df_sens["prob_act_pct"],
          color=C_ROJO, lw=1.5, ls="--", label="P(activación) %")
ax3.set_xlabel("Strike HDD (°C·día)")
ax3.set_ylabel("Prima pura (USD)", color=C_AZUL)
ax3b.set_ylabel("P(activación) %", color=C_ROJO)
ax3b.tick_params(colors=C_ROJO)
ax3.set_title("C  Sensibilidad Strike vs Prima y Probabilidad", fontsize=10, pad=8)
lns = ax3.get_lines() + ax3b.get_lines()
ax3.legend(lns, [l.get_label() for l in lns], fontsize=8, framealpha=0.2)
ax3.grid(alpha=0.3)

# ── Panel D: Bootstrap ──────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(primas_boot, bins=60, color=C_AZUL, alpha=0.65, edgecolor="none")
ax4.axvline(primas_boot.mean(), color=C_VERDE, lw=2,
            label=f"Media boot  USD {primas_boot.mean():,.0f}")
ax4.axvline(prima_comerc, color=C_NARANJA, lw=1.8, ls="--",
            label=f"Prima comercial  USD {prima_comerc:,.0f}")
if ic95_hi > 0:
    ax4.axvspan(ic95_lo, ic95_hi, alpha=0.10, color=C_VERDE,
                label=f"IC 95%  [{ic95_lo:,.0f}–{ic95_hi:,.0f}]")
    ax4.axvspan(ic80_lo, ic80_hi, alpha=0.18, color=C_VERDE,
                label=f"IC 80%  [{ic80_lo:,.0f}–{ic80_hi:,.0f}]")
ax4.set_xlabel("Prima pura estimada (USD)")
ax4.set_ylabel("Frecuencia")
ax4.set_title("D  Distribución Bootstrap Prima (N=10,000)", fontsize=10, pad=8)
ax4.legend(fontsize=8, framealpha=0.2); ax4.grid(axis="y", alpha=0.3)

# ── Panel E: Función de payoff con realizaciones históricas ─────────
ax5 = fig.add_subplot(gs[2, 1])
hdd_eje = np.linspace(hdd_min * 0.8, hdd_max * 1.1, 500)
pf_eje  = np.minimum(np.maximum(0, hdd_eje - STRIKE_HDD),
                     CAP_HDD - STRIKE_HDD) * TICK
ax5.plot(hdd_eje, pf_eje, color=C_VERDE, lw=2.5, label="Función de payoff")
ax5.fill_between(hdd_eje, pf_eje, alpha=0.15, color=C_VERDE)
ax5.axvline(STRIKE_HDD, color=C_NARANJA, lw=1.5, ls="--",
            label=f"Strike {STRIKE_HDD:.0f}")
ax5.axvline(CAP_HDD,    color=C_ROJO,    lw=1.2, ls=":",
            label=f"Cap {CAP_HDD:.0f}")
ax5.axhline(prima_comerc, color=C_MORADO, lw=1.2, ls="--",
            label=f"Prima USD {prima_comerc:,.0f}")
for _, row in temporada.iterrows():
    h = row["hdd_anual"]
    p = min(max(0, h - STRIKE_HDD), CAP_HDD - STRIKE_HDD) * TICK
    c = C_ROJO if row["activado"] else C_AZUL
    ax5.scatter(h, p, color=c, s=55, zorder=5)
    ax5.text(h + (hdd_max-hdd_min)*0.01, p + payoff_max_pos*0.02,
             str(int(row["anio"])), fontsize=7.5, color=c)
ax5.set_xlabel("HDD anual realizado (°C·día)")
ax5.set_ylabel("Payoff (USD)")
ax5.set_title("E  Función de Payoff + Realizaciones Históricas", fontsize=10, pad=8)
ax5.legend(fontsize=8, framealpha=0.2); ax5.grid(alpha=0.3)

plt.savefig("outputs/fig6_burn_analysis.png", dpi=150,
            bbox_inches="tight", facecolor=C_FONDO)
plt.close()

temporada.to_csv("outputs/09_burn_analysis.csv", index=False)

print(f"\n{'='*65}")
print("RESUMEN EJECUTIVO — PARA EL MÓDULO IA DE LA APP")
print(f"{'='*65}")
print(f"  Umbral agronómico:       {UMBRAL_HDD}°C")
print(f"  Strike calibrado:        {STRIKE_HDD:.0f} °C·día  (P40 histórico)")
print(f"  Cap calibrado:           {CAP_HDD:.0f} °C·día  (P85 histórico)")
print(f"  P(activación histórica): {activaciones/n_años*100:.0f}%  ({activaciones}/{n_años} años)")
print(f"  Prima pura (burn):       USD {prima_pura:,.2f}/año")
print(f"  Prima comercial:         USD {prima_comerc:,.2f}/año  ({tasa_prima_pct:.2f}% del nocional)")
print(f"  Payoff condicional:      USD {payoff_cond:,.0f}  (si el evento ocurre)")
if prima_comerc > 0:
    print(f"  Ratio retorno/prima:     {payoff_cond/prima_comerc:.1f}×")
print(f"\n✅ fig6_burn_analysis.png guardada")
print(f"✅ 09_burn_analysis.csv exportado")
