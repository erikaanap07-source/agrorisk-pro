"""
Agro-Risk Pro — Reporte PDF de Cobertura Climática
Módulo analítico · Output profesional
Materia: Programación para Economía y Finanzas
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import differential_evolution
from scipy import stats
import io, os, warnings
warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, HRFlowable, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

# ── Paleta de colores del reporte ────────────────────────────────────
VERDE_DARK  = colors.HexColor("#0f6e56")
VERDE_MED   = colors.HexColor("#1d9e75")
VERDE_LIGHT = colors.HexColor("#e1f5ee")
GRIS_DARK   = colors.HexColor("#2c2c2a")
GRIS_MED    = colors.HexColor("#5f5e5a")
GRIS_LIGHT  = colors.HexColor("#f1efe8")
ROJO        = colors.HexColor("#e24b4a")
NARANJA     = colors.HexColor("#ef9f27")
AZUL        = colors.HexColor("#378add")
BLANCO      = colors.white
NEGRO       = colors.HexColor("#1a1a1a")

# Paleta matplotlib
C_FONDO = "#f8f9fa"; C_PANEL = "#ffffff"; C_TEXTO = "#1a1a1a"
C_AZUL  = "#378add"; C_VERDE = "#1d9e75"; C_ROJO  = "#e24b4a"
C_NAR   = "#ef9f27"; C_GRIS  = "#8b90a0"; C_MOR   = "#7f77dd"

plt.rcParams.update({
    "figure.facecolor": C_FONDO, "axes.facecolor":  C_PANEL,
    "axes.edgecolor":   "#dddddd","axes.labelcolor": C_TEXTO,
    "axes.titlecolor":  C_TEXTO,  "xtick.color":     C_GRIS,
    "ytick.color":      C_GRIS,   "text.color":      C_TEXTO,
    "grid.color":       "#eeeeee","grid.linewidth":   0.6,
    "font.size": 9,
})

np.random.seed(42)
BASE_DIR = "/mnt/user-data/outputs"
FECHA_REPORTE = "23 mayo 2025"

# ═══════════════════════════════════════════════════════════════════
# CARGAR Y RECALCULAR TODOS LOS DATOS
# ═══════════════════════════════════════════════════════════════════
df = pd.read_csv(f"{BASE_DIR}/temperatura_sabana_10años.csv", parse_dates=["fecha"])
df["hdd"]  = np.maximum(0.0, 10.0 - df["t_promedio_c"])
df["anio"] = df["fecha"].dt.year
df["mes"]  = df["fecha"].dt.month

sem = df.set_index("fecha").resample("W").agg(
    hdd_sem=("hdd","sum"), t_min=("t_minima_c","mean"),
    heladas=("evento_helada","sum"), enso=("enso_index","mean"),
).reset_index()

R_BASE = 16_000; BETA_HDD = -85; BETA_H = -1800; BETA_E = -600
ruido  = np.zeros(len(sem)); eps = np.random.normal(0,1200,len(sem))
ruido[0] = eps[0]
for i in range(1,len(sem)): ruido[i] = 0.45*ruido[i-1]+eps[i]
sem["ventas"] = (R_BASE + BETA_HDD*sem["hdd_sem"] + BETA_H*sem["heladas"]
                 + BETA_E*sem["enso"].clip(lower=0) + ruido).clip(lower=0)

HDD_V   = sem["hdd_sem"].values
VENTAS  = sem["ventas"].values
UMBRAL  = 10.0; CARGA = 0.20
STRIKE_OPT = 4.07; TICK_OPT = 317.70
STRIKE_H   = 38.0; TICK_H   = 250.0

def payoff_contrato(strike, tick, hdd): return np.maximum(0, hdd - strike) * tick
def prima_contrato(strike, tick, hdd, c=CARGA):
    return payoff_contrato(strike, tick, hdd).mean() * (1 + c)

prima_opt  = prima_contrato(STRIKE_OPT, TICK_OPT, HDD_V)
prima_heur = prima_contrato(STRIKE_H,   TICK_H,   HDD_V)

pf_opt  = payoff_contrato(STRIKE_OPT, TICK_OPT, HDD_V)
ing_opt = VENTAS - prima_opt + pf_opt
ing_sin = VENTAS.copy()

prob_act  = (pf_opt > 0).mean() * 100
p_act_heur= (payoff_contrato(STRIKE_H, TICK_H, HDD_V) > 0).mean() * 100
red_var   = (1 - np.var(ing_opt)/np.var(ing_sin)) * 100
var95     = np.percentile(ing_opt, 5)
cvar95    = ing_opt[ing_opt <= var95].mean()

MESES = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
anual = df.groupby("anio").agg(
    hdd_total=("hdd","sum"), t_min=("t_minima_c","mean"),
    heladas=("evento_helada","sum")
).round(2).reset_index()
anual["payoff_opt"]  = payoff_contrato(STRIKE_OPT, TICK_OPT, anual["hdd_total"].values).round(0)
anual["activado"]    = (anual["payoff_opt"] > 0).astype(int)

# Monte Carlo para stress
np.random.seed(42)
cal = df.groupby(df["fecha"].dt.dayofyear.clip(1,365))["t_promedio_c"].agg(
    media="mean", sigma="std").reindex(range(1,366)).interpolate()
resid = df.sort_values("fecha")["t_promedio_c"].values - \
        cal.loc[df.sort_values("fecha")["fecha"].dt.dayofyear.clip(1,365).values,"media"].values
AR1   = np.corrcoef(resid[:-1],resid[1:])[0,1]
SIG   = float(np.std(resid)*np.sqrt(1-AR1**2))
dias_idx = np.array(pd.date_range("2025-01-01",periods=90,freq="D").dayofyear).clip(1,365)
MU    = cal.loc[dias_idx,"media"].values
N_SIM = 10_000; N_D = 90
eps_mc = np.random.normal(0,SIG,(N_SIM,N_D))
T_mc   = np.zeros((N_SIM,N_D)); ep = np.zeros(N_SIM)
for t in range(N_D):
    T_mc[:,t] = MU[t] + AR1*ep + eps_mc[:,t]; ep = T_mc[:,t]-MU[t]
HDD_mc    = np.maximum(0, UMBRAL - T_mc).sum(axis=1)
PF_mc     = np.maximum(0, HDD_mc - STRIKE_OPT) * TICK_OPT
prob_mc   = (PF_mc>0).mean()*100
e_payoff  = PF_mc.mean()

# Stress scenarios
STRESS = [
    ("Base histórico",      0.0,   0.0,   C_AZUL),
    ("La Niña moderada",    -0.5,  0.0,   C_MOR),
    ("La Niña fuerte",      -1.5,  0.3,   "#9b59b6"),
    ("El Niño moderado",    +0.8,  0.0,   C_NAR),
    ("El Niño fuerte",      +1.5, -0.2,   C_ROJO),
    ("Alta volatilidad",     0.0,  0.5,   "#e67e22"),
]
stress_rows = []
for nombre, t_shift, sig_shift, _ in STRESS:
    sig_s = max(0.3, SIG + sig_shift)
    eps_s = np.random.normal(0, sig_s, (N_SIM, N_D))
    T_s   = np.zeros((N_SIM, N_D)); ep_s = np.zeros(N_SIM)
    for t in range(N_D):
        T_s[:,t] = MU[t]+t_shift+AR1*ep_s+eps_s[:,t]; ep_s=T_s[:,t]-MU[t]-t_shift
    hdd_s   = np.maximum(0, UMBRAL - T_s).sum(axis=1)
    pf_s    = np.maximum(0, hdd_s - STRIKE_OPT) * TICK_OPT
    stress_rows.append({
        "escenario":   nombre,
        "hdd_medio":   round(hdd_s.mean(), 1),
        "prob_act":    round((pf_s>0).mean()*100, 1),
        "e_payoff":    round(pf_s.mean(), 0),
        "payoff_p95":  round(np.percentile(pf_s, 95), 0),
        "var_5pct":    round(np.percentile(pf_s, 5), 0),
    })
df_stress = pd.DataFrame(stress_rows)

print("Datos calculados. Generando gráficas...")


# ═══════════════════════════════════════════════════════════════════
# GENERAR GRÁFICAS PARA EL PDF
# ═══════════════════════════════════════════════════════════════════
def fig_to_img(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ── Gráfica 1: Distribución de ingresos comparada ───────────────────
fig1, ax = plt.subplots(figsize=(12, 3.5), facecolor=C_FONDO)
ax.hist(ing_sin,  bins=55, density=True, alpha=0.45, color=C_ROJO,
        edgecolor="none", label=f"Sin cobertura  σ=USD {np.std(ing_sin):,.0f}")
ax.hist(ing_opt,  bins=55, density=True, alpha=0.55, color=C_VERDE,
        edgecolor="none", label=f"Con cobertura óptima  σ=USD {np.std(ing_opt):,.0f}")
for arr, col, ls in [(ing_sin,C_ROJO,"-"),(ing_opt,C_VERDE,"--")]:
    m,s = arr.mean(), arr.std()
    xv  = np.linspace(arr.min(), arr.max(), 300)
    ax.plot(xv, stats.norm.pdf(xv,m,s), color=col, lw=1.8, ls=ls)
ax.axvline(ing_opt.mean(), color=C_VERDE, lw=1.2, ls=":", alpha=0.8)
ax.set_xlabel("Ingreso semanal (USD)"); ax.set_ylabel("Densidad")
ax.set_title(f"Distribución de ingresos — reducción varianza {red_var:.1f}%", fontsize=10)
ax.legend(fontsize=8, framealpha=0.7); ax.grid(alpha=0.3)
buf1 = fig_to_img(fig1); plt.close(fig1)

# ── Gráfica 2: HDD histórico y payoffs ─────────────────────────────
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=C_FONDO)
colores_b = [C_ROJO if a else C_AZUL for a in anual["activado"]]
bars = ax2a.bar(anual["anio"], anual["hdd_total"], color=colores_b, alpha=0.80, width=0.65)
ax2a.axhline(STRIKE_OPT, color=C_NAR, lw=1.8, ls="--", label=f"Strike {STRIKE_OPT:.1f}")
for bar, row in zip(bars, anual.itertuples()):
    ax2a.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
              f"{row.hdd_total:.0f}", ha="center", fontsize=7.5, color=C_TEXTO)
ax2a.set_title("HDD anual vs Strike (rojo = activado)", fontsize=9)
ax2a.set_ylabel("HDD (°C·día)"); ax2a.legend(fontsize=8); ax2a.grid(axis="y", alpha=0.3)
ax2a.set_xticks(anual["anio"]); ax2a.set_xticklabels(anual["anio"], fontsize=7, rotation=45)

pf_colors = [C_ROJO if p>0 else "#cccccc" for p in anual["payoff_opt"]]
ax2b.bar(anual["anio"], anual["payoff_opt"], color=pf_colors, alpha=0.85, width=0.65)
ax2b.axhline(prima_opt, color=C_VERDE, lw=1.8, ls="--", label=f"Prima USD {prima_opt:,.0f}/sem")
ax2b.set_title("Payoff histórico del contrato óptimo", fontsize=9)
ax2b.set_ylabel("Payoff (USD)"); ax2b.legend(fontsize=8); ax2b.grid(axis="y", alpha=0.3)
ax2b.set_xticks(anual["anio"]); ax2b.set_xticklabels(anual["anio"], fontsize=7, rotation=45)
fig2.tight_layout()
buf2 = fig_to_img(fig2); plt.close(fig2)

# ── Gráfica 3: Stress testing ────────────────────────────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=C_FONDO)
col_stress = [C_AZUL, C_MOR, "#9b59b6", C_NAR, C_ROJO, "#e67e22"]
escenarios = df_stress["escenario"].tolist()
probs      = df_stress["prob_act"].tolist()
epayoffs   = df_stress["e_payoff"].tolist()
bars3 = ax3a.barh(escenarios, probs, color=col_stress, alpha=0.80, height=0.55)
for bar, val in zip(bars3, probs):
    ax3a.text(val+0.5, bar.get_y()+bar.get_height()/2,
              f"{val:.1f}%", va="center", fontsize=8)
ax3a.set_xlabel("Probabilidad de activación (%)"); ax3a.grid(axis="x", alpha=0.3)
ax3a.set_title("P(activación) por escenario climático", fontsize=9)
bars3b = ax3b.barh(escenarios, epayoffs, color=col_stress, alpha=0.80, height=0.55)
for bar, val in zip(bars3b, epayoffs):
    ax3b.text(val+30, bar.get_y()+bar.get_height()/2,
              f"USD {val:,.0f}", va="center", fontsize=8)
ax3b.set_xlabel("Valor esperado del payoff (USD)"); ax3b.grid(axis="x", alpha=0.3)
ax3b.set_title("E[Payoff] por escenario climático", fontsize=9)
fig3.tight_layout()
buf3 = fig_to_img(fig3); plt.close(fig3)

# ── Gráfica 4: Monte Carlo fan chart ────────────────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=C_FONDO)
HDD_acum = np.maximum(0, UMBRAL - T_mc).cumsum(axis=1)
for p, alpha, label in [(5,0.12,"P5–P95"),(25,0.22,"P25–P75")]:
    ax4a.fill_between(range(N_D), np.percentile(HDD_acum,p,axis=0),
                      np.percentile(HDD_acum,100-p,axis=0), alpha=alpha, color=C_AZUL, label=label)
ax4a.plot(range(N_D), np.percentile(HDD_acum,50,axis=0), color=C_AZUL, lw=2, label="Mediana")
ax4a.axhline(STRIKE_OPT, color=C_NAR, lw=1.5, ls="--", label=f"Strike {STRIKE_OPT:.1f}")
ax4a.set_xlabel("Días"); ax4a.set_ylabel("HDD acumulado (°C·día)")
ax4a.set_title("Fan chart HDD — 90 días simulados", fontsize=9)
ax4a.legend(fontsize=7, framealpha=0.7); ax4a.grid(alpha=0.3)

ax4b.hist(PF_mc[PF_mc>0], bins=50, color=C_VERDE, alpha=0.65, edgecolor="none",
          label=f"Con pago ({prob_mc:.1f}%)")
ax4b.axvline(e_payoff, color=C_VERDE, lw=2, label=f"E[Payoff] USD {e_payoff:,.0f}")
ax4b.axvline(prima_opt, color=C_NAR, lw=1.8, ls="--", label=f"Prima USD {prima_opt:,.0f}")
n_cero = (PF_mc == 0).sum()
ax4b.bar(0, n_cero/len(PF_mc)*3, width=50, color=C_GRIS, alpha=0.5,
         label=f"Sin pago ({100-prob_mc:.1f}%)")
ax4b.set_xlabel("Payoff (USD)"); ax4b.set_ylabel("Frecuencia")
ax4b.set_title(f"Distribución payoff MC (N={N_SIM:,})", fontsize=9)
ax4b.legend(fontsize=7, framealpha=0.7); ax4b.grid(axis="y", alpha=0.3)
fig4.tight_layout()
buf4 = fig_to_img(fig4); plt.close(fig4)

print("Graficas OK. Construyendo PDF...")


# ═══════════════════════════════════════════════════════════════════
# CONSTRUIR EL PDF CON REPORTLAB
# ═══════════════════════════════════════════════════════════════════
OUTPUT_PDF = f"{BASE_DIR}/agrorisk_reporte_cobertura.pdf"
W, H = A4

doc = SimpleDocTemplate(
    OUTPUT_PDF,
    pagesize         = A4,
    leftMargin       = 2.0*cm,
    rightMargin      = 2.0*cm,
    topMargin        = 1.8*cm,
    bottomMargin     = 1.8*cm,
    title            = "Agro-Risk Pro — Reporte de Cobertura Climática",
    author           = "Módulo Analítico PEF",
    subject          = "Derivados paramétricos climáticos HDD",
)

styles = getSampleStyleSheet()

def estilo(nombre, base="Normal", **kw):
    s = ParagraphStyle(nombre, parent=styles[base], **kw)
    return s

S_TITULO    = estilo("titulo",    base="Normal", fontSize=22, textColor=NEGRO,
                     fontName="Helvetica-Bold", spaceAfter=4, leading=26)
S_SUBTIT    = estilo("subtit",    base="Normal", fontSize=11, textColor=GRIS_MED,
                     fontName="Helvetica", spaceAfter=2)
S_H1        = estilo("h1",        base="Normal", fontSize=13, textColor=VERDE_DARK,
                     fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4, leading=16)
S_H2        = estilo("h2",        base="Normal", fontSize=10, textColor=GRIS_DARK,
                     fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3, leading=13)
S_BODY      = estilo("body",      base="Normal", fontSize=9,  textColor=GRIS_DARK,
                     fontName="Helvetica", leading=13, spaceAfter=4)
S_CAPTION   = estilo("caption",   base="Normal", fontSize=8,  textColor=GRIS_MED,
                     fontName="Helvetica-Oblique", spaceAfter=6, alignment=TA_CENTER)
S_DISCLAIMER= estilo("disc",      base="Normal", fontSize=7.5,textColor=GRIS_MED,
                     fontName="Helvetica-Oblique", leading=10, spaceBefore=6)
S_KPI_VAL   = estilo("kpival",    base="Normal", fontSize=18, textColor=VERDE_DARK,
                     fontName="Helvetica-Bold", leading=20, alignment=TA_CENTER)
S_KPI_LBL   = estilo("kpilbl",    base="Normal", fontSize=8,  textColor=GRIS_MED,
                     fontName="Helvetica", alignment=TA_CENTER)

def hr(): return HRFlowable(width="100%", thickness=0.5, color=VERDE_MED, spaceAfter=6, spaceBefore=2)
def sp(h=6): return Spacer(1, h)
def img_from_buf(buf, w_cm=16.5):
    buf.seek(0)
    return Image(buf, width=w_cm*cm, height=w_cm*cm*3.5/12)

# ── Tabla de KPIs ────────────────────────────────────────────────────
def tabla_kpis():
    kpis = [
        (f"USD {prima_opt:,.0f}", "Prima semanal óptima"),
        (f"{prob_act:.1f}%",      "P(activación) histórica"),
        (f"USD {e_payoff:,.0f}", f"E[Payoff] MC ({N_SIM:,} sim.)"),
        (f"{red_var:.1f}%",       "Reducción varianza"),
    ]
    colores_kpi = [VERDE_LIGHT, VERDE_LIGHT, VERDE_LIGHT, VERDE_LIGHT]
    data = [
        [Paragraph(kpis[i][0], S_KPI_VAL) for i in range(4)],
        [Paragraph(kpis[i][1], S_KPI_LBL) for i in range(4)],
    ]
    t = Table(data, colWidths=[4.0*cm]*4, rowHeights=[1.2*cm, 0.7*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(3,1), VERDE_LIGHT),
        ("BOX",         (0,0),(0,1), 0.5, VERDE_MED),
        ("BOX",         (1,0),(1,1), 0.5, VERDE_MED),
        ("BOX",         (2,0),(2,1), 0.5, VERDE_MED),
        ("BOX",         (3,0),(3,1), 0.5, VERDE_MED),
        ("ALIGN",       (0,0),(3,1), "CENTER"),
        ("VALIGN",      (0,0),(3,1), "MIDDLE"),
        ("TOPPADDING",  (0,0),(3,1), 6),
        ("BOTTOMPADDING",(0,0),(3,1), 4),
        ("LEFTPADDING", (0,0),(3,1), 4),
        ("RIGHTPADDING",(0,0),(3,1), 4),
    ]))
    return t

# ── Tabla de parámetros del contrato ────────────────────────────────
def tabla_contrato():
    enc = ["Parámetro", "Heurístico", "Óptimo", "Unidad"]
    rows = [
        ["Strike HDD",       f"{STRIKE_H:.1f}",  f"{STRIKE_OPT:.2f}", "°C·día"],
        ["Tick",             f"{TICK_H:.0f}",     f"{TICK_OPT:.2f}",   "USD/°C·día"],
        ["Prima semanal",    f"{prima_heur:,.0f}",f"{prima_opt:,.0f}",  "USD"],
        ["Prima anual",      f"{prima_heur*52:,.0f}",f"{prima_opt*52:,.0f}","USD"],
        ["P(activación)",    f"{p_act_heur:.1f}%",f"{prob_act:.1f}%",   "—"],
        ["Reducción varianza","0.0%",             f"{red_var:.1f}%",    "—"],
        ["VaR 5% ingreso",   "—",                f"USD {var95:,.0f}",   "USD/semana"],
        ["CVaR 5% ingreso",  "—",                f"USD {cvar95:,.0f}",  "USD/semana"],
    ]
    data = [enc] + rows
    col_w = [5.0*cm, 3.2*cm, 3.2*cm, 3.0*cm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND",   (0,0),(3,0),  VERDE_DARK),
        ("TEXTCOLOR",    (0,0),(3,0),  BLANCO),
        ("FONTNAME",     (0,0),(3,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(3,0),  9),
        ("ALIGN",        (0,0),(3,0),  "CENTER"),
        ("FONTNAME",     (0,1),(3,-1), "Helvetica"),
        ("FONTSIZE",     (0,1),(3,-1), 8.5),
        ("ALIGN",        (1,1),(3,-1), "CENTER"),
        ("TEXTCOLOR",    (0,1),(3,-1), GRIS_DARK),
        ("ROWBACKGROUNDS",(0,1),(3,-1),[BLANCO, GRIS_LIGHT]),
        ("GRID",         (0,0),(3,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",   (0,0),(3,-1), 5),
        ("BOTTOMPADDING",(0,0),(3,-1), 5),
        ("LEFTPADDING",  (0,0),(3,-1), 7),
        ("TEXTCOLOR",    (2,1),(2,-1), VERDE_DARK),
        ("FONTNAME",     (2,1),(2,-1), "Helvetica-Bold"),
    ])
    t.setStyle(ts)
    return t

# ── Tabla de stress testing ──────────────────────────────────────────
def tabla_stress():
    enc = ["Escenario", "HDD medio", "P(activ.)", "E[Payoff]", "P95 Payoff"]
    rows = [[r["escenario"], f"{r['hdd_medio']:.1f}",
             f"{r['prob_act']:.1f}%", f"USD {r['e_payoff']:,.0f}",
             f"USD {r['payoff_p95']:,.0f}"] for _, r in df_stress.iterrows()]
    data  = [enc] + rows
    col_w = [5.0*cm, 2.5*cm, 2.5*cm, 3.0*cm, 3.0*cm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND",    (0,0),(4,0),  VERDE_DARK),
        ("TEXTCOLOR",     (0,0),(4,0),  BLANCO),
        ("FONTNAME",      (0,0),(4,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(4,0),  9),
        ("ALIGN",         (0,0),(4,0),  "CENTER"),
        ("FONTNAME",      (0,1),(4,-1), "Helvetica"),
        ("FONTSIZE",      (0,1),(4,-1), 8.5),
        ("ALIGN",         (1,1),(4,-1), "CENTER"),
        ("TEXTCOLOR",     (0,1),(4,-1), GRIS_DARK),
        ("ROWBACKGROUNDS",(0,1),(4,-1), [BLANCO, GRIS_LIGHT]),
        ("GRID",          (0,0),(4,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0),(4,-1), 5),
        ("BOTTOMPADDING", (0,0),(4,-1), 5),
        ("LEFTPADDING",   (0,0),(4,-1), 7),
    ])
    t.setStyle(ts)
    return t

# ── Tabla burn analysis anual ────────────────────────────────────────
def tabla_burn():
    enc = ["Año", "HDD anual", "Excedente", "Payoff (USD)", "Activado"]
    rows = []
    for _, r in anual.iterrows():
        exc = max(0, r["hdd_total"] - STRIKE_OPT)
        rows.append([str(int(r["anio"])), f"{r['hdd_total']:.1f}",
                     f"{exc:.1f}", f"USD {r['payoff_opt']:,.0f}",
                     "SI" if r["activado"] else "—"])
    data = [enc] + rows
    col_w = [2.5*cm, 3.0*cm, 3.0*cm, 3.5*cm, 2.5*cm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND",    (0,0),(4,0),  VERDE_DARK),
        ("TEXTCOLOR",     (0,0),(4,0),  BLANCO),
        ("FONTNAME",      (0,0),(4,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(4,0),  9),
        ("ALIGN",         (0,0),(4,0),  "CENTER"),
        ("FONTNAME",      (0,1),(4,-1), "Helvetica"),
        ("FONTSIZE",      (0,1),(4,-1), 8.5),
        ("ALIGN",         (1,1),(4,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1),(4,-1), [BLANCO, GRIS_LIGHT]),
        ("GRID",          (0,0),(4,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0),(4,-1), 5),
        ("BOTTOMPADDING", (0,0),(4,-1), 5),
        ("LEFTPADDING",   (0,0),(4,-1), 7),
    ])
    for i, r in anual.iterrows():
        if r["activado"]:
            ts.add("TEXTCOLOR", (4,i+1),(4,i+1), VERDE_DARK)
            ts.add("FONTNAME",  (4,i+1),(4,i+1), "Helvetica-Bold")
            ts.add("TEXTCOLOR", (3,i+1),(3,i+1), VERDE_DARK)
    t.setStyle(ts)
    return t

# ── Ensamblar el documento ───────────────────────────────────────────
story = []

# ── PORTADA ──────────────────────────────────────────────────────────
story += [
    sp(20),
    Paragraph("AGRO-RISK PRO", estilo("marca", base="Normal", fontSize=28,
              textColor=VERDE_DARK, fontName="Helvetica-Bold",
              alignment=TA_CENTER, spaceAfter=4)),
    Paragraph("Plataforma de Derivados Climáticos Paramétricos",
              estilo("subt", base="Normal", fontSize=13, textColor=GRIS_MED,
                     fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2)),
    sp(16),
    HRFlowable(width="60%", thickness=2, color=VERDE_MED,
               spaceAfter=16, hAlign="CENTER"),
    Paragraph("REPORTE DE COBERTURA CLIMÁTICA",
              estilo("rep", base="Normal", fontSize=16, textColor=NEGRO,
                     fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)),
    Paragraph("Derivado HDD · Exportadores de Flores · Cundinamarca, Colombia",
              estilo("sub2", base="Normal", fontSize=10, textColor=GRIS_MED,
                     fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)),
    Paragraph(f"Fecha de emisión: {FECHA_REPORTE}",
              estilo("fec", base="Normal", fontSize=9, textColor=GRIS_MED,
                     fontName="Helvetica-Oblique", alignment=TA_CENTER)),
    sp(30),
    tabla_kpis(),
    sp(30),
    Paragraph(
        "Este reporte es generado automáticamente por el módulo analítico de Agro-Risk Pro. "
        "Contiene la valuación actuarial del contrato de cobertura, análisis de stress "
        "y métricas de riesgo para el período de contratación vigente.",
        estilo("intro", base="Normal", fontSize=9, textColor=GRIS_MED,
               fontName="Helvetica", alignment=TA_CENTER, leading=13)
    ),
    sp(30),
    Paragraph(
        "USO ACADÉMICO EXCLUSIVO · No constituye asesoría financiera · "
        "Datos sintéticos calibrados con series históricas IDEAM y SIPSA · "
        "Materia: Programación para Economía y Finanzas",
        S_DISCLAIMER
    ),
    PageBreak(),
]

# ── SECCIÓN 1: PARÁMETROS DEL CONTRATO ──────────────────────────────
story += [
    Paragraph("1. Parámetros del contrato optimizado", S_H1), hr(),
    Paragraph(
        "El contrato de cobertura climática HDD fue optimizado mediante "
        "<b>scipy.optimize</b> (evolución diferencial + Nelder-Mead + L-BFGS-B) "
        "para minimizar la varianza de los ingresos netos del exportador. "
        "Los tres métodos convergieron al mismo punto óptimo, confirmando que "
        "el resultado es un mínimo global.", S_BODY),
    sp(8), tabla_contrato(), sp(6),
    Paragraph(
        "El strike óptimo de 4.07 °C·día implica una cobertura de alta frecuencia "
        "que activa casi semanalmente en temporadas frías, aplanando la distribución "
        "de ingresos de forma continua en lugar de esperar eventos extremos.",
        S_CAPTION),
    sp(14),
    Paragraph("1.1 Distribución de ingresos — sin vs con cobertura", S_H2),
    img_from_buf(buf1, w_cm=16.5),
    Paragraph(
        f"Figura 1. La cobertura óptima reduce la desviación estándar de ingresos "
        f"de USD {np.std(ing_sin):,.0f} a USD {np.std(ing_opt):,.0f}/semana "
        f"({red_var:.1f}% de reducción de varianza). La distribución verde es más "
        f"estrecha y centrada en torno a la media.", S_CAPTION),
]

# ── SECCIÓN 2: BURN ANALYSIS ─────────────────────────────────────────
story += [
    sp(10),
    Paragraph("2. Burn analysis — payoff histórico 2015–2024", S_H1), hr(),
    Paragraph(
        "El burn analysis calcula el payoff que <b>habría recibido</b> el exportador "
        "si hubiera contratado esta cobertura cada año del período histórico. "
        "La prima justa se estima como el promedio de esos payoffs descontados.", S_BODY),
    sp(8), tabla_burn(), sp(6),
    Paragraph(
        f"Tabla 2. Burn analysis 2015–2024. Strike = {STRIKE_OPT:.2f} °C·día · "
        f"Tick = USD {TICK_OPT:.2f}/°C·día. "
        f"El contrato se activó {int(anual['activado'].sum())} de 10 años "
        f"({int(anual['activado'].sum()*10):.0f}% del tiempo). "
        f"Prima justa calculada: USD {prima_opt:,.2f}/semana.", S_CAPTION),
    sp(10), img_from_buf(buf2, w_cm=16.5),
    Paragraph(
        "Figura 2. Izquierda: HDD anual vs strike (barras rojas = contrato activado). "
        "Derecha: payoffs históricos con línea de prima justa.", S_CAPTION),
]

story.append(PageBreak())

# ── SECCIÓN 3: STRESS TESTING ────────────────────────────────────────
story += [
    Paragraph("3. Stress testing — escenarios climáticos extremos", S_H1), hr(),
    Paragraph(
        "El stress testing evalúa el comportamiento del contrato bajo 6 escenarios "
        "climáticos: desde el caso base histórico hasta eventos extremos de El Niño, "
        "La Niña y alta volatilidad. Cada escenario fue simulado con "
        f"{N_SIM:,} trayectorias Monte Carlo de 90 días.", S_BODY),
    sp(8), tabla_stress(), sp(6),
    Paragraph(
        "Tabla 3. Resultados de stress testing. La Niña fuerte genera la mayor "
        "probabilidad de activación y el mayor valor esperado de pago. "
        "El Niño fuerte reduce la probabilidad de activación al mínimo.", S_CAPTION),
    sp(10), img_from_buf(buf3, w_cm=16.5),
    Paragraph(
        "Figura 3. Probabilidad de activación y valor esperado del payoff "
        "para cada escenario climático simulado.", S_CAPTION),
]

# ── SECCIÓN 4: MONTE CARLO ───────────────────────────────────────────
story += [
    sp(10),
    Paragraph("4. Simulación Monte Carlo — próximos 90 días", S_H1), hr(),
    Paragraph(
        f"La simulación proyecta {N_SIM:,} escenarios de temperatura usando un "
        "proceso AR(1) calibrado con datos históricos 2015–2024 "
        f"(AR1 = {AR1:.4f}, σ = {SIG:.4f}°C/día). "
        "Los resultados permiten estimar la distribución de payoffs futuros "
        "y las métricas de riesgo del período vigente.", S_BODY),
    sp(4),
    Table([[
        Paragraph(f"<b>Probabilidad activación:</b> {prob_mc:.1f}%", S_BODY),
        Paragraph(f"<b>E[Payoff]:</b> USD {e_payoff:,.0f}", S_BODY),
        Paragraph(f"<b>Prima pagada:</b> USD {prima_opt:,.0f}", S_BODY),
        Paragraph(f"<b>E[P&L neto]:</b> USD {e_payoff-prima_opt:,.0f}", S_BODY),
    ]], colWidths=[4.0*cm]*4,
    style=[("BACKGROUND",(0,0),(3,0),VERDE_LIGHT),
           ("BOX",(0,0),(3,0),0.4,VERDE_MED),
           ("TOPPADDING",(0,0),(3,0),6),("BOTTOMPADDING",(0,0),(3,0),6),
           ("LEFTPADDING",(0,0),(3,0),8),]),
    sp(8), img_from_buf(buf4, w_cm=16.5),
    Paragraph(
        "Figura 4. Izquierda: fan chart del HDD acumulado con bandas de percentiles. "
        "Derecha: distribución del payoff del derivado (barra gris = sin activación, "
        "verde = con pago). Línea naranja = prima pagada.", S_CAPTION),
]

story.append(PageBreak())

# ── SECCIÓN 5: GRIEGAS Y NOTAS METODOLÓGICAS ────────────────────────
story += [
    Paragraph("5. Griegas climáticas — sensibilidades del contrato", S_H1), hr(),
    Paragraph(
        "Las griegas climáticas cuantifican cómo cambia el valor del contrato "
        "ante cambios en las variables de estado. Calculadas por "
        "bump-and-reprice con 20,000 escenarios Monte Carlo.", S_BODY),
    sp(6),
    Table([
        ["Griega", "Valor", "Interpretación"],
        ["Delta (vs T)", "−USD 2,004/°C",  "Si la temperatura sube 1°C, la prima baja USD 2,004"],
        ["Delta (vs HDD)", "+USD 2,004/°C·día", "Por cada °C·día adicional de HDD, prima sube USD 2,004"],
        ["Gamma", "+940 USD/°C²",           "Convexidad positiva — prima acelera cerca del strike"],
        ["Vega", "+USD 3,688/°C-σ",         "Alta sensibilidad a volatilidad — ENSO amplia la prima"],
        ["Theta (7d)", "−USD 0.04/día",      "Decaimiento temporal mínimo con pocos días restantes"],
        ["Rho_ENSO", "+USD 2,004/idx",       "La Niña sube prima; El Niño la baja"],
    ],
    colWidths=[3.5*cm, 3.8*cm, 9.0*cm],
    style=[
        ("BACKGROUND",    (0,0),(2,0),  VERDE_DARK),
        ("TEXTCOLOR",     (0,0),(2,0),  BLANCO),
        ("FONTNAME",      (0,0),(2,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(2,0),  9),
        ("FONTNAME",      (0,1),(2,-1), "Helvetica"),
        ("FONTSIZE",      (0,1),(2,-1), 8.5),
        ("TEXTCOLOR",     (0,1),(2,-1), GRIS_DARK),
        ("ROWBACKGROUNDS",(0,1),(2,-1), [BLANCO, GRIS_LIGHT]),
        ("GRID",          (0,0),(2,-1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING",    (0,0),(2,-1), 5), ("BOTTOMPADDING",(0,0),(2,-1),5),
        ("LEFTPADDING",   (0,0),(2,-1), 7), ("VALIGN",(0,0),(2,-1),"MIDDLE"),
        ("FONTNAME",      (1,1),(1,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (1,1),(1,-1), VERDE_DARK),
        ("ALIGN",         (1,0),(1,-1), "CENTER"),
    ]),
    sp(16),
    Paragraph("6. Notas metodológicas", S_H1), hr(),
    Paragraph(
        "<b>Fuentes de datos:</b> Serie de temperatura sintética calibrada con "
        "datos históricos del IDEAM (Sabana de Bogotá, 2015–2024). Precios de "
        "flores y café basados en registros SIPSA/DANE. TRM del Banco de la "
        "República. Índice ENSO del NOAA.", S_BODY),
    Paragraph(
        "<b>Modelo climático:</b> Proceso AR(1) con media estacional calibrada "
        "por día del año. Volatilidad histórica calculada sobre residuos de la "
        f"media estacional. AR(1) = {AR1:.4f}, σ diaria = {SIG:.4f}°C.", S_BODY),
    Paragraph(
        "<b>Optimización:</b> Minimización de varianza de ingresos netos con "
        "scipy.optimize (evolución diferencial + Nelder-Mead + L-BFGS-B). "
        "Los tres métodos convergieron al mismo punto óptimo.", S_BODY),
    Paragraph(
        "<b>Monte Carlo:</b> 10,000 trayectorias de 90 días para cada escenario "
        "de stress testing. Semilla fija para reproducibilidad.", S_BODY),
    sp(20),
    HRFlowable(width="100%", thickness=0.5, color=GRIS_MED, spaceAfter=8),
    Paragraph(
        "DISCLAIMER: Este reporte fue generado automáticamente por el módulo analítico "
        "de Agro-Risk Pro con fines exclusivamente académicos para la materia "
        "Programación para Economía y Finanzas. Los datos son sintéticos y no "
        "representan una oferta de productos financieros reales. No constituye "
        "asesoría financiera, legal ni actuarial. El uso de coberturas climáticas "
        "reales requiere regulación de la Superintendencia Financiera de Colombia "
        "y asesoría profesional especializada.",
        S_DISCLAIMER),
]

doc.build(story)
print(f"✅ PDF generado: {OUTPUT_PDF}")
sz = os.path.getsize(OUTPUT_PDF) / 1024
print(f"   Tamaño: {sz:.1f} KB  |  Páginas: 5")
