"""
Agro-Risk Pro — Aplicación Streamlit
3 pestañas: Dashboard · Simulador · Historial
Materia: Programación para Economía y Finanzas

CÓMO CORRER:
  pip install streamlit plotly pandas numpy scipy
  streamlit run agrorisk_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title     = "Agro-Risk Pro",
    page_icon      = "🌿",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ── Estilos globales ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Métricas */
  [data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 14px 18px !important;
  }
  [data-testid="metric-container"] label {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6c757d !important;
    font-family: 'DM Mono', monospace;
  }
  [data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: #0f6e56 !important;
  }

  /* Encabezado brand */
  .brand-header {
    display: flex;
    align-items: baseline;
    gap: 10px;
    margin-bottom: 0.5rem;
  }
  .brand-name {
    font-size: 22px;
    font-weight: 600;
    color: #0f6e56;
    letter-spacing: -0.5px;
  }
  .brand-tag {
    font-size: 12px;
    color: #6c757d;
    font-family: 'DM Mono', monospace;
  }

  /* Badges */
  .badge-green {
    background: #e1f5ee;
    color: #0f6e56;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
  }
  .badge-red {
    background: #fdeaea;
    color: #c0392b;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
  }
  .badge-amber {
    background: #fef9e7;
    color: #b7770d;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
  }

  /* Tarjeta de liquidación */
  .tx-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-left: 4px solid #0f6e56;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
  }
  .tx-card-inactive {
    border-left-color: #adb5bd;
    opacity: 0.7;
  }
  .tx-hash { color: #6c757d; font-size: 10px; word-break: break-all; }

  /* Disclaimer */
  .disclaimer {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: #856404;
    margin-top: 1.5rem;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f8f9fa;
    border-radius: 10px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 20px;
  }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# DATOS Y FUNCIONES COMPARTIDAS
# ════════════════════════════════════════════════════════════════════
@st.cache_data
def cargar_datos():
    """Carga o genera datos base del proyecto."""
    try:
        df = pd.read_csv("temperatura_sabana_10años.csv", parse_dates=["fecha"])
    except FileNotFoundError:
        np.random.seed(42)
        fechas = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        n = len(fechas)
        t = np.arange(n)
        tendencia     = t / 365.25 * 0.02
        estacionalidad = 1.4 * np.sin(2 * np.pi * t / 365.25 - 1.9) + \
                         0.6 * np.sin(4 * np.pi * t / 365.25 + 0.8)
        ruido = np.zeros(n); eps = np.random.normal(0, 1.8, n)
        ruido[0] = eps[0]
        for i in range(1, n): ruido[i] = 0.72 * ruido[i-1] + eps[i]
        t_prom = (13.0 + tendencia + estacionalidad + ruido).round(2)
        t_min  = (t_prom - abs(np.random.normal(4, 1.5, n))).round(2)
        hdd    = np.maximum(0, 10.0 - t_prom).round(4)
        df = pd.DataFrame({
            "fecha": fechas, "t_promedio_c": t_prom, "t_minima_c": t_min,
            "hdd": hdd, "evento_helada": (t_min < 2).astype(int),
            "enso_index": (0.8*np.sin(2*np.pi*t/(4*365.25)+0.3) +
                           np.random.normal(0,0.2,n)).round(3),
            "anio": fechas.year, "mes": fechas.month,
        })
    df["hdd"] = np.maximum(0, 10.0 - df["t_promedio_c"])
    return df

@st.cache_data
def generar_rendimientos(df):
    np.random.seed(42)
    sem = df.set_index("fecha").resample("W").agg(
        hdd_sem=("hdd","sum"), t_min=("t_minima_c","mean"),
        heladas=("evento_helada","sum"), enso=("enso_index","mean"),
    ).reset_index()
    R_BASE = 16_000
    ruido  = np.zeros(len(sem)); eps = np.random.normal(0, 1200, len(sem))
    ruido[0] = eps[0]
    for i in range(1, len(sem)): ruido[i] = 0.45*ruido[i-1]+eps[i]
    sem["ventas"] = (
        R_BASE - 85*sem["hdd_sem"] - 1800*sem["heladas"]
        - 600*sem["enso"].clip(lower=0) + ruido
    ).clip(lower=0)
    return sem

@st.cache_data
def generar_precios_cafe():
    np.random.seed(77)
    fechas = pd.date_range("2020-01-06", "2024-12-31", freq="W")
    precio = 120.0
    precios = []
    for _ in fechas:
        precio = max(85, min(280, precio + np.random.normal(0.3, 5)))
        precios.append(round(precio, 2))
    return pd.DataFrame({"fecha": fechas, "cafe_usc_lb": precios})

def calcular_payoff(hdd_vec, strike, tick):
    return np.maximum(0, hdd_vec - strike) * tick

def prima_justa(hdd_vec, strike, tick, carga=0.20):
    return calcular_payoff(hdd_vec, strike, tick).mean() * (1 + carga)

def colores_plotly():
    return {
        "verde":   "#0f6e56", "verde_light": "#1d9e75",
        "rojo":    "#e24b4a", "naranja":     "#ef9f27",
        "azul":    "#378add", "morado":      "#7f77dd",
        "gris":    "#8b90a0", "fondo":       "#f8f9fa",
        "panel":   "#ffffff",
    }

C = colores_plotly()
PLOTLY_LAYOUT = dict(
    paper_bgcolor = C["panel"],
    plot_bgcolor  = C["panel"],
    font          = dict(family="DM Sans, sans-serif", size=11, color="#495057"),
    margin        = dict(l=10, r=10, t=40, b=10),
    xaxis         = dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.5,
                         zeroline=False),
    yaxis         = dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.5,
                         zeroline=False),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    hovermode     = "x unified",
)

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
df      = cargar_datos()
sem_df  = generar_rendimientos(df)
cafe_df = generar_precios_cafe()

col_brand, col_status = st.columns([3, 1])
with col_brand:
    st.markdown("""
    <div class="brand-header">
      <span class="brand-name">🌿 Agro-Risk Pro</span>
      <span class="brand-tag">Derivados climáticos · Cundinamarca</span>
    </div>
    """, unsafe_allow_html=True)
with col_status:
    st.markdown(
        f'<div style="text-align:right; padding-top:8px">'
        f'<span class="badge-green">● Simulación activa</span>&nbsp;'
        f'<span class="badge-amber">Uso académico</span>'
        f'</div>', unsafe_allow_html=True
    )

st.divider()

# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Parámetros globales")
    municipio  = st.selectbox("Estación meteorológica",
                              ["Facatativá","Zipaquirá","Chía","Ubaté","Fusagasugá"])
    cultivo    = st.selectbox("Tipo de cultivo",
                              ["Flores (clavel)","Papa pastusa","Café"])
    trm        = st.number_input("TRM (COP/USD)", 3800, 5500, 4180, step=10)
    ano_ref    = st.slider("Año de referencia", 2015, 2024, 2024)

    st.divider()
    st.markdown("### 📋 Contrato vigente")
    strike_sidebar = st.number_input("Strike HDD (°C·día)", 0.0, 200.0, 4.07, step=0.5,
                                     format="%.2f")
    tick_sidebar   = st.number_input("Tick (USD/°C·día)", 1.0, 2000.0, 317.7, step=10.0,
                                     format="%.1f")
    hdd_acum_hoy   = st.number_input("HDD acumulado hoy", 0.0, 300.0, 2.3, step=0.1)

    prima_act = prima_justa(df["hdd"].values, strike_sidebar, tick_sidebar)
    st.metric("Prima semanal estimada", f"USD {prima_act:,.0f}")

    en_dinero = hdd_acum_hoy >= strike_sidebar
    if en_dinero:
        pf_actual = (hdd_acum_hoy - strike_sidebar) * tick_sidebar
        st.markdown(
            f'<span class="badge-green">✓ En el dinero — Payoff: USD {pf_actual:,.0f}</span>',
            unsafe_allow_html=True
        )
    else:
        falta = strike_sidebar - hdd_acum_hoy
        st.markdown(
            f'<span class="badge-red">Faltan {falta:.2f} °C·día para activar</span>',
            unsafe_allow_html=True
        )

    st.divider()
    st.caption("v1.0 · PEF 2025 · Datos sintéticos")

# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Dashboard de mercado",
    "🎛️ Simulador de coberturas",
    "📜 Historial de liquidaciones",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD DE MERCADO
# ════════════════════════════════════════════════════════════════════
with tab1:

    # ── KPIs de cabecera ────────────────────────────────────────────
    df_año = df[df["anio"] == ano_ref]
    hdd_año = df_año["hdd"].sum()
    t_min_año = df_año["t_minima_c"].min()
    heladas_año = df_año["evento_helada"].sum()
    enso_actual = df["enso_index"].iloc[-1]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("HDD acumulado", f"{hdd_año:.0f} °C·día",
              f"{hdd_año - df[df['anio']==ano_ref-1]['hdd'].sum():+.0f} vs año ant.")
    k2.metric("T mínima absoluta", f"{t_min_año:.1f}°C")
    k3.metric("Días de helada", f"{heladas_año}",
              delta_color="inverse")
    k4.metric("Índice ENSO", f"{enso_actual:.2f}",
              "El Niño" if enso_actual > 0.5 else ("La Niña" if enso_actual < -0.5 else "Neutro"))
    k5.metric("Café NY", f"{cafe_df['cafe_usc_lb'].iloc[-1]:.1f} ¢/lb",
              f"{cafe_df['cafe_usc_lb'].iloc[-1] - cafe_df['cafe_usc_lb'].iloc[-5]:+.1f}")

    st.markdown("---")

    # ── Gráfica principal: Temperatura + HDD ────────────────────────
    col_g1, col_g2 = st.columns([2, 1])

    with col_g1:
        st.markdown("**Temperatura mínima diaria y HDD — últimos 2 años**")
        df_2y = df[df["anio"] >= ano_ref - 1].copy()
        fig1  = make_subplots(specs=[[{"secondary_y": True}]])

        fig1.add_trace(go.Scatter(
            x=df_2y["fecha"], y=df_2y["t_minima_c"],
            name="T mínima (°C)", line=dict(color=C["azul"], width=1),
            opacity=0.7, fill="tozeroy", fillcolor="rgba(55,138,221,0.06)",
        ), secondary_y=False)

        fig1.add_trace(go.Bar(
            x=df_2y[df_2y["evento_helada"]==1]["fecha"],
            y=[1]*df_2y["evento_helada"].sum(),
            name="Helada", marker_color=C["rojo"], opacity=0.6, width=86400000,
        ), secondary_y=True)

        fig1.add_hline(y=2.0, line_dash="dot", line_color=C["rojo"],
                       opacity=0.5, annotation_text="Helada 2°C")
        fig1.add_hline(y=10.0, line_dash="dot", line_color=C["naranja"],
                       opacity=0.5, annotation_text="Umbral HDD 10°C")

        fig1.update_layout(**PLOTLY_LAYOUT, height=320,
                           title_text="Temperatura mínima · Sabana de Bogotá")
        fig1.update_yaxes(title_text="°C", secondary_y=False)
        fig1.update_yaxes(title_text="Helada", secondary_y=True, showgrid=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col_g2:
        st.markdown("**HDD mensual acumulado**")
        hdd_mes = df_2y.groupby("mes")["hdd"].sum().reset_index()
        hdd_mes["mes_label"] = ["Ene","Feb","Mar","Abr","May","Jun",
                                  "Jul","Ago","Sep","Oct","Nov","Dic"][:len(hdd_mes)]
        fig_bar = go.Figure(go.Bar(
            x=hdd_mes["mes_label"], y=hdd_mes["hdd"],
            marker_color=[C["rojo"] if v > hdd_mes["hdd"].quantile(0.7) else C["azul"]
                          for v in hdd_mes["hdd"]],
            text=hdd_mes["hdd"].round(0).astype(int),
            textposition="outside", textfont_size=10,
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=320,
                               title_text="HDD mensual", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Segunda fila: Café + Correlación ────────────────────────────
    col_cafe, col_corr = st.columns(2)

    with col_cafe:
        st.markdown("**Precio café Arábica NY (USc/lb) — 2020–2024**")
        fig_cafe = go.Figure()
        fig_cafe.add_trace(go.Scatter(
            x=cafe_df["fecha"], y=cafe_df["cafe_usc_lb"],
            name="Café NY", line=dict(color=C["naranja"], width=1.8),
            fill="tozeroy", fillcolor="rgba(239,159,39,0.08)"
        ))
        mm8 = cafe_df["cafe_usc_lb"].rolling(8, min_periods=1).mean()
        fig_cafe.add_trace(go.Scatter(
            x=cafe_df["fecha"], y=mm8,
            name="MM 8 sem", line=dict(color=C["gris"], width=1, dash="dot"),
        ))
        fig_cafe.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_cafe, use_container_width=True)

    with col_corr:
        st.markdown("**Correlación: temperatura vs rendimiento exportador**")
        sub = sem_df.dropna()
        fig_sc = go.Figure(go.Scatter(
            x=sub["t_min"], y=sub["ventas"],
            mode="markers",
            marker=dict(
                color=sub["hdd_sem"], colorscale="RdYlGn_r",
                size=5, opacity=0.6,
                colorbar=dict(title="HDD<br>sem.", thickness=10, len=0.7)
            ),
            text=sub["fecha"].dt.strftime("%b %Y"),
            hovertemplate="<b>%{text}</b><br>T: %{x:.1f}°C<br>Ventas: USD %{y:,.0f}<extra></extra>"
        ))
        fig_sc.update_layout(**PLOTLY_LAYOUT, height=280,
                              xaxis_title="T mínima media (°C)",
                              yaxis_title="Ventas semanales (USD)")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("""
    <div class="disclaimer">
      ⚠️ Los datos climáticos son sintéticos calibrados con series históricas IDEAM.
      Los precios de café son aproximaciones sin datos de bolsa en tiempo real.
      Solo uso académico.
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — SIMULADOR DE COBERTURAS
# ════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("### Diseña tu contrato de cobertura HDD")
    st.caption("Ajusta los parámetros y observa cómo cambia la prima, el payoff esperado y la protección de ingresos.")

    # ── Controles ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    strike = c1.slider("Strike (°C·día)", 0.0, 150.0, strike_sidebar, step=0.5,
                       help="Umbral de activación del contrato")
    tick   = c2.slider("Tick (USD/°C·día)", 10.0, 1500.0, tick_sidebar, step=10.0,
                       help="Pago por cada grado-día sobre el strike")
    cap    = c3.slider("Cap (°C·día)", strike + 5.0, 300.0,
                       min(strike + 40, 200.0), step=5.0,
                       help="Techo máximo del pago")
    carga  = c4.slider("Carga aseguradora (%)", 0, 50, 20, step=5) / 100

    # Calcular métricas
    hdd_hist = df.groupby("anio")["hdd"].sum().values
    pf_hist  = np.minimum(np.maximum(0, hdd_hist - strike), cap - strike) * tick
    prima    = pf_hist.mean() * (1 + carga)
    prob_act = (pf_hist > 0).mean() * 100
    e_payoff = pf_hist.mean()
    payoff_max = (cap - strike) * tick
    pf_sem   = np.minimum(np.maximum(0, sem_df["hdd_sem"] - strike), cap - strike) * tick
    ing_netos = sem_df["ventas"].values - prima + pf_sem.values
    red_var  = (1 - np.var(ing_netos) / np.var(sem_df["ventas"])) * 100

    # ── KPIs ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Prima estimada", f"USD {prima:,.0f}/sem",
              f"≈ {prima/16000*100:.1f}% del ingreso base")
    m2.metric("P(activación)", f"{prob_act:.0f}%",
              f"{int(prob_act/100*10)}/10 años históricos")
    m3.metric("E[Payoff]", f"USD {e_payoff:,.0f}")
    m4.metric("Payoff máximo", f"USD {payoff_max:,.0f}")
    m5.metric("Reducción varianza", f"{red_var:.1f}%",
              delta_color="normal" if red_var > 0 else "inverse")

    st.divider()

    # ── Gráficas del simulador ───────────────────────────────────────
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Función de payoff del contrato**")
        hdd_eje = np.linspace(0, max(hdd_hist) * 1.1, 400)
        pf_eje  = np.minimum(np.maximum(0, hdd_eje - strike), cap - strike) * tick

        fig_pf = go.Figure()
        fig_pf.add_trace(go.Scatter(
            x=hdd_eje, y=pf_eje, name="Payoff",
            line=dict(color=C["verde_light"], width=2.5),
            fill="tozeroy", fillcolor="rgba(29,158,117,0.10)"
        ))
        fig_pf.add_vline(x=strike, line_dash="dash", line_color=C["naranja"],
                         annotation_text=f"Strike {strike:.1f}", annotation_position="top right")
        fig_pf.add_vline(x=cap,    line_dash="dot",  line_color=C["rojo"],
                         annotation_text=f"Cap {cap:.1f}", annotation_position="top left")
        fig_pf.add_hline(y=prima,  line_dash="dot",  line_color=C["gris"],
                         annotation_text=f"Prima USD {prima:,.0f}")

        for hdd_v, anio in zip(hdd_hist, range(2015, 2025)):
            pf_v = min(max(0, hdd_v - strike), cap - strike) * tick
            color = C["rojo"] if pf_v > 0 else C["gris"]
            fig_pf.add_trace(go.Scatter(
                x=[hdd_v], y=[pf_v], mode="markers+text",
                marker=dict(size=8, color=color, symbol="circle"),
                text=[str(anio)], textposition="top center",
                textfont=dict(size=9), showlegend=False,
            ))

        fig_pf.update_layout(**PLOTLY_LAYOUT, height=340,
                              xaxis_title="HDD anual realizado (°C·día)",
                              yaxis_title="Payoff (USD)")
        st.plotly_chart(fig_pf, use_container_width=True)

    with col_s2:
        st.markdown("**Distribución de ingresos — sin vs con cobertura**")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=sem_df["ventas"], name="Sin cobertura",
            nbinsx=50, opacity=0.55,
            marker_color=C["rojo"],
            histnorm="probability density"
        ))
        fig_dist.add_trace(go.Histogram(
            x=ing_netos, name=f"Con cobertura (σ −{red_var:.0f}%)",
            nbinsx=50, opacity=0.65,
            marker_color=C["verde_light"],
            histnorm="probability density"
        ))
        fig_dist.add_vline(x=ing_netos.mean(), line_dash="dot",
                           line_color=C["verde"],
                           annotation_text=f"Media USD {ing_netos.mean():,.0f}")
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=340,
                                barmode="overlay",
                                xaxis_title="Ingreso semanal (USD)",
                                yaxis_title="Densidad")
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Análisis de sensibilidad ────────────────────────────────────
    st.markdown("**Sensibilidad: prima y probabilidad según el strike**")
    strikes_s = np.linspace(max(0.1, strike*0.3), min(150, strike*3 + 20), 30)
    primas_s  = [prima_justa(hdd_hist, s, tick, carga) for s in strikes_s]
    probs_s   = [(np.maximum(0, hdd_hist - s) * tick > 0).mean() * 100 for s in strikes_s]

    fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sens.add_trace(go.Scatter(
        x=strikes_s, y=primas_s, name="Prima (USD)",
        line=dict(color=C["azul"], width=2), fill="tozeroy",
        fillcolor="rgba(55,138,221,0.06)"
    ), secondary_y=False)
    fig_sens.add_trace(go.Scatter(
        x=strikes_s, y=probs_s, name="P(activación) %",
        line=dict(color=C["rojo"], width=2, dash="dash")
    ), secondary_y=True)
    fig_sens.add_vline(x=strike, line_dash="dot", line_color=C["verde"],
                       annotation_text=f"Strike actual {strike:.1f}")
    fig_sens.update_layout(**PLOTLY_LAYOUT, height=260)
    fig_sens.update_yaxes(title_text="Prima (USD)", secondary_y=False)
    fig_sens.update_yaxes(title_text="P(activación) %", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_sens, use_container_width=True)

    # ── Calculadora de escenario puntual ────────────────────────────
    st.markdown("---")
    st.markdown("**Calculadora de escenario puntual**")
    hdd_esc = st.number_input("¿Cuánto HDD esperas que acumule esta temporada?",
                              0.0, 300.0, float(np.median(hdd_hist)), step=1.0,
                              format="%.1f")
    pf_esc  = min(max(0, hdd_esc - strike), cap - strike) * tick
    pl_esc  = pf_esc - prima
    col_calc1, col_calc2, col_calc3 = st.columns(3)
    col_calc1.metric("Payoff del contrato", f"USD {pf_esc:,.0f}")
    col_calc2.metric("P&L neto (payoff − prima)", f"USD {pl_esc:,.0f}",
                     delta_color="normal" if pl_esc >= 0 else "inverse")
    col_calc3.metric("Payoff en COP", f"COP {pf_esc * trm:,.0f}",
                     f"TRM {trm:,}")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORIAL DE LIQUIDACIONES
# ════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("### Historial de liquidaciones en USDC")
    st.caption("Registro simulado de todos los contratos del período histórico. Cada liquidación muestra el hash de transacción simulado.")

    # Construir historial
    hdd_anual = df.groupby("anio").agg(
        hdd_total  = ("hdd",            "sum"),
        t_min_med  = ("t_minima_c",     "mean"),
        dias_helada= ("evento_helada",  "sum"),
    ).round(2).reset_index()

    np.random.seed(123)
    registros = []
    for _, row in hdd_anual.iterrows():
        pf = min(max(0, row["hdd_total"] - strike_sidebar), cap - strike_sidebar) * tick_sidebar \
             if "cap" in dir() else min(max(0, row["hdd_total"] - strike_sidebar), 40 * tick_sidebar)
        activado = pf > 0
        tx_hash  = "0x" + "".join(np.random.choice(list("0123456789abcdef"), 64))
        registros.append({
            "año":          int(row["anio"]),
            "hdd_realizado": row["hdd_total"],
            "excedente":    max(0, row["hdd_total"] - strike_sidebar),
            "payoff_usd":   round(pf, 2),
            "payoff_cop":   round(pf * trm, 0),
            "payoff_usdc":  round(pf, 2),
            "activado":     activado,
            "tx_hash":      tx_hash if activado else "—",
            "t_min_media":  row["t_min_med"],
            "dias_helada":  int(row["dias_helada"]),
        })

    hist_df = pd.DataFrame(registros)

    # ── Métricas resumen ────────────────────────────────────────────
    total_pag  = hist_df[hist_df["activado"]]["payoff_usd"].sum()
    total_prim = prima_justa(hdd_anual["hdd_total"].values, strike_sidebar, tick_sidebar) * 52 * 10
    n_activ    = hist_df["activado"].sum()

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Liquidaciones totales", f"USD {total_pag:,.0f}",
              f"{n_activ} de 10 contratos")
    r2.metric("Primas acumuladas", f"USD {total_prim:,.0f}",
              f"USD {total_prim/10:,.0f}/año promedio")
    r3.metric("Ratio payoff / prima",
              f"{total_pag/(total_prim if total_prim > 0 else 1):.2f}×")
    r4.metric("Total en USDC", f"{total_pag:.2f} USDC",
              "1 USDC ≡ USD 1.00")

    st.divider()

    # ── Gráfica: payoffs por año ────────────────────────────────────
    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Bar(
            x=hist_df["año"],
            y=hist_df["payoff_usd"],
            name="Payoff USD",
            marker_color=[C["rojo"] if a else "#dee2e6" for a in hist_df["activado"]],
            text=[f"USD {v:,.0f}" if v > 0 else "—" for v in hist_df["payoff_usd"]],
            textposition="outside", textfont_size=10,
        ))
        fig_hist.add_hline(
            y=prima_justa(hdd_anual["hdd_total"].values, strike_sidebar, tick_sidebar),
            line_dash="dot", line_color=C["verde"],
            annotation_text="Prima semanal"
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=300,
                                title_text="Payoffs históricos por año",
                                showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        st.markdown("**Activaciones por mes histórico**")
        heladas_mes = df.groupby("mes")["evento_helada"].sum().reset_index()
        meses_labels = ["Ene","Feb","Mar","Abr","May","Jun",
                        "Jul","Ago","Sep","Oct","Nov","Dic"]
        fig_mes = go.Figure(go.Bar(
            x=meses_labels,
            y=heladas_mes["evento_helada"],
            marker_color=[C["rojo"] if v > heladas_mes["evento_helada"].median()
                          else C["azul"] for v in heladas_mes["evento_helada"]],
        ))
        fig_mes.update_layout(**PLOTLY_LAYOUT, height=300,
                               title_text="Días helada por mes", showlegend=False)
        st.plotly_chart(fig_mes, use_container_width=True)

    # ── Tarjetas de transacciones ────────────────────────────────────
    st.markdown("**Registro de transacciones simuladas**")
    filtro = st.radio("Mostrar", ["Todas","Solo activadas","Solo no activadas"],
                      horizontal=True)
    df_show = hist_df.copy()
    if filtro == "Solo activadas":    df_show = df_show[df_show["activado"]]
    if filtro == "Solo no activadas": df_show = df_show[~df_show["activado"]]

    for _, row in df_show.sort_values("año", ascending=False).iterrows():
        clase = "tx-card" if row["activado"] else "tx-card tx-card-inactive"
        badge = (f'<span class="badge-green">LIQUIDADO</span>' if row["activado"]
                 else f'<span class="badge-red">NO ACTIVADO</span>')
        hash_txt = (f'<div class="tx-hash">Tx: {row["tx_hash"]}</div>'
                    if row["activado"] else "")
        st.markdown(f"""
        <div class="{clase}">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <strong>Contrato {row['año']}</strong>
            {badge}
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:6px">
            <div>HDD: <strong>{row['hdd_realizado']:.1f}</strong> °C·día</div>
            <div>Excedente: <strong>{row['excedente']:.1f}</strong></div>
            <div>USD: <strong>{row['payoff_usd']:,.0f}</strong></div>
            <div>USDC: <strong>{row['payoff_usdc']:.2f}</strong></div>
          </div>
          {hash_txt}
        </div>
        """, unsafe_allow_html=True)

    # ── Exportar CSV ────────────────────────────────────────────────
    st.divider()
    csv_hist = hist_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "⬇ Descargar historial como CSV",
        data      = csv_hist,
        file_name = "agrorisk_liquidaciones.csv",
        mime      = "text/csv",
    )

    st.markdown("""
    <div class="disclaimer">
      Los hashes de transacción son simulados y no corresponden a ninguna blockchain real.
      Este historial es de uso académico exclusivo. No constituye registro financiero oficial.
    </div>""", unsafe_allow_html=True)
