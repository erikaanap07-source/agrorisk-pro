"""
Agro-Risk Pro — Calculadora HDD (Heating Degree Days)
Umbral configurable · Acumulado mensual y anual
Materia: Programación para Economía y Finanzas
"""

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

MESES = ["Ene","Feb","Mar","Abr","May","Jun",
         "Jul","Ago","Sep","Oct","Nov","Dic"]


# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: calcular_hdd
# ═══════════════════════════════════════════════════════════════════
def calcular_hdd(
    serie: pd.Series,
    fechas: pd.Series,
    umbral: float = 18.0,
    col_temp: str = "temperatura"
) -> dict:
    """
    Calcula el índice HDD (Heating Degree Days) a partir de
    una serie de temperaturas diarias.

    Fórmula:
        HDD_diario = max(0, umbral - T_diaria)

    El máx(0, ...) garantiza que valores negativos se ignoran:
    si la temperatura está POR ENCIMA del umbral, HDD = 0
    (no hay demanda de calefacción ese día).

    Parámetros
    ----------
    serie    : pd.Series — temperaturas diarias (°C)
    fechas   : pd.Series — fechas correspondientes (datetime)
    umbral   : float     — temperatura base en °C (default 18°C, estándar internacional)
    col_temp : str       — nombre descriptivo de la columna (para reportes)

    Retorna
    -------
    dict con:
        "diario"   : DataFrame con HDD por día
        "mensual"  : DataFrame con HDD total por mes × año
        "anual"    : DataFrame con HDD acumulado por año
        "tabla_pivot": tabla pivote año × mes (lista para presentar)
        "stats"    : dict con estadísticas clave
    """

    # ── Validaciones ────────────────────────────────────────────────
    if not isinstance(serie, pd.Series):
        raise TypeError("'serie' debe ser un pd.Series de temperaturas.")
    if len(serie) != len(fechas):
        raise ValueError("'serie' y 'fechas' deben tener la misma longitud.")
    if not pd.api.types.is_numeric_dtype(serie):
        raise TypeError("'serie' debe contener valores numéricos.")

    # ── Construir DataFrame de trabajo ──────────────────────────────
    df = pd.DataFrame({
        "fecha":    pd.to_datetime(fechas),
        col_temp:   serie.values
    }).dropna(subset=[col_temp]).copy()

    df = df.sort_values("fecha").reset_index(drop=True)

    # ── Cálculo HDD diario ──────────────────────────────────────────
    # max(0, umbral - T) — nunca negativo
    df["hdd_diario"] = np.maximum(0.0, umbral - df[col_temp])

    # Columnas auxiliares de tiempo
    df["anio"] = df["fecha"].dt.year
    df["mes"]  = df["fecha"].dt.month

    # ── Agregación mensual ──────────────────────────────────────────
    mensual = (
        df.groupby(["anio", "mes"])
        .agg(
            hdd_mes        = ("hdd_diario", "sum"),
            t_media_mes    = (col_temp,     "mean"),
            t_min_mes      = (col_temp,     "min"),
            dias_con_hdd   = ("hdd_diario", lambda x: (x > 0).sum()),
            dias_total     = ("hdd_diario", "count"),
        )
        .round(2)
        .reset_index()
    )

    # HDD acumulado dentro del año (columna para gráficos)
    mensual["hdd_acum_año"] = mensual.groupby("anio")["hdd_mes"].cumsum().round(2)

    # ── Agregación anual ────────────────────────────────────────────
    anual = (
        mensual.groupby("anio")
        .agg(
            hdd_total      = ("hdd_mes",      "sum"),
            mes_mas_frio   = ("hdd_mes",      lambda x: MESES[x.idxmax() % 12]
                                              if not x.empty else "—"),
            hdd_mes_max    = ("hdd_mes",      "max"),
            dias_con_hdd   = ("dias_con_hdd", "sum"),
        )
        .round(2)
        .reset_index()
    )

    # ── Tabla pivote: filas = año, columnas = mes ───────────────────
    pivot = mensual.pivot(index="anio", columns="mes", values="hdd_mes")
    pivot.columns = [MESES[m - 1] for m in pivot.columns]
    pivot["TOTAL"] = pivot.sum(axis=1).round(1)
    pivot = pivot.round(1)

    # ── Estadísticas clave ──────────────────────────────────────────
    stats = {
        "umbral_c":         umbral,
        "n_dias":           len(df),
        "hdd_total_periodo":round(df["hdd_diario"].sum(), 1),
        "hdd_promedio_dia": round(df["hdd_diario"].mean(), 3),
        "hdd_max_dia":      round(df["hdd_diario"].max(), 2),
        "dias_sin_hdd":     int((df["hdd_diario"] == 0).sum()),
        "pct_dias_sin_hdd": round((df["hdd_diario"] == 0).sum() / len(df) * 100, 1),
        "año_mas_frio":     int(anual.loc[anual["hdd_total"].idxmax(), "anio"]),
        "año_mas_calido":   int(anual.loc[anual["hdd_total"].idxmin(), "anio"]),
    }

    return {
        "diario":      df,
        "mensual":     mensual,
        "anual":       anual,
        "tabla_pivot": pivot,
        "stats":       stats,
    }


# ═══════════════════════════════════════════════════════════════════
# FUNCIÓN AUXILIAR: comparar_umbrales
# ═══════════════════════════════════════════════════════════════════
def comparar_umbrales(
    serie: pd.Series,
    fechas: pd.Series,
    umbrales: list = [10.0, 15.0, 18.0, 20.0]
) -> pd.DataFrame:
    """
    Calcula HDD anual para múltiples umbrales y retorna tabla comparativa.
    Útil para calibrar el strike de derivados paramétricos.
    """
    resultados = []
    for u in umbrales:
        res = calcular_hdd(serie, fechas, umbral=u)
        for _, row in res["anual"].iterrows():
            resultados.append({
                "umbral_c": u,
                "anio":     int(row["anio"]),
                "hdd_total":row["hdd_total"]
            })
    return pd.DataFrame(resultados).pivot(index="anio",
                                          columns="umbral_c",
                                          values="hdd_total").round(1)


# ═══════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Cargar datos climáticos del proyecto
    df_raw = pd.read_csv(
        "outputs/temperatura_sabana_10años.csv",
        parse_dates=["fecha"]
    )

    # ── Calcular HDD con umbral 18°C (estándar internacional) ───────
    print("=" * 65)
    print("CALCULANDO HDD — Umbral 18°C — Sabana de Bogotá 2015–2024")
    print("=" * 65)

    resultado = calcular_hdd(
        serie    = df_raw["t_promedio_c"],
        fechas   = df_raw["fecha"],
        umbral   = 18.0,
        col_temp = "t_promedio_c"
    )

    # ── Estadísticas generales ───────────────────────────────────────
    s = resultado["stats"]
    print(f"\n  Umbral base:              {s['umbral_c']}°C")
    print(f"  Días analizados:          {s['n_dias']:,}")
    print(f"  HDD total del período:    {s['hdd_total_periodo']:,.1f} °C·día")
    print(f"  HDD promedio diario:      {s['hdd_promedio_dia']} °C·día")
    print(f"  HDD máximo en un día:     {s['hdd_max_dia']} °C·día")
    print(f"  Días sin demanda (HDD=0): {s['dias_sin_hdd']:,} ({s['pct_dias_sin_hdd']}%)")
    print(f"  Año más frío:             {s['año_mas_frio']}")
    print(f"  Año más cálido:           {s['año_mas_calido']}")

    # ── Tabla pivote: año × mes ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("TABLA HDD MENSUAL ACUMULADO POR AÑO (°C·día)")
    print("Umbral 18°C · valores = 0 si T ≥ 18°C ese mes")
    print("=" * 65)
    print(resultado["tabla_pivot"].to_string())

    # ── Tabla anual resumida ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESUMEN ANUAL")
    print("=" * 65)
    print(resultado["anual"].to_string(index=False))

    # ── Comparación de umbrales ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("SENSIBILIDAD DE HDD A DISTINTOS UMBRALES (HDD anual total)")
    print("Columnas = umbral °C  |  Filas = año")
    print("=" * 65)
    comp = comparar_umbrales(
        df_raw["t_promedio_c"],
        df_raw["fecha"],
        umbrales=[10.0, 15.0, 18.0, 20.0]
    )
    print(comp.to_string())
    print("\nInterpretación para derivados:")
    print("  Umbral 10°C → strike conservador (cubre heladas fuertes)")
    print("  Umbral 18°C → strike estándar internacional")
    print("  Umbral 20°C → strike agresivo (más pagos, prima más alta)")


    # ═══════════════════════════════════════════════════════════════
    # VISUALIZACIÓN
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(18, 14), facecolor=C_FONDO)
    fig.suptitle("Agro-Risk Pro  ·  HDD 18°C — Heating Degree Days — Sabana de Bogotá",
                 fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.32,
                           left=0.07, right=0.97, top=0.94, bottom=0.06)

    mensual = resultado["mensual"]
    anual   = resultado["anual"]
    años    = sorted(mensual["anio"].unique())

    # ── Panel A: HDD mensual por año (barras apiladas) ───────────────
    ax1   = fig.add_subplot(gs[0, :])
    cmap  = plt.cm.get_cmap("Blues", len(años) + 2)
    colores_año = [cmap(i + 2) for i in range(len(años))]

    bottom = np.zeros(12)
    for i, año in enumerate(años):
        sub = mensual[mensual["anio"] == año].set_index("mes").reindex(range(1, 13))
        vals = sub["hdd_mes"].fillna(0).values
        bars = ax1.bar(np.arange(12), vals, bottom=bottom,
                       color=colores_año[i], alpha=0.88,
                       label=str(año), width=0.75)
        bottom += vals

    ax1.set_xticks(range(12)); ax1.set_xticklabels(MESES, fontsize=9)
    ax1.set_ylabel("HDD mensual acumulado 10 años (°C·día)")
    ax1.set_title("A  HDD Mensual Apilado por Año — Umbral 18°C", fontsize=10, pad=8)
    ax1.legend(ncol=len(años), fontsize=8, framealpha=0.15,
               loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel B: Serie temporal HDD diario ──────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    df_d = resultado["diario"]
    ax2.fill_between(df_d["fecha"], df_d["hdd_diario"],
                     alpha=0.35, color=C_AZUL)
    ax2.plot(df_d["fecha"], df_d["hdd_diario"],
             color=C_AZUL, lw=0.6, alpha=0.7)
    # MM 30 días
    mm30 = df_d["hdd_diario"].rolling(30, min_periods=1).mean()
    ax2.plot(df_d["fecha"], mm30, color=C_NARANJA, lw=1.5,
             label="MM 30 días")
    ax2.axhline(df_d["hdd_diario"].mean(), color=C_VERDE,
                lw=1, ls="--", alpha=0.7,
                label=f"Media {df_d['hdd_diario'].mean():.2f}°C·día")
    ax2.set_ylabel("HDD diario (°C·día)")
    ax2.set_title("B  Serie Temporal HDD Diario", fontsize=10, pad=8)
    ax2.legend(fontsize=8, framealpha=0.2); ax2.grid(alpha=0.2)

    # ── Panel C: HDD total anual + tendencia ────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    x_a = np.arange(len(años))
    bars_a = ax3.bar(x_a, anual["hdd_total"], color=C_AZUL,
                     alpha=0.75, width=0.65, label="HDD anual")

    # Colorear el año más frío y más cálido
    idx_max = anual["hdd_total"].idxmax()
    idx_min = anual["hdd_total"].idxmin()
    bars_a[idx_max].set_color(C_ROJO)
    bars_a[idx_min].set_color(C_VERDE)

    # Etiquetas
    for bar, val in zip(bars_a, anual["hdd_total"]):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 10,
                 f"{val:.0f}", ha="center", fontsize=8.5, color=C_TEXTO)

    # Línea de tendencia
    from numpy.polynomial import polynomial as P
    coef = np.polyfit(x_a, anual["hdd_total"], 1)
    ax3.plot(x_a, np.polyval(coef, x_a),
             color=C_NARANJA, lw=1.5, ls="--",
             label=f"Tendencia {coef[0]:+.1f}/año")

    ax3.set_xticks(x_a); ax3.set_xticklabels(años, fontsize=8.5)
    ax3.set_ylabel("HDD total anual (°C·día)")
    ax3.set_title("C  HDD Total Anual + Tendencia", fontsize=10, pad=8)
    ax3.legend(fontsize=8, framealpha=0.2); ax3.grid(axis="y", alpha=0.3)

    # ── Panel D: Mapa de calor pivote año × mes ─────────────────────
    ax4   = fig.add_subplot(gs[2, 0])
    piv   = resultado["tabla_pivot"].drop(columns="TOTAL")
    im    = ax4.imshow(piv.values, aspect="auto",
                       cmap="YlOrRd", vmin=0, vmax=piv.values.max())
    ax4.set_xticks(range(12)); ax4.set_xticklabels(MESES, fontsize=8.5)
    ax4.set_yticks(range(len(años))); ax4.set_yticklabels(años, fontsize=8.5)
    for i in range(len(años)):
        for j in range(12):
            v = piv.values[i, j]
            ax4.text(j, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if v > piv.values.max() * 0.6 else C_TEXTO)
    cb = fig.colorbar(im, ax=ax4, fraction=0.03, pad=0.02)
    cb.set_label("°C·día", fontsize=8)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C_SUB, fontsize=8)
    ax4.set_title("D  Mapa de Calor HDD — Año × Mes", fontsize=10, pad=8)

    # ── Panel E: Sensibilidad de HDD a distintos umbrales ───────────
    ax5   = fig.add_subplot(gs[2, 1])
    colores_u = [C_VERDE, C_AZUL, C_NARANJA, C_ROJO]
    for col, (umbral_c, serie_u) in zip(colores_u, comp.items()):
        ax5.plot(comp.index, serie_u, marker="o", ms=5,
                 color=col, lw=1.8, label=f"Umbral {umbral_c}°C")
    ax5.set_xticks(comp.index); ax5.set_xticklabels(comp.index, fontsize=8.5)
    ax5.set_ylabel("HDD total anual (°C·día)")
    ax5.set_title("E  Sensibilidad HDD a Distintos Umbrales", fontsize=10, pad=8)
    ax5.legend(fontsize=8.5, framealpha=0.2); ax5.grid(alpha=0.3)
    ax5.fill_between(comp.index, comp[10.0], comp[20.0],
                     alpha=0.08, color=C_AZUL, label="Rango 10–20°C")

    plt.savefig("outputs/fig5_hdd.png", dpi=150,
                bbox_inches="tight", facecolor=C_FONDO)
    plt.close()
    print("\n✅ fig5_hdd.png guardada")

    # Exportar tablas
    resultado["tabla_pivot"].to_csv("outputs/08_hdd_pivot.csv")
    resultado["mensual"].to_csv("outputs/08_hdd_mensual.csv", index=False)
    print("✅ 08_hdd_pivot.csv y 08_hdd_mensual.csv exportados")
