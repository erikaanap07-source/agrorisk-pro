"""
Agro-Risk Pro — Serie de Tiempo Sintética de Temperatura
Sabana de Bogotá / Cundinamarca — 10 años diarios
Materia: Programación para Economía y Finanzas

Modelo:
    T(t) = T_base + estacionalidad(t) + tendencia(t) + ruido(t)

    T_base        : temperatura media histórica de la Sabana (~13°C)
    estacionalidad: patrón anual basado en datos IDEAM
                    Cundinamarca tiene 2 temporadas secas (dic-feb, jun-ago)
                    y 2 lluviosas (mar-may, sep-nov)
    tendencia     : calentamiento global ~+0.02°C/año (IPCC AR6 para Andes)
    ruido         : ARMA(1,1) para autocorrelación climática realista
"""

import pandas as pd
import numpy as np

# ── Semilla reproducible ────────────────────────────────────────────
np.random.seed(2024)

# ── Parámetros del modelo ───────────────────────────────────────────
FECHA_INICIO    = "2015-01-01"
FECHA_FIN       = "2024-12-31"
T_BASE          = 13.0          # °C media histórica Sabana de Bogotá
CALENTAMIENTO   = 0.02          # °C por año (tendencia IPCC AR6 Andes tropicales)
T_HELADA        = 2.0           # °C umbral agronómico de helada
T_BASE_HDD      = 10.0          # °C umbral para HDD (Heating Degree Days)

# Parámetros del ruido ARMA(1,1)
AR_COEF         = 0.72          # autocorrelación diaria (persistencia climática)
MA_COEF         = 0.30          # componente media móvil
SIGMA_RUIDO     = 1.8           # desviación estándar del shock diario (°C)

# ── Fechas ──────────────────────────────────────────────────────────
fechas = pd.date_range(start=FECHA_INICIO, end=FECHA_FIN, freq="D")
n      = len(fechas)
t      = np.arange(n)           # días desde el inicio

# ── 1. Tendencia de calentamiento global ────────────────────────────
tendencia = (t / 365.25) * CALENTAMIENTO

# ── 2. Estacionalidad anual (Sabana de Bogotá) ──────────────────────
# Cundinamarca: más frío en jul-ago (temporada seca de mitad de año)
# más cálido en dic-feb (temporada seca de fin de año)
# Se modela como suma de armónicos de Fourier calibrados con IDEAM
dia_del_año = t % 365.25

# Primer armónico: ciclo anual principal
amp1   = 1.4
fase1  = 2 * np.pi * dia_del_año / 365.25
# Segundo armónico: semestral (2 temporadas lluviosas)
amp2   = 0.6
fase2  = 4 * np.pi * dia_del_año / 365.25
# Desfase empírico para Cundinamarca: mínimo en jul (~día 195)
desfase = -1.9

estacionalidad = (amp1 * np.sin(fase1 + desfase) +
                  amp2 * np.sin(fase2 + 0.8))

# ── 3. Ruido ARMA(1,1) ──────────────────────────────────────────────
epsilon  = np.random.normal(0, SIGMA_RUIDO, n)  # shocks blancos
ruido    = np.zeros(n)
ruido[0] = epsilon[0]
for i in range(1, n):
    ruido[i] = AR_COEF * ruido[i-1] + epsilon[i] + MA_COEF * epsilon[i-1]

# ── 4. Temperatura promedio diaria ──────────────────────────────────
t_promedio = T_BASE + tendencia + estacionalidad + ruido

# ── 5. Temperatura mínima y máxima ──────────────────────────────────
# Rango diario: mayor en temporada seca, menor en lluviosa
rango_base  = 8.5   # °C rango típico Sabana
rango_estac = 1.5 * np.cos(fase1 + desfase + 0.5)
rango_diario = rango_base + rango_estac + np.random.normal(0, 0.8, n)
rango_diario = np.clip(rango_diario, 4, 16)

t_minima = t_promedio - rango_diario * 0.45 + np.random.normal(0, 0.4, n)
t_maxima = t_promedio + rango_diario * 0.55 + np.random.normal(0, 0.4, n)

# ── 6. Índices derivados ─────────────────────────────────────────────
hdd           = np.maximum(0, T_BASE_HDD - t_promedio)
cdd           = np.maximum(0, t_promedio - T_BASE_HDD)
evento_helada = (t_minima < T_HELADA).astype(int)

# ── 7. Precipitación sintética (correlacionada con temperatura) ──────
# Más lluvia cuando temperatura baja (temporadas lluviosas)
base_precip   = 3.5 - 0.15 * estacionalidad
precip_latente = base_precip + np.random.normal(0, 2.5, n)
precipitacion = np.where(
    precip_latente > 0,
    np.random.exponential(precip_latente.clip(min=0.5)),
    0.0
).round(2)
precipitacion = precipitacion.clip(0, 80)

# ── 8. Índice ENSO simplificado ───────────────────────────────────────
# Ciclo aproximado de 3.5-7 años con deriva estocástica
enso_ciclo = 0.7 * np.sin(2 * np.pi * t / (4 * 365.25) + 0.3)
enso_ruido = pd.Series(np.random.normal(0, 0.2, n)).ewm(span=90).mean().values
enso_index = (enso_ciclo + enso_ruido).round(3)

# ── 9. Construir DataFrame ────────────────────────────────────────────
df = pd.DataFrame({
    "fecha":            fechas,
    "t_promedio_c":     t_promedio.round(2),
    "t_minima_c":       t_minima.round(2),
    "t_maxima_c":       t_maxima.round(2),
    "precipitacion_mm": precipitacion,
    "hdd":              hdd.round(4),
    "cdd":              cdd.round(4),
    "evento_helada":    evento_helada,
    "enso_index":       enso_index,
    "tendencia_c":      tendencia.round(4),
    "estacionalidad_c": estacionalidad.round(4),
    "anio":             fechas.year,
    "mes":              fechas.month,
    "dia_semana":       fechas.day_name(),
    "trimestre":        fechas.quarter
})

# ── 10. Columnas acumuladas mensuales (para valuación de derivados) ───
df["hdd_acum_mes"] = df.groupby([df["anio"], df["mes"]])["hdd"].cumsum().round(2)
df["cdd_acum_mes"] = df.groupby([df["anio"], df["mes"]])["cdd"].cumsum().round(2)
df["dias_helada_mes"] = df.groupby([df["anio"], df["mes"]])["evento_helada"].cumsum()

# ── 11. Estadísticas de validación ────────────────────────────────────
print("=" * 60)
print("VALIDACIÓN DEL MODELO — Sabana de Bogotá (2015–2024)")
print("=" * 60)
print(f"  Registros generados:       {len(df):,} días")
print(f"  T promedio total:          {df['t_promedio_c'].mean():.2f}°C  (ref IDEAM: ~13°C)")
print(f"  T mínima absoluta:         {df['t_minima_c'].min():.2f}°C")
print(f"  T máxima absoluta:         {df['t_maxima_c'].max():.2f}°C")
print(f"  Días de helada total:      {df['evento_helada'].sum():,}")
print(f"  Días de helada por año:    {df['evento_helada'].sum()/10:.1f}")
print(f"  Precipitación media/día:   {df['precipitacion_mm'].mean():.2f} mm")
print(f"  HDD promedio diario:       {df['hdd'].mean():.3f}°C")
print()

# Tendencia por año (validar calentamiento)
print("  Temperatura media por año:")
print("  " + "-" * 35)
for anio, grp in df.groupby("anio"):
    delta = grp["t_promedio_c"].mean() - df[df["anio"]==2015]["t_promedio_c"].mean()
    barra = "█" * int(abs(delta) * 10)
    signo = "+" if delta >= 0 else ""
    print(f"  {anio}: {grp['t_promedio_c'].mean():.2f}°C  ({signo}{delta:.2f}°C)  {barra}")

print()

# Heladas por mes (validar estacionalidad)
print("  Heladas por mes (promedio anual):")
heladas_mes = df.groupby("mes")["evento_helada"].sum() / 10
meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
for i, (mes, val) in enumerate(heladas_mes.items()):
    barra = "█" * int(val)
    print(f"  {meses[i]:3s}: {val:4.1f}  {barra}")

# ── 12. Exportar CSV ──────────────────────────────────────────────────
output_path = "outputs/temperatura_sabana_10años.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n✅ CSV exportado: {output_path}")
print(f"   Columnas: {list(df.columns)}")
