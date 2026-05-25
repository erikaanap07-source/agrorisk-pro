"""
Agro-Risk Pro — Esquema de Base de Datos con Pandas
Derivados Paramétricos Climáticos para Exportadores de Cundinamarca
Materia: Programación para Economía y Finanzas
"""

import pandas as pd
import numpy as np
from datetime import datetime, date

# ============================================================
# TABLA 1: ESTACIONES METEOROLÓGICAS
# ============================================================
estaciones = pd.DataFrame({
    "estacion_id":    ["EST001", "EST002", "EST003", "EST004", "EST005"],
    "nombre":         ["Facatativá", "Zipaquirá", "Ubaté", "Chía", "Fusagasugá"],
    "municipio":      ["Facatativá", "Zipaquirá", "Ubaté", "Chía", "Fusagasugá"],
    "departamento":   ["Cundinamarca"] * 5,
    "latitud":        [4.8147, 5.0222, 5.3122, 4.8610, 4.3361],
    "longitud":       [-74.3541, -74.0058, -73.8181, -74.0317, -74.3641],
    "altitud_msnm":   [2586, 2652, 2556, 2562, 1728],
    "fuente_datos":   ["IDEAM"] * 3 + ["Open-Meteo"] * 2,
    "activa":         [True] * 5
})

estaciones = estaciones.set_index("estacion_id")
estaciones.index.name = "estacion_id"

# Tipos correctos
estaciones["altitud_msnm"] = estaciones["altitud_msnm"].astype(int)
estaciones["activa"] = estaciones["activa"].astype(bool)

print("=" * 60)
print("TABLA 1: ESTACIONES METEOROLÓGICAS")
print("=" * 60)
print(estaciones.to_string())
print(f"\nDtypes:\n{estaciones.dtypes}\n")


# ============================================================
# TABLA 2: PRECIOS DE ACTIVOS (Papa, Flores, Café)
# ============================================================
np.random.seed(42)
fechas = pd.date_range(start="2023-01-01", end="2024-12-31", freq="W")

precios_records = []
for fecha in fechas:
    # Papa pastusa — precio mayorista Bogotá (COP/kg) — SIPSA
    precios_records.append({
        "fecha":           fecha,
        "activo_id":       "PAPA_PAS",
        "nombre_activo":   "Papa Pastusa",
        "precio_cop_kg":   round(np.random.normal(1850, 300), 2),
        "volumen_ton":     round(np.random.normal(450, 80), 1),
        "mercado":         "Corabastos Bogotá",
        "fuente":          "DANE-SIPSA",
        "moneda":          "COP"
    })
    # Clavel — flores de exportación (USD/docena) — Asocolflores
    precios_records.append({
        "fecha":           fecha,
        "activo_id":       "FLOR_CLV",
        "nombre_activo":   "Clavel Estándar",
        "precio_cop_kg":   round(np.random.normal(4200, 500), 2),
        "volumen_ton":     round(np.random.normal(120, 30), 1),
        "mercado":         "Exportación Miami",
        "fuente":          "Asocolflores",
        "moneda":          "USD"
    })
    # Café pergamino seco (COP/125kg) — FNC
    precios_records.append({
        "fecha":           fecha,
        "activo_id":       "CAFE_PER",
        "nombre_activo":   "Café Pergamino Seco",
        "precio_cop_kg":   round(np.random.normal(3100, 400), 2),
        "volumen_ton":     round(np.random.normal(85, 20), 1),
        "mercado":         "FNC Colombia",
        "fuente":          "Federación Nacional de Cafeteros",
        "moneda":          "COP"
    })

precios_activos = pd.DataFrame(precios_records)
precios_activos["fecha"] = pd.to_datetime(precios_activos["fecha"])
precios_activos["precio_cop_kg"] = precios_activos["precio_cop_kg"].clip(lower=0)
precios_activos = precios_activos.reset_index(drop=True)
precios_activos.index.name = "precio_id"

print("=" * 60)
print("TABLA 2: PRECIOS DE ACTIVOS")
print("=" * 60)
print(precios_activos.head(9).to_string())
print(f"\nRegistros totales: {len(precios_activos)}")
print(f"Activos: {precios_activos['activo_id'].unique()}")
print(f"Rango fechas: {precios_activos['fecha'].min().date()} → {precios_activos['fecha'].max().date()}\n")


# ============================================================
# TABLA 3: ÍNDICES CLIMÁTICOS (HDD, CDD, Índice El Niño)
# ============================================================
"""
HDD (Heating Degree Days): acumula frío por debajo del umbral base.
    HDD = max(0, T_base - T_promedio_diaria)
    T_base = 10°C para cultivos de Cundinamarca
CDD (Cooling Degree Days): acumula calor por encima del umbral.
    CDD = max(0, T_promedio_diaria - T_base)
Índice Helada: 1 si temperatura mínima < 2°C, 0 si no.
ENSO_index: índice El Niño/La Niña (-2 La Niña fuerte, +2 El Niño fuerte)
"""

T_BASE = 10.0  # °C umbral base para cultivos de Cundinamarca
T_HELADA = 2.0  # °C umbral de helada agronómica

fechas_diarias = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")

indices_records = []
for fecha in fechas_diarias:
    for est_id in ["EST001", "EST002", "EST003"]:
        altitud = estaciones.loc[est_id, "altitud_msnm"]
        # Temperatura promedio simulada (más fría en mayor altitud)
        t_prom = round(np.random.normal(13 - (altitud - 2500) * 0.005, 2.5), 2)
        t_min  = round(t_prom - abs(np.random.normal(4, 1.5)), 2)
        t_max  = round(t_prom + abs(np.random.normal(5, 1.5)), 2)
        hdd    = round(max(0, T_BASE - t_prom), 4)
        cdd    = round(max(0, t_prom - T_BASE), 4)
        helada = 1 if t_min < T_HELADA else 0
        # ENSO simplificado: ciclo sinusoidal + ruido
        enso   = round(0.8 * np.sin(2 * np.pi * fecha.dayofyear / 365) + np.random.normal(0, 0.3), 3)

        indices_records.append({
            "fecha":           fecha,
            "estacion_id":     est_id,
            "t_promedio_c":    t_prom,
            "t_minima_c":      t_min,
            "t_maxima_c":      t_max,
            "hdd":             hdd,
            "cdd":             cdd,
            "hdd_acumulado":   None,   # se calcula abajo
            "evento_helada":   helada,
            "enso_index":      enso,
            "precipitacion_mm": round(abs(np.random.normal(4, 3)), 2)
        })

indices_climaticos = pd.DataFrame(indices_records)
indices_climaticos["fecha"] = pd.to_datetime(indices_climaticos["fecha"])

# Calcular HDD acumulado mensual por estación
indices_climaticos["hdd_acumulado"] = (
    indices_climaticos
    .groupby(["estacion_id", indices_climaticos["fecha"].dt.to_period("M")])["hdd"]
    .transform("cumsum")
    .round(2)
)

print("=" * 60)
print("TABLA 3: ÍNDICES CLIMÁTICOS (HDD / CDD / ENSO)")
print("=" * 60)
print(indices_climaticos.head(6).to_string())
print(f"\nRegistros totales: {len(indices_climaticos):,}")
print(f"Heladas detectadas: {indices_climaticos['evento_helada'].sum()}")
print(f"HDD promedio diario: {indices_climaticos['hdd'].mean():.2f}°C\n")


# ============================================================
# TABLA 4: USUARIOS — EXPORTADORES
# ============================================================
usuarios = pd.DataFrame({
    "usuario_id":       ["USR001", "USR002", "USR003", "USR004", "USR005"],
    "nombre_empresa":   [
        "Flores del Sabana S.A.S.",
        "Exportadora Andina Ltda.",
        "Agropecuaria Villa Rosa",
        "Hacienda El Roble",
        "CundiFlores Export"
    ],
    "tipo_cultivo":     ["Flores", "Papa", "Café", "Flores", "Flores"],
    "municipio":        ["Chía", "Zipaquirá", "Facatativá", "Ubaté", "Facatativá"],
    "estacion_id":      ["EST004", "EST002", "EST001", "EST003", "EST001"],
    "hectareas":        [45.0, 120.0, 30.0, 80.0, 60.0],
    "exporta_usd_anual":[850000, 420000, 180000, 620000, 540000],
    "perfil_riesgo":    ["moderado", "conservador", "agresivo", "moderado", "conservador"],
    "email":            [
        "riesgos@floressabana.co",
        "finanzas@expandina.co",
        "gerencia@villarosa.co",
        "admin@haciendaroble.co",
        "cfo@cundiflores.co"
    ],
    "fecha_registro":   pd.to_datetime([
        "2024-01-15", "2024-02-01", "2024-03-10",
        "2024-01-28", "2024-04-05"
    ]),
    "activo":           [True] * 5
})

usuarios = usuarios.set_index("usuario_id")

print("=" * 60)
print("TABLA 4: USUARIOS — EXPORTADORES")
print("=" * 60)
print(usuarios.to_string())
print()


# ============================================================
# TABLA 5: TRANSACCIONES DE COBERTURA (Derivados Paramétricos)
# ============================================================
"""
Derivado paramétrico: el pago se activa automáticamente si el índice
climático supera/baja del STRIKE, sin necesidad de demostrar pérdida.

Tipos de derivado:
  - HDD_PUT:  paga si HDD_acumulado > strike_hdd (exceso de frío → helada)
  - CDD_CALL: paga si CDD_acumulado > strike_cdd (exceso de calor → El Niño)
  - HELADA_BINARIA: paga monto fijo si hay N días de helada en el período

Prima = precio que paga el exportador por la cobertura
Payoff = lo que recibe si el evento ocurre
"""

transacciones = pd.DataFrame({
    "tx_id":            ["TX0001","TX0002","TX0003","TX0004","TX0005","TX0006"],
    "usuario_id":       ["USR001","USR002","USR001","USR003","USR004","USR005"],
    "estacion_id":      ["EST004","EST002","EST004","EST001","EST003","EST001"],
    "tipo_derivado":    ["HDD_PUT","HDD_PUT","HELADA_BINARIA","CDD_CALL","HDD_PUT","HELADA_BINARIA"],
    "activo_subyacente":["FLOR_CLV","PAPA_PAS","FLOR_CLV","CAFE_PER","FLOR_CLV","FLOR_CLV"],
    "fecha_inicio":     pd.to_datetime(["2024-04-01","2024-04-01","2024-05-01",
                                        "2024-03-15","2024-04-15","2024-05-01"]),
    "fecha_vencimiento":pd.to_datetime(["2024-08-31","2024-08-31","2024-07-31",
                                        "2024-09-30","2024-09-30","2024-08-31"]),
    # Strike: umbral que activa el pago
    "strike_hdd":       [180.0, 200.0, None, None, 160.0, None],
    "strike_cdd":       [None, None, None, 50.0, None, None],
    "strike_dias_helada":[None, None, 5, None, None, 3],
    # Monto nocional: valor del cultivo que se desea cubrir (COP)
    "monto_nocional_cop":[120_000_000, 85_000_000, 60_000_000,
                          35_000_000, 95_000_000, 75_000_000],
    # Prima pagada por la cobertura (COP) — aprox 3-6% del nocional
    "prima_cop":        [5_400_000, 3_400_000, 2_400_000,
                         1_400_000, 4_275_000, 3_000_000],
    # Tasa de prima (%)
    "tasa_prima_pct":   [4.5, 4.0, 4.0, 4.0, 4.5, 4.0],
    # Payoff máximo si el evento ocurre (COP)
    "payoff_maximo_cop":[96_000_000, 68_000_000, 48_000_000,
                         28_000_000, 76_000_000, 60_000_000],
    "estado":           ["activa","activa","activa","activa","activa","activa"],
    # HDD/CDD acumulado a la fecha (se actualizaría con datos reales)
    "indice_actual":    [142.3, 178.6, None, 38.2, 95.1, None],
    "dias_helada_actual":[None, None, 2, None, None, 1],
    "evento_activado":  [False, False, False, False, False, False]
})

transacciones = transacciones.set_index("tx_id")

# Calcular si el derivado ya se activó
def verificar_activacion(row):
    if row["tipo_derivado"] == "HDD_PUT":
        if pd.notna(row["indice_actual"]) and pd.notna(row["strike_hdd"]):
            return row["indice_actual"] >= row["strike_hdd"]
    elif row["tipo_derivado"] == "CDD_CALL":
        if pd.notna(row["indice_actual"]) and pd.notna(row["strike_cdd"]):
            return row["indice_actual"] >= row["strike_cdd"]
    elif row["tipo_derivado"] == "HELADA_BINARIA":
        if pd.notna(row["dias_helada_actual"]) and pd.notna(row["strike_dias_helada"]):
            return row["dias_helada_actual"] >= row["strike_dias_helada"]
    return False

transacciones["evento_activado"] = transacciones.apply(verificar_activacion, axis=1)

print("=" * 60)
print("TABLA 5: TRANSACCIONES DE COBERTURA")
print("=" * 60)
print(transacciones[["usuario_id","tipo_derivado","activo_subyacente",
                       "strike_hdd","monto_nocional_cop","prima_cop",
                       "payoff_maximo_cop","evento_activado"]].to_string())


# ============================================================
# MÉTRICAS RESUMEN DEL PORTAFOLIO
# ============================================================
print("\n" + "=" * 60)
print("MÉTRICAS DEL PORTAFOLIO DE COBERTURAS")
print("=" * 60)
total_nocional = transacciones["monto_nocional_cop"].sum()
total_primas   = transacciones["prima_cop"].sum()
total_payoff   = transacciones["payoff_maximo_cop"].sum()
cobertura_pct  = (total_payoff / total_nocional) * 100
activadas      = transacciones["evento_activado"].sum()

print(f"  Contratos activos:         {len(transacciones)}")
print(f"  Nocional total cubierto:   COP {total_nocional:>15,.0f}")
print(f"  Primas cobradas:           COP {total_primas:>15,.0f}")
print(f"  Payoff máximo posible:     COP {total_payoff:>15,.0f}")
print(f"  Cobertura sobre nocional:  {cobertura_pct:.1f}%")
print(f"  Derivados activados:       {activadas} de {len(transacciones)}")


# ============================================================
# EXPORTAR TODAS LAS TABLAS A CSV
# ============================================================
estaciones.to_csv("outputs/01_estaciones.csv")
precios_activos.to_csv("outputs/02_precios_activos.csv")
indices_climaticos.to_csv("outputs/03_indices_climaticos.csv", index=False)
usuarios.to_csv("outputs/04_usuarios.csv")
transacciones.to_csv("outputs/05_transacciones.csv")

print("\n✅ Todas las tablas exportadas a CSV correctamente.")
print("   Archivos: 01_estaciones.csv | 02_precios_activos.csv |")
print("             03_indices_climaticos.csv | 04_usuarios.csv | 05_transacciones.csv")
