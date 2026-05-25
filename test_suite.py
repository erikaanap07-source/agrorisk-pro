"""
Agro-Risk Pro — Suite de Pruebas
12 casos de prueba: errores, límites, validaciones
Materia: Programación para Economía y Finanzas
"""

import json
import math
import numpy as np
import pandas as pd
from decimal import Decimal
import traceback
import sys
import os

# Importar funciones del proyecto
sys.path.insert(0, "/mnt/user-data/outputs")

# ── Funciones core a testear (definidas inline para independencia) ───
def calcular_hdd(temperatura: float, umbral: float = 10.0) -> float:
    return max(0.0, umbral - temperatura)

def calcular_payoff(hdd_realizado: float, strike: float, tick: float,
                    cap: float = None) -> float:
    if strike < 0:
        raise ValueError(f"Strike no puede ser negativo: {strike}")
    if tick < 0:
        raise ValueError(f"Tick no puede ser negativo: {tick}")
    excedente = max(0.0, hdd_realizado - strike)
    if cap is not None:
        excedente = min(excedente, cap - strike)
    return excedente * tick

def calcular_prima(hdd_serie: list, strike: float, tick: float,
                   carga: float = 0.20, cap: float = None) -> float:
    if not hdd_serie:
        raise ValueError("Serie HDD vacía")
    if carga < 0 or carga > 1:
        raise ValueError(f"Carga debe estar entre 0 y 1: {carga}")
    payoffs = [calcular_payoff(h, strike, tick, cap) for h in hdd_serie]
    return float(np.mean(payoffs) * (1 + carga))

def convertir_cop(monto_usd: float, trm: float) -> float:
    if trm <= 0:
        raise ValueError(f"TRM debe ser positiva: {trm}")
    return round(monto_usd * trm, 0)

def generar_tx_json(usuario_id: str, monto_usdc: float,
                    valor_cop: float, fecha: str) -> dict:
    from datetime import datetime
    try:
        datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Fecha inválida: {fecha}. Usar YYYY-MM-DD")
    if monto_usdc < 0:
        raise ValueError(f"Monto USDC no puede ser negativo: {monto_usdc}")
    return {
        "usuario_id":   usuario_id,
        "monto_usdc":   round(monto_usdc, 6),
        "valor_cop":    int(valor_cop),
        "fecha":        fecha,
    }


# ════════════════════════════════════════════════════════════════════
# FRAMEWORK DE PRUEBAS
# ════════════════════════════════════════════════════════════════════
class ResultadoPrueba:
    def __init__(self, id, nombre, categoria, descripcion):
        self.id          = id
        self.nombre      = nombre
        self.categoria   = categoria
        self.descripcion = descripcion
        self.estado      = None   # PASS / FAIL / ERROR
        self.esperado    = None
        self.obtenido    = None
        self.mensaje     = ""
        self.tiempo_ms   = 0

    def to_dict(self):
        return {
            "id":          self.id,
            "nombre":      self.nombre,
            "categoria":   self.categoria,
            "descripcion": self.descripcion,
            "estado":      self.estado,
            "esperado":    str(self.esperado),
            "obtenido":    str(self.obtenido),
            "mensaje":     self.mensaje,
        }

def ejecutar_caso(caso_fn) -> ResultadoPrueba:
    import time
    r = caso_fn()
    t0 = time.perf_counter()
    try:
        caso_fn._run(r)
    except Exception as e:
        if r.estado is None:
            r.estado  = "ERROR"
            r.mensaje = f"Excepción inesperada: {e}"
    r.tiempo_ms = round((time.perf_counter() - t0) * 1000, 2)
    return r


# ════════════════════════════════════════════════════════════════════
# 12 CASOS DE PRUEBA
# ════════════════════════════════════════════════════════════════════

CASOS = []

# ── GRUPO 1: ERRORES (inputs inválidos) ─────────────────────────────

def caso_01():
    """TC-01: Strike negativo → debe lanzar ValueError"""
    r = ResultadoPrueba("TC-01","Strike negativo","ERROR",
        "calcular_payoff con strike=-5 debe lanzar ValueError")
    r.esperado = "ValueError: Strike no puede ser negativo"
    try:
        calcular_payoff(hdd_realizado=50, strike=-5, tick=100)
        r.estado  = "FAIL"
        r.obtenido = "No lanzó excepción"
        r.mensaje = "La función aceptó un strike negativo sin protestar"
    except ValueError as e:
        r.estado   = "PASS"
        r.obtenido = f"ValueError: {e}"
        r.mensaje  = "Validación correcta"
    return r
CASOS.append(caso_01)

def caso_02():
    """TC-02: Tick negativo → debe lanzar ValueError"""
    r = ResultadoPrueba("TC-02","Tick negativo","ERROR",
        "calcular_payoff con tick=-50 debe lanzar ValueError")
    r.esperado = "ValueError: Tick no puede ser negativo"
    try:
        calcular_payoff(hdd_realizado=50, strike=10, tick=-50)
        r.estado  = "FAIL"
        r.obtenido = "No lanzó excepción"
        r.mensaje = "Tick negativo pasó sin error"
    except ValueError as e:
        r.estado   = "PASS"
        r.obtenido = f"ValueError: {e}"
        r.mensaje  = "Validación correcta"
    return r
CASOS.append(caso_02)

def caso_03():
    """TC-03: Serie HDD vacía → prima debe lanzar ValueError"""
    r = ResultadoPrueba("TC-03","Serie HDD vacía","ERROR",
        "calcular_prima con lista vacía debe lanzar ValueError")
    r.esperado = "ValueError: Serie HDD vacía"
    try:
        calcular_prima(hdd_serie=[], strike=10, tick=100)
        r.estado  = "FAIL"
        r.obtenido = "No lanzó excepción"
        r.mensaje = "Lista vacía pasó sin error"
    except ValueError as e:
        r.estado   = "PASS"
        r.obtenido = f"ValueError: {e}"
        r.mensaje  = "Validación correcta"
    return r
CASOS.append(caso_03)

def caso_04():
    """TC-04: TRM negativa → convertir_cop debe lanzar ValueError"""
    r = ResultadoPrueba("TC-04","TRM negativa","ERROR",
        "convertir_cop con trm=-100 debe lanzar ValueError")
    r.esperado = "ValueError: TRM debe ser positiva"
    try:
        convertir_cop(monto_usd=500, trm=-100)
        r.estado  = "FAIL"
        r.obtenido = "No lanzó excepción"
        r.mensaje = "TRM negativa pasó sin error"
    except ValueError as e:
        r.estado   = "PASS"
        r.obtenido = f"ValueError: {e}"
        r.mensaje  = "Validación correcta"
    return r
CASOS.append(caso_04)

def caso_05():
    """TC-05: Fecha con formato inválido → generar_tx_json debe lanzar ValueError"""
    r = ResultadoPrueba("TC-05","Fecha formato inválido","ERROR",
        "generar_tx_json con fecha='31/03/2024' debe lanzar ValueError")
    r.esperado = "ValueError: Fecha inválida"
    try:
        generar_tx_json("USR001", 500.0, 2_090_000.0, "31/03/2024")
        r.estado  = "FAIL"
        r.obtenido = "No lanzó excepción"
        r.mensaje = "Formato de fecha inválido aceptado"
    except ValueError as e:
        r.estado   = "PASS"
        r.obtenido = f"ValueError: {e}"
        r.mensaje  = "Validación correcta"
    return r
CASOS.append(caso_05)

def caso_06():
    """TC-06: Monto USDC negativo en transacción → debe lanzar ValueError"""
    r = ResultadoPrueba("TC-06","Monto USDC negativo","ERROR",
        "generar_tx_json con monto_usdc=-100 debe lanzar ValueError")
    r.esperado = "ValueError: Monto USDC no puede ser negativo"
    try:
        generar_tx_json("USR001", -100.0, 0.0, "2024-03-31")
        r.estado  = "FAIL"
        r.obtenido = "No lanzó excepción"
        r.mensaje = "Monto negativo aceptado"
    except ValueError as e:
        r.estado   = "PASS"
        r.obtenido = f"ValueError: {e}"
        r.mensaje  = "Validación correcta"
    return r
CASOS.append(caso_06)

# ── GRUPO 2: CASOS LÍMITE ────────────────────────────────────────────

def caso_07():
    """TC-07: Temperatura exactamente igual al umbral → HDD debe ser 0"""
    r = ResultadoPrueba("TC-07","T = umbral exacto","LÍMITE",
        "calcular_hdd(10.0, umbral=10.0) debe retornar 0.0 exacto")
    r.esperado = 0.0
    resultado  = calcular_hdd(temperatura=10.0, umbral=10.0)
    r.obtenido = resultado
    if resultado == 0.0:
        r.estado  = "PASS"
        r.mensaje = "HDD = 0 cuando T = umbral: correcto"
    else:
        r.estado  = "FAIL"
        r.mensaje = f"Esperaba 0.0, obtuvo {resultado}"
    return r
CASOS.append(caso_07)

def caso_08():
    """TC-08: HDD exactamente igual al strike → payoff debe ser 0"""
    r = ResultadoPrueba("TC-08","HDD = strike exacto","LÍMITE",
        "calcular_payoff(hdd=38.0, strike=38.0) debe retornar 0.0")
    STRIKE = 38.0; TICK = 250.0
    r.esperado = 0.0
    resultado  = calcular_payoff(hdd_realizado=STRIKE, strike=STRIKE, tick=TICK)
    r.obtenido = resultado
    if resultado == 0.0:
        r.estado  = "PASS"
        r.mensaje = "Payoff = 0 cuando HDD = strike: correcto (frontera no incluida)"
    else:
        r.estado  = "FAIL"
        r.mensaje = f"Esperaba 0.0, obtuvo {resultado}"
    return r
CASOS.append(caso_08)

def caso_09():
    """TC-09: HDD = strike + 1 épsilon → primer pago mínimo"""
    r = ResultadoPrueba("TC-09","HDD = strike + ε","LÍMITE",
        "Con HDD=38.001, strike=38, tick=250 → payoff = 0.001 × 250 = 0.25")
    STRIKE = 38.0; TICK = 250.0; EPS = 0.001
    r.esperado = round(EPS * TICK, 6)
    resultado  = round(calcular_payoff(STRIKE + EPS, STRIKE, TICK), 6)
    r.obtenido = resultado
    if abs(resultado - r.esperado) < 1e-9:
        r.estado  = "PASS"
        r.mensaje = "Primer pago marginal calculado correctamente"
    else:
        r.estado  = "FAIL"
        r.mensaje = f"Error numérico: esperado {r.esperado}, obtenido {resultado}"
    return r
CASOS.append(caso_09)

def caso_10():
    """TC-10: Strike = 0 → toda la serie HDD activa el contrato"""
    r = ResultadoPrueba("TC-10","Strike = 0","LÍMITE",
        "Con strike=0, payoff = HDD_total × tick para todos los días")
    hdd_serie = [5.0, 3.0, 8.0, 0.0, 2.5]
    TICK = 100.0
    r.esperado = sum(hdd_serie) * TICK  # 18.5 × 100 = 1850
    payoffs    = [calcular_payoff(h, 0.0, TICK) for h in hdd_serie]
    r.obtenido = sum(payoffs)
    if abs(r.obtenido - r.esperado) < 1e-6:
        r.estado  = "PASS"
        r.mensaje = "Strike=0 activa toda la serie correctamente"
    else:
        r.estado  = "FAIL"
        r.mensaje = f"Esperado {r.esperado}, obtenido {r.obtenido}"
    return r
CASOS.append(caso_10)

# ── GRUPO 3: VALIDACIONES DE CÁLCULO ────────────────────────────────

def caso_11():
    """TC-11: Prima con carga=0 debe ser igual al payoff promedio puro"""
    r = ResultadoPrueba("TC-11","Prima pura sin carga","CÁLCULO",
        "calcular_prima(carga=0) debe ser igual a mean(payoffs) sin margen")
    hdd_serie = [10.0, 45.0, 60.0, 25.0, 0.0,
                 55.0, 30.0, 5.0,  70.0, 15.0]
    STRIKE = 20.0; TICK = 300.0
    payoffs_manuales = [max(0, h - STRIKE) * TICK for h in hdd_serie]
    prima_esperada   = np.mean(payoffs_manuales)   # carga = 0

    prima_obtenida   = calcular_prima(hdd_serie, STRIKE, TICK, carga=0.0)
    r.esperado = round(prima_esperada, 6)
    r.obtenido = round(prima_obtenida, 6)

    if abs(prima_obtenida - prima_esperada) < 1e-4:
        r.estado  = "PASS"
        r.mensaje = f"Prima pura = USD {prima_esperada:.2f} — coherente"
    else:
        r.estado  = "FAIL"
        r.mensaje = f"Discrepancia: esperado {prima_esperada:.6f}, obtenido {prima_obtenida:.6f}"
    return r
CASOS.append(caso_11)

def caso_12():
    """TC-12: Prima con carga=20% debe ser exactamente 1.20 × prima pura"""
    r = ResultadoPrueba("TC-12","Prima comercial = 1.2 × prima pura","CÁLCULO",
        "Prima con carga=0.20 debe ser 20% mayor que prima con carga=0")
    hdd_serie = [8.0, 42.0, 15.0, 60.0, 0.0,
                 35.0, 55.0, 20.0, 80.0, 10.0]
    STRIKE = 15.0; TICK = 200.0

    prima_pura    = calcular_prima(hdd_serie, STRIKE, TICK, carga=0.00)
    prima_comerc  = calcular_prima(hdd_serie, STRIKE, TICK, carga=0.20)
    ratio         = prima_comerc / prima_pura if prima_pura > 0 else 0

    r.esperado = f"ratio = 1.20  (prima_comerc = USD {prima_pura*1.2:.2f})"
    r.obtenido = f"ratio = {ratio:.6f}  (prima_comerc = USD {prima_comerc:.2f})"

    if abs(ratio - 1.20) < 1e-9:
        r.estado  = "PASS"
        r.mensaje = f"Carga del 20% aplicada correctamente. Prima pura={prima_pura:.2f} → comercial={prima_comerc:.2f}"
    else:
        r.estado  = "FAIL"
        r.mensaje = f"Ratio esperado 1.20, obtenido {ratio:.8f}"
    return r
CASOS.append(caso_12)


# ════════════════════════════════════════════════════════════════════
# EJECUTAR SUITE Y REPORTAR
# ════════════════════════════════════════════════════════════════════
def correr_suite():
    import time
    resultados = []
    for caso_fn in CASOS:
        t0 = time.perf_counter()
        try:
            r = caso_fn()
        except Exception as e:
            r = ResultadoPrueba(
                "??", caso_fn.__name__, "ERROR_INTERNO",
                f"Error al construir el caso: {e}"
            )
            r.estado   = "ERROR"
            r.esperado = "—"
            r.obtenido = str(e)
            r.mensaje  = traceback.format_exc()
        r.tiempo_ms = round((time.perf_counter() - t0) * 1000, 3)
        resultados.append(r)
    return resultados


if __name__ == "__main__":
    resultados = correr_suite()

    # ── Imprimir tabla ───────────────────────────────────────────────
    PASS  = sum(1 for r in resultados if r.estado == "PASS")
    FAIL  = sum(1 for r in resultados if r.estado == "FAIL")
    ERROR = sum(1 for r in resultados if r.estado == "ERROR")

    ICONO = {"PASS":"✅","FAIL":"❌","ERROR":"🔥"}
    CAT_COLOR = {"ERROR":"[ERROR]","LÍMITE":"[LÍMITE]","CÁLCULO":"[CÁLCULO]"}

    print("=" * 72)
    print("AGRO-RISK PRO — SUITE DE PRUEBAS  ·  12 CASOS")
    print("=" * 72)
    print(f"{'ID':<8} {'Categoría':<12} {'Estado':<7} {'Tiempo':>7}  Nombre")
    print("─" * 72)

    for r in resultados:
        icono = ICONO.get(r.estado, "?")
        print(f"{r.id:<8} {r.categoria:<12} {icono} {r.estado:<5} "
              f"{r.tiempo_ms:>6.2f}ms  {r.nombre}")

    print("─" * 72)
    print(f"TOTAL: {len(resultados)} casos  |  "
          f"✅ {PASS} PASS  |  ❌ {FAIL} FAIL  |  🔥 {ERROR} ERROR")
    print(f"Cobertura: {PASS/len(resultados)*100:.0f}%")

    # ── Detalle de fallos ────────────────────────────────────────────
    fallos = [r for r in resultados if r.estado != "PASS"]
    if fallos:
        print(f"\n{'='*72}")
        print("DETALLE DE FALLOS:")
        for r in fallos:
            print(f"\n  {r.id} — {r.nombre}")
            print(f"  Esperado: {r.esperado}")
            print(f"  Obtenido: {r.obtenido}")
            print(f"  Mensaje:  {r.mensaje}")
    else:
        print("\n🎉 Todos los casos pasaron.")

    # ── Exportar JSON ────────────────────────────────────────────────
    output = {
        "suite":      "Agro-Risk Pro Test Suite",
        "version":    "1.0",
        "fecha":      pd.Timestamp.now().isoformat(),
        "resumen":    {"total": len(resultados), "pass": PASS,
                       "fail": FAIL, "error": ERROR,
                       "cobertura_pct": round(PASS/len(resultados)*100,1)},
        "casos":      [r.to_dict() for r in resultados],
    }
    with open("outputs/13_test_results.json","w",encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 13_test_results.json exportado")


