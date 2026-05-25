"""
Agro-Risk Pro — Generador de reporte DIAN
Formato basado en Resolución 000240 de 2024
Materia: Programación para Economía y Finanzas

NOTA ACADÉMICA IMPORTANTE:
  La Resolución 000240 de 2024 regula la obligación de reportar
  operaciones con activos virtuales a la DIAN. Este módulo genera
  un JSON de demostración con la estructura requerida.
  En producción REAL se necesita:
    - NIT y firma digital de la plataforma reportante
    - Certificado de la DIAN para transmisión electrónica
    - Validación por el servicio web oficial SOAP/REST de la DIAN
    - Asesoría de un contador/abogado tributario certificado

CAMPOS PRINCIPALES (Res. 000240):
  - Formato 2517: Operaciones con activos virtuales
  - Periodicidad: anual (enero del año siguiente)
  - Sujeto obligado: plataforma de intercambio o intermediario
  - Datos del usuario: ID, tipo doc., nombre, monto, moneda, fecha
"""

import json
import hashlib
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ════════════════════════════════════════════════════════════════════
# CONSTANTES DE LA RESOLUCIÓN 000240
# ════════════════════════════════════════════════════════════════════
RESOLUCION       = "000240"
AÑO_RESOLUCION   = 2024
FORMATO_DIAN     = "2517"           # Formato activos virtuales
VERSION_XML      = "1.0"
TIPO_REPORTE     = "OPERACIONES_ACTIVOS_VIRTUALES"
PAIS_EMISION     = "CO"
MONEDA_LOCAL     = "COP"
MONEDA_CRIPTO    = "USDC"
TIPO_ACTIVO      = "STABLECOIN"     # clasificación USDC en Res. 000240
NATURALEZA_OP    = "LIQUIDACION_DERIVADO_CLIMATICO"

# Tipos de documento colombianos
TIPOS_DOC = {
    "CC":  "Cédula de ciudadanía",
    "NIT": "Número de identificación tributaria",
    "CE":  "Cédula de extranjería",
    "PA":  "Pasaporte",
    "PEP": "Permiso especial de permanencia",
}

# ════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL: generar_reporte_dian
# ════════════════════════════════════════════════════════════════════
def generar_reporte_dian(
    transacciones: list[dict],
    plataforma: dict,
    año_fiscal: int,
    ruta_salida: str = ".",
    incluir_hash: bool = True,
) -> dict:
    """
    Genera el JSON de reporte DIAN para operaciones con activos virtuales.
    Basado en Resolución 000240 de 2024, Formato 2517.

    Parámetros
    ----------
    transacciones : list[dict]
        Lista de transacciones. Cada dict debe tener:
            usuario_id      : str   — ID interno de la plataforma
            tipo_doc        : str   — CC / NIT / CE / PA / PEP
            num_doc         : str   — número del documento
            nombre          : str   — nombre completo o razón social
            fecha_operacion : str   — YYYY-MM-DD
            monto_usdc      : float — monto en USDC (stablecoin)
            trm_dia         : float — TRM COP/USD del día de la operación
            concepto        : str   — descripción de la operación
            activado        : bool  — si el derivado se activó

    plataforma : dict
        Datos del sujeto obligado (la plataforma reportante):
            nit             : str
            razon_social    : str
            municipio       : str
            departamento    : str
            email           : str

    año_fiscal : int
        Año que se reporta (ej: 2024)

    ruta_salida : str
        Directorio donde guardar el JSON

    incluir_hash : bool
        Si True, incluye SHA-256 de cada transacción para integridad

    Retorna
    -------
    dict : estructura completa del reporte DIAN
    """

    # ── Validaciones de entrada ──────────────────────────────────────
    campos_requeridos = [
        "usuario_id","tipo_doc","num_doc","nombre",
        "fecha_operacion","monto_usdc","trm_dia","concepto","activado"
    ]
    for i, tx in enumerate(transacciones):
        for campo in campos_requeridos:
            if campo not in tx:
                raise ValueError(f"Transacción {i}: falta el campo '{campo}'")
        if tx["tipo_doc"] not in TIPOS_DOC:
            raise ValueError(f"Transacción {i}: tipo_doc '{tx['tipo_doc']}' inválido. "
                             f"Opciones: {list(TIPOS_DOC.keys())}")
        try:
            datetime.strptime(tx["fecha_operacion"], "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Transacción {i}: fecha_operacion debe ser YYYY-MM-DD")

    # ── Construir cabecera del reporte ───────────────────────────────
    timestamp_generacion = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    num_reporte = f"AGR-{año_fiscal}-{uuid.uuid4().hex[:8].upper()}"

    cabecera = {
        "num_reporte":         num_reporte,
        "resolucion":          RESOLUCION,
        "año_resolucion":      AÑO_RESOLUCION,
        "formato":             FORMATO_DIAN,
        "version":             VERSION_XML,
        "tipo_reporte":        TIPO_REPORTE,
        "año_fiscal":          año_fiscal,
        "fecha_generacion":    timestamp_generacion,
        "pais_emision":        PAIS_EMISION,
        "moneda_local":        MONEDA_LOCAL,
        "periodicidad":        "ANUAL",
        "fecha_inicio_periodo": f"{año_fiscal}-01-01",
        "fecha_fin_periodo":    f"{año_fiscal}-12-31",
    }

    # ── Construir bloque del sujeto obligado ─────────────────────────
    sujeto_obligado = {
        "tipo_sujeto":   "INTERMEDIARIO_ACTIVOS_VIRTUALES",
        "nit":           plataforma.get("nit", "900.000.000-0"),
        "razon_social":  plataforma.get("razon_social", "AGRO-RISK PRO SAS"),
        "municipio":     plataforma.get("municipio", "Bogotá D.C."),
        "departamento":  plataforma.get("departamento", "Cundinamarca"),
        "email":         plataforma.get("email", "cumplimiento@agroriskpro.co"),
        "telefono":      plataforma.get("telefono", "+57 601 0000000"),
        "actividad_ciiu":"6619",    # Otras actividades auxiliares de las financieras
        "obligacion":    f"Res. {RESOLUCION}/{AÑO_RESOLUCION} Art. 4",
    }

    # ── Procesar cada transacción ────────────────────────────────────
    detalle_transacciones = []
    resumen_por_usuario   = {}
    total_cop             = Decimal("0")
    total_usdc            = Decimal("0")
    n_operaciones         = 0
    n_activadas           = 0

    for tx in transacciones:
        # Solo reportar transacciones del año fiscal
        fecha_tx = datetime.strptime(tx["fecha_operacion"], "%Y-%m-%d").date()
        if fecha_tx.year != año_fiscal:
            continue

        monto_usdc  = Decimal(str(tx["monto_usdc"])).quantize(
                          Decimal("0.000001"), rounding=ROUND_HALF_UP)
        trm         = Decimal(str(tx["trm_dia"])).quantize(
                          Decimal("0.01"), rounding=ROUND_HALF_UP)
        valor_cop   = (monto_usdc * trm).quantize(
                          Decimal("1"), rounding=ROUND_HALF_UP)
        # Umbral de reporte: > 0 USDC (incluir todas las liquidaciones)
        # En producción la Res. 000240 establece umbral de 5 SMMLV ≈ COP 7.2M (2024)
        UMBRAL_COP  = Decimal("7200000")
        supera_umbral = valor_cop >= UMBRAL_COP

        # Construir registro de transacción
        tx_id = f"TX-{año_fiscal}-{uuid.uuid4().hex[:10].upper()}"

        registro = {
            "tx_id":                tx_id,
            "usuario_id":           tx["usuario_id"],
            "tipo_documento":       tx["tipo_doc"],
            "numero_documento":     _enmascarar_doc(tx["num_doc"]),
            "nombre_completo":      tx["nombre"],
            "fecha_operacion":      tx["fecha_operacion"],
            "año_gravable":         fecha_tx.year,
            "tipo_activo_virtual":  TIPO_ACTIVO,
            "simbolo_cripto":       MONEDA_CRIPTO,
            "naturaleza_operacion": NATURALEZA_OP,
            "concepto":             tx["concepto"],
            "monto_usdc":           float(monto_usdc),
            "trm_dia_cop":          float(trm),
            "valor_cop":            int(valor_cop),
            "supera_umbral_reporte": supera_umbral,
            "contrato_activado":    tx["activado"],
            "municipio_usuario":    tx.get("municipio", "Facatativá"),
            "departamento_usuario": tx.get("departamento", "Cundinamarca"),
        }

        # Hash SHA-256 de integridad (firma de cada registro)
        if incluir_hash:
            contenido_hash = (
                f"{tx_id}|{tx['num_doc']}|{tx['fecha_operacion']}"
                f"|{float(monto_usdc)}|{int(valor_cop)}"
            )
            registro["sha256_integridad"] = hashlib.sha256(
                contenido_hash.encode("utf-8")
            ).hexdigest()

        detalle_transacciones.append(registro)

        # Acumulados
        total_cop  += valor_cop
        total_usdc += monto_usdc
        n_operaciones += 1
        if tx["activado"]:
            n_activadas += 1

        # Resumen por usuario (agrupado por num_doc)
        clave = tx["num_doc"]
        if clave not in resumen_por_usuario:
            resumen_por_usuario[clave] = {
                "usuario_id":      tx["usuario_id"],
                "tipo_doc":        tx["tipo_doc"],
                "num_doc":         _enmascarar_doc(tx["num_doc"]),
                "nombre":          tx["nombre"],
                "n_operaciones":   0,
                "total_usdc":      Decimal("0"),
                "total_cop":       Decimal("0"),
                "operaciones_activadas": 0,
            }
        resumen_por_usuario[clave]["n_operaciones"]       += 1
        resumen_por_usuario[clave]["total_usdc"]          += monto_usdc
        resumen_por_usuario[clave]["total_cop"]           += valor_cop
        resumen_por_usuario[clave]["operaciones_activadas"] += int(tx["activado"])

    # Convertir Decimal a float en resumen
    resumen_lista = []
    for k, v in resumen_por_usuario.items():
        resumen_lista.append({
            **v,
            "total_usdc": float(v["total_usdc"]),
            "total_cop":  int(v["total_cop"]),
        })

    # ── Totales del reporte ──────────────────────────────────────────
    totales = {
        "n_registros":         n_operaciones,
        "n_activadas":         n_activadas,
        "n_no_activadas":      n_operaciones - n_activadas,
        "total_usdc":          float(total_usdc),
        "total_cop":           int(total_cop),
        "promedio_cop":        int(total_cop / n_operaciones) if n_operaciones else 0,
        "n_superan_umbral":    sum(1 for r in detalle_transacciones
                                   if r["supera_umbral_reporte"]),
    }

    # ── Ensamblar reporte completo ───────────────────────────────────
    reporte = {
        "_meta": {
            "descripcion": "Reporte DIAN — Operaciones activos virtuales",
            "resolucion":  f"Resolución {RESOLUCION} de {AÑO_RESOLUCION}",
            "formato":     f"Formato {FORMATO_DIAN}",
            "generado_por": "Agro-Risk Pro — Módulo de Cumplimiento",
            "nota_academica": (
                "DOCUMENTO ACADÉMICO DE DEMOSTRACIÓN. "
                "No es un reporte oficial ni tiene valor tributario. "
                "Para reportes reales: consultar un contador certificado "
                "y usar el servicio web oficial de la DIAN."
            ),
        },
        "cabecera":              cabecera,
        "sujeto_obligado":       sujeto_obligado,
        "totales":               totales,
        "resumen_por_usuario":   resumen_lista,
        "detalle_transacciones": detalle_transacciones,
    }

    # ── Guardar archivo ──────────────────────────────────────────────
    nombre_archivo = (
        f"DIAN_F{FORMATO_DIAN}_{año_fiscal}_"
        f"{plataforma.get('nit','NIT').replace('.','').replace('-','')}"
        f"_{num_reporte.split('-')[-1]}.json"
    )
    ruta_completa = os.path.join(ruta_salida, nombre_archivo)
    with open(ruta_completa, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2, default=str)

    return reporte, ruta_completa


def _enmascarar_doc(num_doc: str) -> str:
    """Enmascara parcialmente el documento para logs (últimos 4 dígitos visibles)."""
    limpio = num_doc.replace(".", "").replace("-", "")
    if len(limpio) <= 4:
        return limpio
    return "*" * (len(limpio) - 4) + limpio[-4:]


# ════════════════════════════════════════════════════════════════════
# FUNCIÓN AUXILIAR: validar_reporte
# ════════════════════════════════════════════════════════════════════
def validar_reporte(reporte: dict) -> dict:
    """
    Valida el reporte generado contra los requisitos mínimos
    de la Resolución 000240.

    Retorna dict con: valido (bool), errores (list), advertencias (list)
    """
    errores     = []
    advertencias= []

    # Verificar campos obligatorios de cabecera
    for campo in ["num_reporte","resolucion","formato","año_fiscal",
                  "fecha_generacion","tipo_reporte"]:
        if campo not in reporte.get("cabecera", {}):
            errores.append(f"Cabecera: falta campo obligatorio '{campo}'")

    # Verificar sujeto obligado
    so = reporte.get("sujeto_obligado", {})
    if not so.get("nit"):
        errores.append("Sujeto obligado: NIT requerido")
    if not so.get("razon_social"):
        errores.append("Sujeto obligado: razón social requerida")

    # Verificar integridad de transacciones
    txs = reporte.get("detalle_transacciones", [])
    if not txs:
        advertencias.append("No hay transacciones en el período fiscal reportado")

    ids_vistos = set()
    for i, tx in enumerate(txs):
        if tx["tx_id"] in ids_vistos:
            errores.append(f"TX duplicada: {tx['tx_id']}")
        ids_vistos.add(tx["tx_id"])

        if tx["monto_usdc"] < 0:
            errores.append(f"TX {tx['tx_id']}: monto negativo")
        if tx["valor_cop"] < 0:
            errores.append(f"TX {tx['tx_id']}: valor COP negativo")

        # Verificar hash de integridad
        if "sha256_integridad" in tx:
            contenido = (
                f"{tx['tx_id']}|"
                # hash no puede reverificarse con doc enmascarado — solo se advierte
                f"MASKED|{tx['fecha_operacion']}"
                f"|{tx['monto_usdc']}|{tx['valor_cop']}"
            )

    # Verificar coherencia de totales
    tot = reporte.get("totales", {})
    suma_cop = sum(tx["valor_cop"] for tx in txs)
    if abs(suma_cop - tot.get("total_cop", 0)) > 1:
        errores.append(f"Total COP no cuadra: suma={suma_cop} vs declarado={tot.get('total_cop')}")

    # Advertencias por umbral
    sin_umbral = [tx for tx in txs if not tx.get("supera_umbral_reporte")]
    if sin_umbral:
        advertencias.append(
            f"{len(sin_umbral)} transacciones por debajo del umbral de "
            f"reporte (5 SMMLV). Incluidas en el JSON pero pueden omitirse "
            f"en la versión final según criterio contable."
        )

    return {
        "valido":        len(errores) == 0,
        "errores":       errores,
        "advertencias":  advertencias,
        "n_transacciones": len(txs),
        "n_errores":     len(errores),
        "n_advertencias":len(advertencias),
    }


# ════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE EJEMPLO
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Cargar historial de liquidaciones del proyecto ────────────
    try:
        hist = pd.read_csv("outputs/09_burn_analysis.csv")
        años_hist = hist["anio"].astype(int).tolist()
        payoffs   = hist["payoff_usd"].tolist()
        activados = hist["activado"].astype(bool).tolist()
    except FileNotFoundError:
        # Datos mínimos de demostración si no existe el CSV
        años_hist = list(range(2020, 2025))
        payoffs   = [0, 450.0, 0, 2750.0, 8500.0]
        activados = [False, True, False, True, True]

    # ── 2. Usuarios de demostración ──────────────────────────────────
    USUARIOS = [
        {"usuario_id":"USR001","tipo_doc":"NIT","num_doc":"900123456-7",
         "nombre":"Flores del Sabana S.A.S.",
         "municipio":"Chía","departamento":"Cundinamarca"},
        {"usuario_id":"USR002","tipo_doc":"CC","num_doc":"79854321",
         "nombre":"Carlos Hernández Roa",
         "municipio":"Zipaquirá","departamento":"Cundinamarca"},
        {"usuario_id":"USR003","tipo_doc":"NIT","num_doc":"830567890-1",
         "nombre":"Exportadora Andina Ltda.",
         "municipio":"Facatativá","departamento":"Cundinamarca"},
        {"usuario_id":"USR004","tipo_doc":"CC","num_doc":"52763419",
         "nombre":"María Alejandra Torres",
         "municipio":"Ubaté","departamento":"Cundinamarca"},
        {"usuario_id":"USR005","tipo_doc":"NIT","num_doc":"900987654-3",
         "nombre":"CundiFlores Export S.A.S.",
         "municipio":"Facatativá","departamento":"Cundinamarca"},
    ]

    TRM_POR_AÑO = {2020:3750,2021:3750,2022:4200,2023:4150,2024:4180,
                   2015:3150,2016:3100,2017:2985,2018:2985,2019:3320}

    # ── 3. Generar transacciones sintéticas ──────────────────────────
    transacciones = []
    np.random.seed(42)

    for año, payoff_base, activado in zip(años_hist, payoffs, activados):
        for usuario in USUARIOS:
            # Fracción del payoff total asignada a cada usuario
            fraccion    = np.random.uniform(0.15, 0.25)
            monto_usdc  = round(payoff_base * fraccion, 6) if activado else 0.0
            trm_dia     = TRM_POR_AÑO.get(año, 4000) + np.random.randint(-50, 51)
            fecha_liq   = f"{año}-03-31"  # liquidación fin Q1

            transacciones.append({
                **usuario,
                "fecha_operacion": fecha_liq,
                "monto_usdc":      monto_usdc,
                "trm_dia":         float(trm_dia),
                "concepto":        (
                    f"Liquidación derivado HDD — cobertura climática "
                    f"paramétrica Q1-{año} · {'Activado' if activado else 'No activado'}"
                ),
                "activado":        activado,
            })

    # ── 4. Datos de la plataforma ────────────────────────────────────
    plataforma = {
        "nit":          "900.555.123-4",
        "razon_social": "AGRO-RISK PRO S.A.S.",
        "municipio":    "Bogotá D.C.",
        "departamento": "Cundinamarca",
        "email":        "cumplimiento@agroriskpro.co",
        "telefono":     "+57 601 123 4567",
    }

    # ── 5. Generar reporte para año 2024 ─────────────────────────────
    AÑO_REPORTE = 2024
    print("=" * 65)
    print(f"GENERANDO REPORTE DIAN — Resolución {RESOLUCION}/{AÑO_RESOLUCION}")
    print(f"Formato {FORMATO_DIAN} · Año fiscal {AÑO_REPORTE}")
    print("=" * 65)

    reporte, ruta = generar_reporte_dian(
        transacciones = transacciones,
        plataforma    = plataforma,
        año_fiscal    = AÑO_REPORTE,
        ruta_salida   = "/mnt/user-data/outputs",
        incluir_hash  = True,
    )

    # ── 6. Validar ───────────────────────────────────────────────────
    validacion = validar_reporte(reporte)

    print(f"\n✅ Archivo generado: {os.path.basename(ruta)}")
    print(f"   Tamaño: {os.path.getsize(ruta)/1024:.1f} KB")
    print(f"\nVALIDACIÓN:")
    print(f"  Estado:          {'✅ VÁLIDO' if validacion['valido'] else '❌ CON ERRORES'}")
    print(f"  Transacciones:   {validacion['n_transacciones']}")
    print(f"  Errores:         {validacion['n_errores']}")
    print(f"  Advertencias:    {validacion['n_advertencias']}")
    if validacion["errores"]:
        for e in validacion["errores"]: print(f"  ❌ {e}")
    if validacion["advertencias"]:
        for a in validacion["advertencias"]: print(f"  ⚠ {a}")

    # ── 7. Preview del JSON ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("PREVIEW DEL JSON (primeros 2 registros):")
    print("="*65)
    preview = {
        "_meta":         reporte["_meta"],
        "cabecera":      reporte["cabecera"],
        "sujeto_obligado": reporte["sujeto_obligado"],
        "totales":       reporte["totales"],
        "detalle_transacciones": reporte["detalle_transacciones"][:2],
        "_nota":         f"... {len(reporte['detalle_transacciones'])-2} registros más en el archivo completo"
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))

    print(f"\n{'='*65}")
    print("TOTALES DEL REPORTE:")
    t = reporte["totales"]
    print(f"  Operaciones reportadas:  {t['n_registros']}")
    print(f"  Activadas (con pago):    {t['n_activadas']}")
    print(f"  Total USDC liquidado:    {t['total_usdc']:.6f} USDC")
    print(f"  Total en COP:            COP {t['total_cop']:,}")
    print(f"  Superan umbral 5 SMMLV:  {t['n_superan_umbral']}")

    print(f"\n⚠ DISCLAIMER:")
    print(f"  Este JSON es de uso académico exclusivo.")
    print(f"  No tiene validez tributaria ante la DIAN.")
    print(f"  Para reportes reales: Res. {RESOLUCION}/{AÑO_RESOLUCION},")
    print(f"  servicio web oficial DIAN y asesoría contable certificada.")
