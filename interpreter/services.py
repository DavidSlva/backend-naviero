from collection_manager.models import Aduana, Pais, Puerto, TipoCarga
from interpreter.models import AgenciaTransporte, Registro
enero_path = 'C:/Users/David/Documents/Github/Proyecto Semestral Grafos y Algoritmos/Trazabilidad Exportaciones/downloads/Importaciones abril 2024.txt'

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from datetime import datetime

def procesar_registros_lote(filas, columnas, puertos, aduanas, tiposCargas, agenciasTransporte, registros_existentes, pbar):
    registros = []
    nuevos_ruts = {}

    for fila in filas:
        registro = {columnas[i]: fila[i] for i in range(len(fila))}

        # Verificar si el número de registro ya existe en los registros cargados previamente
        if registro['NUMENCRIPTADO'] in registros_existentes:
            print(f"Registro {registro['NUMENCRIPTADO']} ya existe. Saltando...")
            continue

        # Convertir la fecha al formato correcto (YYYY-MM-DD)
        try:
            fecha_aceptacion = datetime.strptime(registro['FECACEP'], '%d%m%Y').strftime('%Y-%m-%d')
        except ValueError as e:
            print(f"Error de formato en la fecha {registro['FECACEP']}: {e}")
            continue  # O puedes optar por manejar esto de otra forma (p. ej., asignar `None` a la fecha)

        # Verificar si la agencia de transporte ya existe o agregarla al dict de nuevos_ruts
        if registro['NUMRUTCIA'] not in agenciasTransporte and registro['NUMRUTCIA'] not in nuevos_ruts:
            nuevos_ruts[registro['NUMRUTCIA']] = AgenciaTransporte(
                nombre=registro['GNOM_CIA_T'],
                rut=registro['NUMRUTCIA'],
                dig_v=registro['DIGVERCIA']
            )
            print(f"Agregando nueva Agencia de Transporte: {registro['GNOM_CIA_T']}")

        # Crear el nuevo registro de importación
        nuevo_registro = Registro(
            puerto_embarque=puertos.get(int(registro['PTO_EMB']), None),
            puerto_desembarque=puertos.get(int(registro['PTO_DESEM']), None),
            num_registro=registro['NUMENCRIPTADO'],
            fecha_aceptacion=fecha_aceptacion,  # Usamos la fecha convertida
            aduana=aduanas.get(int(registro['ADU']), None),
            tipo_carga=tiposCargas.get(registro['TPO_CARGA'], None),
            nro_manifiesto=registro['NUM_MANIF'],
            agencia_transporte=agenciasTransporte.get(registro['NUMRUTCIA'], None) or nuevos_ruts.get(registro['NUMRUTCIA'], None)
        )
        registros.append(nuevo_registro)

        # Actualizar la barra de progreso después de procesar cada fila
        pbar.update(1)

    # Guardar nuevas agencias de transporte
    if nuevos_ruts:
        AgenciaTransporte.objects.bulk_create(nuevos_ruts.values())
        agenciasTransporte.update(nuevos_ruts)  # Actualizamos el diccionario de agencias cargadas

    # Inserción masiva de registros
    if registros:
        Registro.objects.bulk_create(registros, batch_size=1000)  # Se usa un tamaño de lote para mejorar el rendimiento

def cargar_registros_importacion():
    # Nombres de las columnas
    columnas = [
        'NUMENCRIPTADO', 'TIPO_DOCTO', 'ADU', 'FORM', 'FECVENCI', 'CODCOMUN', 'NUM_UNICO_IMPORTADOR', 'CODPAISCON',
        'DESDIRALM', 'CODCOMRS', 'ADUCTROL', 'NUMPLAZO', 'INDPARCIAL', 'NUMHOJINS', 'TOTINSUM', 'CODALMA', 'NUM_RS',
        'FEC_RS', 'ADUA_RS', 'NUMHOJANE', 'NUM_SEC', 'PA_ORIG', 'PA_ADQ', 'VIA_TRAN', 'TRANSB', 'PTO_EMB', 'PTO_DESEM',
        'TPO_CARGA', 'ALMACEN', 'FEC_ALMAC', 'FECRETIRO', 'NU_REGR', 'ANO_REG', 'CODVISBUEN', 'NUMREGLA', 'NUMANORES',
        'CODULTVB', 'PAGO_GRAV', 'FECTRA', 'FECACEP', 'GNOM_CIA_T', 'CODPAISCIA', 'NUMRUTCIA', 'DIGVERCIA', 'NUM_MANIF',
        'NUM_MANIF1', 'NUM_MANIF2', 'FEC_MANIF', 'NUM_CONOC', 'FEC_CONOC', 'NOMEMISOR', 'NUMRUTEMI', 'DIGVEREMI',
        'GREG_IMP', 'REG_IMP', 'BCO_COM', 'CODORDIV', 'FORM_PAGO', 'NUMDIAS', 'VALEXFAB', 'MONEDA', 'MONGASFOB',
        'CL_COMPRA', 'TOT_ITEMS', 'FOB', 'TOT_HOJAS', 'COD_FLE', 'FLETE', 'TOT_BULTOS', 'COD_SEG', 'SEGURO', 'TOT_PESO',
        'CIF', 'NUM_AUT', 'FEC_AUT', 'GBCOCEN', 'ID_BULTOS', 'TPO_BUL1', 'CANT_BUL1', 'TPO_BUL2', 'CANT_BUL2',
        'TPO_BUL3', 'CANT_BUL3', 'TPO_BUL4', 'CANT_BUL4', 'TPO_BUL5', 'CANT_BUL5', 'TPO_BUL6', 'CANT_BUL6', 'TPO_BUL7',
        'CANT_BUL7', 'TPO_BUL8', 'CANT_BUL8', 'CTA_OTRO', 'MON_OTRO', 'CTA_OTR1', 'MON_OTR1', 'CTA_OTR2', 'MON_OTR2',
        'CTA_OTR3', 'MON_OTR3', 'CTA_OTR4', 'MON_OTR4', 'CTA_OTR5', 'MON_OTR5', 'CTA_OTR6', 'MON_OTR6', 'CTA_OTR7',
        'MON_OTR7', 'MON_178', 'MON_191', 'FEC_501', 'VAL_601', 'FEC_502', 'VAL_602', 'FEC_503', 'VAL_603', 'FEC_504',
        'VAL_604', 'FEC_505', 'VAL_605', 'FEC_506', 'VAL_606', 'FEC_507', 'VAL_607', 'TASA', 'NCUOTAS', 'ADU_DI', 
        'NUM_DI', 'FEC_DI', 'MON_699', 'MON_199', 'NUMITEM', 'DNOMBRE', 'DMARCA', 'DVARIEDAD', 'DOTRO1', 'DOTRO2', 
        'ATR-5', 'ATR-6', 'SAJU-ITEM', 'AJU-ITEM', 'CANT-MERC', 'MERMAS', 'MEDIDA', 'PRE-UNIT', 'ARANC-ALA', 'NUMCOR', 
        'NUMACU', 'CODOBS1', 'DESOBS1', 'CODOBS2', 'DESOBS2', 'CODOBS3', 'DESOBS3', 'CODOBS4', 'DESOBS4', 'ARANC-NAC', 
        'CIF-ITEM', 'ADVAL-ALA', 'ADVAL', 'VALAD', 'OTRO1', 'CTA1', 'SIGVAL1', 'VAL1', 'OTRO2', 'CTA2', 'SIGVAL2', 'VAL2', 
        'OTRO3', 'CTA3', 'SIGVAL3', 'VAL3', 'OTRO4', 'CTA4', 'SIGVAL4', 'VAL4'
    ]

    # Cargar objetos en caché para evitar múltiples consultas
    puertos = {puerto.codigo: puerto for puerto in Puerto.objects.all()}
    aduanas = {aduana.codigo: aduana for aduana in Aduana.objects.all()}
    tiposCargas = {tipoCarga.codigo: tipoCarga for tipoCarga in TipoCarga.objects.all()}
    agenciasTransporte = {agencia.rut: agencia for agencia in AgenciaTransporte.objects.all()}

    # Pre-cargar todos los registros existentes en memoria
    registros_existentes = set(Registro.objects.values_list('num_registro', flat=True))

    with open(enero_path, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.reader(csvfile, delimiter=';'))
        total_filas = len(reader)

        # Dividir los registros en lotes para los hilos
        num_hilos = os.cpu_count()  # Usar tantos hilos como núcleos de CPU disponibles
        tamaño_lote = total_filas // num_hilos

        # Crear la barra de progreso
        with tqdm(total=total_filas, desc="Procesando registros") as pbar:
            # Crear el pool de hilos
            with ThreadPoolExecutor(max_workers=num_hilos) as executor:
                futuros = []
                for i in range(0, total_filas, tamaño_lote):
                    lote_filas = reader[i:i + tamaño_lote]
                    futuros.append(executor.submit(procesar_registros_lote, lote_filas, columnas, puertos, aduanas, tiposCargas, agenciasTransporte, registros_existentes, pbar))
                
                # Esperar a que todos los hilos terminen
                for futuro in as_completed(futuros):
                    try:
                        resultado = futuro.result()
                    except Exception as exc:
                        print(f"Generó una excepción: {exc}")
