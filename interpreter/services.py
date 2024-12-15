import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
from backend import settings
from collection_manager.models import Aduana, Pais, Puerto, TipoCarga
from interpreter.models import AgenciaTransporte, Registro
import pandas as pd

# Configuración del logger de errores
logging.basicConfig(
    filename='error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PARQUET_PATH = r"C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet"

def get_dataframe():
    return pd.read_parquet(PARQUET_PATH, engine='pyarrow')

def get_importaciones_file_path():
    archivos = os.listdir(settings.DOWLOADS_PATH)
    output = []
    for archivo in archivos:
        name = archivo.lower()
        if 'importaciones' in name and '.txt' in name:
            output.append(os.path.join(settings.DOWLOADS_PATH, archivo))
    return output

def procesar_registros_lote(filas, columnas, puertos, aduanas, tiposCargas, agenciasTransporte, registros_existentes, pbar, file_path, thread_id):
    registros = []
    nuevos_ruts = {}
    total_in_thread = len(filas)
    procesadas = 0
    ultimo_porcentaje = -1

    for fila in filas:
        procesadas += 1
        # Calcular progreso del hilo
        porcentaje = int((procesadas / total_in_thread) * 100)
        if porcentaje != ultimo_porcentaje:
            print(f"[{file_path}][Thread-{thread_id}] Progreso: {porcentaje}%")
            ultimo_porcentaje = porcentaje

        registro = {columnas[i]: fila[i] for i in range(len(fila))}
        if registro['NUMENCRIPTADO'] in registros_existentes:
            # Registro ya existe, omitir sin imprimir
            pbar.update(1)
            continue

        # Convertir fecha
        try:
            fecha_aceptacion = datetime.strptime(registro['FECACEP'], '%d%m%Y').strftime('%Y-%m-%d')
        except ValueError as e:
            # Loggear error de formato de fecha
            logging.error(f"Error de formato en la fecha {registro['FECACEP']} del archivo {file_path}. Detalle: {e}", exc_info=True)
            pbar.update(1)
            continue

        # Verificar si la agencia existe o se debe crear
        if registro['NUMRUTCIA'] not in agenciasTransporte and registro['NUMRUTCIA'] not in nuevos_ruts:
            nuevos_ruts[registro['NUMRUTCIA']] = AgenciaTransporte(
                nombre=registro['GNOM_CIA_T'],
                rut=registro['NUMRUTCIA'],
                dig_v=registro['DIGVERCIA']
            )

        nuevo_registro = Registro(
            puerto_embarque=puertos.get(int(registro['PTO_EMB']), None),
            puerto_desembarque=puertos.get(int(registro['PTO_DESEM']), None),
            num_registro=registro['NUMENCRIPTADO'],
            fecha_aceptacion=fecha_aceptacion,
            aduana=aduanas.get(int(registro['ADU']), None),
            tipo_carga=tiposCargas.get(registro['TPO_CARGA'], None),
            nro_manifiesto=registro['NUM_MANIF'],
            agencia_transporte=agenciasTransporte.get(registro['NUMRUTCIA'], None) or nuevos_ruts.get(registro['NUMRUTCIA'], None),
            tpo_bul1=registro['TPO_BUL1'],
            cant_bul1=registro['CANT_BUL1']
        )
        registros.append(nuevo_registro)
        pbar.update(1)

    # Crear nuevas agencias de transporte
    if nuevos_ruts:
        try:
            AgenciaTransporte.objects.bulk_create(nuevos_ruts.values())
            agenciasTransporte.update(nuevos_ruts)
        except Exception as e:
            logging.error(f"Error creando agencias de transporte en {file_path}. Detalle: {e}", exc_info=True)

    # Insertar registros en la base de datos
    if registros:
        try:
            Registro.objects.bulk_create(registros, batch_size=1000)
        except Exception as e:
            logging.error(f"Error insertando registros en {file_path}. Detalle: {e}", exc_info=True)

def thread_process(file_path, columnas, puertos, aduanas, tiposCargas, agenciasTransporte, registros_existentes):
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile, delimiter=';'))
            total_filas = len(reader)

            num_hilos = os.cpu_count() or 4  # Por si os.cpu_count() devuelve None
            if num_hilos > total_filas:  
                # Si hay más hilos que filas, limitar el número de hilos para evitar hilos sin trabajo
                num_hilos = total_filas

            tamaño_lote = max(1, total_filas // num_hilos)

            with tqdm(total=total_filas, desc=f"Procesando {file_path}") as pbar:
                with ThreadPoolExecutor(max_workers=num_hilos) as executor:
                    futuros = []
                    thread_id = 0
                    for i in range(0, total_filas, tamaño_lote):
                        lote_filas = reader[i:i + tamaño_lote]
                        futuros.append(
                            executor.submit(
                                procesar_registros_lote,
                                lote_filas, columnas, puertos, aduanas, tiposCargas,
                                agenciasTransporte, registros_existentes, pbar, file_path, thread_id
                            )
                        )
                        thread_id += 1

                    for futuro in as_completed(futuros):
                        try:
                            futuro.result()
                        except Exception as exc:
                            logging.error(f"Excepción en hilo procesando {file_path}: {exc}", exc_info=True)
    except Exception as e:
        logging.error(f"Error abriendo o leyendo el archivo {file_path}: {e}", exc_info=True)

def cargar_registros_importacion():
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

    puertos = {puerto.codigo: puerto for puerto in Puerto.objects.all()}
    aduanas = {aduana.codigo: aduana for aduana in Aduana.objects.all()}
    tiposCargas = {tipoCarga.codigo: tipoCarga for tipoCarga in TipoCarga.objects.all()}
    agenciasTransporte = {agencia.rut: agencia for agencia in AgenciaTransporte.objects.all()}

    registros_existentes = set(Registro.objects.values_list('num_registro', flat=True))
    file_paths = get_importaciones_file_path()

    for file_path in file_paths:
        thread_process(
            file_path, columnas, puertos, aduanas, tiposCargas,
            agenciasTransporte, registros_existentes
        )
