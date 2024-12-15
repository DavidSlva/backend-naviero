# utils.py
import pandas as pd
from django.db import transaction
from .models import VolumenTotal, VolumenPorPuerto, Puerto
import logging

def procesar_y_cargar_datos(parquet_path):
    """
    Función para procesar el archivo Parquet, calcular métricas agregadas y cargar
    los resultados en las tablas VolumenTotal y VolumenPorPuerto.
    """
    logging.info("Iniciando la carga y procesamiento de datos desde el archivo Parquet.")

    # Leer el archivo Parquet
    try:
        df = pd.read_parquet(parquet_path)
        logging.info(f"Archivo Parquet cargado exitosamente con {len(df)} registros.")
    except Exception as e:
        logging.error(f"Error al leer el archivo Parquet: {e}")
        return

    # Asegurarse de que 'FECHAACEPT' sea de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(df['FECHAACEPT']):
        df['FECHAACEPT'] = pd.to_datetime(df['FECHAACEPT'], errors='coerce')
        logging.info("Columna 'FECHAACEPT' convertida a datetime.")

    # Eliminar filas con valores nulos en 'FECHAACEPT' y 'PUERTOEMB'
    df = df.dropna(subset=['FECHAACEPT', 'PUERTOEMB'])
    logging.info(f"Datos después de eliminar nulos en 'FECHAACEPT' y 'PUERTOEMB': {len(df)} registros.")

    # Omitir filas donde 'PUERTOEMB' es 0
    df = df[df['PUERTOEMB'] != 0]
    logging.info(f"Datos después de omitir 'PUERTOEMB' = 0: {len(df)} registros.")

    # Crear columna 'SEMANA' (inicio de la semana)
    df['SEMANA'] = df['FECHAACEPT'].dt.to_period('W').apply(lambda r: r.start_time)
    logging.info("Columna 'SEMANA' creada.")
    logging.debug(df[['FECHAACEPT', 'SEMANA']].head())

    # Calcular Volumen Total por Semana
    volumen_total_df = df.groupby('SEMANA')['PESOBRUTOTOTAL'].sum().reset_index()
    volumen_total_df.rename(columns={'PESOBRUTOTOTAL': 'volumen_total'}, inplace=True)
    logging.info("Volumen Total por Semana calculado.")
    logging.debug(volumen_total_df.head())

    # Calcular Volumen por Puerto y Semana
    volumen_puerto_df = df.groupby(['PUERTOEMB', 'GLOSAPUERTOEMB', 'SEMANA'])['PESOBRUTOTOTAL'].sum().reset_index()
    volumen_puerto_df.rename(columns={'PESOBRUTOTOTAL': 'volumen'}, inplace=True)
    logging.info("Volumen por Puerto y Semana calculado.")
    logging.debug(volumen_puerto_df.head())

    # Obtener todos los puertos existentes en la base de datos
    puertos_db = Puerto.objects.all().values('codigo')
    puerto_set = set(puerto['codigo'] for puerto in puertos_db)
    logging.info(f"Total de puertos en la base de datos: {len(puerto_set)}")

    # Filtrar los puertos que existen en la base de datos
    volumen_puerto_df = volumen_puerto_df[volumen_puerto_df['PUERTOEMB'].isin(puerto_set)]
    logging.info(f"Datos después de filtrar puertos existentes: {len(volumen_puerto_df)} registros.")

    # Preparar objetos para VolumenTotal
    volumen_total_objs = [
        VolumenTotal(
            semana=row['SEMANA'],
            volumen_total=row['volumen_total']
        )
        for _, row in volumen_total_df.iterrows()
    ]
    logging.info(f"Objetos de VolumenTotal preparados: {len(volumen_total_objs)} registros.")

    # Preparar objetos para VolumenPorPuerto
    volumen_puerto_objs = []
    for _, row in volumen_puerto_df.iterrows():
        puerto_codigo = row['PUERTOEMB']
        # No necesitamos mapear a 'id', ya que 'codigo' es la PK
        volumen_puerto_objs.append(
            VolumenPorPuerto(
                puerto_id=puerto_codigo,  # Asignar el 'codigo' del puerto
                glosapuertoemb=row['GLOSAPUERTOEMB'],
                semana=row['SEMANA'],
                volumen=row['volumen']
            )
        )
    logging.info(f"Objetos de VolumenPorPuerto preparados: {len(volumen_puerto_objs)} registros.")

    # Cargar datos en la base de datos
    try:
        with transaction.atomic():
            # Borrar datos existentes si es necesario
            VolumenTotal.objects.all().delete()
            VolumenPorPuerto.objects.all().delete()
            logging.info("Datos anteriores eliminados de VolumenTotal y VolumenPorPuerto.")

            # Crear registros masivamente
            VolumenTotal.objects.bulk_create(volumen_total_objs, batch_size=10000)
            logging.info(f"{len(volumen_total_objs)} registros cargados en VolumenTotal.")

            VolumenPorPuerto.objects.bulk_create(volumen_puerto_objs, batch_size=10000)
            logging.info(f"{len(volumen_puerto_objs)} registros cargados en VolumenPorPuerto.")
    except Exception as e:
        logging.error(f"Error al cargar datos en la base de datos: {e}")
        return

    logging.info("Carga y procesamiento de datos completados exitosamente.")
