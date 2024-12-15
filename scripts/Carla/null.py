import pandas as pd
import numpy as np
import os
import logging
import argparse

# Configuración de los logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("check_nulls.log"),
        logging.StreamHandler()
    ]
)


def load_data(data_path_parquet=None, data_path_csv=None) :
    """
    Carga los datos desde un archivo Parquet o CSV.

    Parámetros:
    - data_path_parquet: Ruta al archivo Parquet.
    - data_path_csv: Ruta al archivo CSV.

    Retorna:
    - DataFrame de pandas con los datos cargados.
    """
    if data_path_parquet :
        if not os.path.exists(data_path_parquet) :
            logging.error(f"No se encontró el archivo Parquet en la ruta: {data_path_parquet}")
            raise FileNotFoundError(f"No se encontró el archivo Parquet en la ruta: {data_path_parquet}")
        df = pd.read_parquet(data_path_parquet)
        logging.info(f"Datos cargados desde Parquet: {data_path_parquet}")
    elif data_path_csv :
        if not os.path.exists(data_path_csv) :
            logging.error(f"No se encontró el archivo CSV en la ruta: {data_path_csv}")
            raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {data_path_csv}")
        df = pd.read_csv(data_path_csv, parse_dates=['FECHAACEPT'])
        logging.info(f"Datos cargados desde CSV: {data_path_csv}")
    else :
        logging.error("Debes proporcionar al menos una ruta de archivo (Parquet o CSV).")
        raise ValueError("Proporciona al menos una ruta de archivo (Parquet o CSV).")
    return df


def analyze_nulls(df, columns) :
    """
    Analiza y reporta los valores nulos en las columnas especificadas.

    Parámetros:
    - df: DataFrame de pandas.
    - columns: Lista de nombres de columnas a analizar.

    Retorna:
    - Un diccionario con la cuenta de nulos por columna.
    """
    null_counts = df[columns].isnull().sum()
    total_nulls = null_counts.sum()
    logging.info("=== Análisis de Valores Nulos ===")
    for col in columns :
        logging.info(f"Columna '{col}': {null_counts[col]} valores nulos ({(null_counts[col] / len(df)) * 100:.2f}%)")
    logging.info(
        f"Total de valores nulos en las columnas analizadas: {total_nulls} ({(total_nulls / len(df)) * 100:.2f}%)")
    return null_counts.to_dict()


def save_nulls_details(df, columns, output_dir="./null_details") :
    """
    Guarda las filas con valores nulos en las columnas especificadas en archivos separados.

    Parámetros:
    - df: DataFrame de pandas.
    - columns: Lista de nombres de columnas a analizar.
    - output_dir: Directorio donde se guardarán los archivos.
    """
    os.makedirs(output_dir, exist_ok=True)
    for col in columns :
        null_df = df[df[col].isnull()]
        count_null = null_df.shape[0]
        if count_null > 0 :
            output_path = os.path.join(output_dir, f"null_{col}.csv")
            null_df.to_csv(output_path, index=False)
            logging.info(f"Guardadas {count_null} filas con valores nulos en '{col}' en: {output_path}")
        else :
            logging.info(f"No hay valores nulos en la columna '{col}'. No se creó ningún archivo.")


def main() :
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(
        description="Script para analizar y reportar valores nulos en un conjunto de datos.")
    parser.add_argument('--parquet', type=str, help="Ruta al archivo Parquet.")
    parser.add_argument('--csv', type=str, help="Ruta al archivo CSV.")
    parser.add_argument('--columns', nargs='+', default=['FECHAACEPT', 'PESOBRUTOTOTAL', 'PUERTOEMB'],
                        help="Lista de columnas a analizar para valores nulos.")
    parser.add_argument('--output_dir', type=str, default="./null_details",
                        help="Directorio para guardar detalles de filas con nulos.")

    args = parser.parse_args()

    # Cargar datos
    df = load_data(data_path_parquet=r"C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet", data_path_csv=args.csv)

    # Analizar valores nulos
    null_counts = analyze_nulls(df, args.columns)

    # Guardar detalles de filas con valores nulos
    save_nulls_details(df, args.columns, output_dir=args.output_dir)

    # Opcional: Guardar resumen de nulos en un archivo CSV
    summary_path = os.path.join(args.output_dir, "null_summary.csv")
    summary_df = pd.DataFrame(list(null_counts.items()), columns=['Column', 'Null_Count'])
    summary_df['Null_Percentage'] = (summary_df['Null_Count'] / len(df)) * 100
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Resumen de valores nulos guardado en: {summary_path}")


if __name__ == "__main__" :
    main()
