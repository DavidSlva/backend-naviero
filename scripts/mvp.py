import pandas as pd
import os
from pathlib import Path
import logging
import chardet
# -------------------------------
# a. Definir la Ruta y Obtener la Lista de Archivos
# -------------------------------

# Define la ruta a la carpeta 'downloads'
downloads_path = Path(r"C:\Users\David\Documents\Github\Proyecto Semestral Grafos y Algoritmos\backend\downloads")  # Reemplaza con tu ruta real

# Patrón de búsqueda para archivos (ejemplo: Exportaciones julio 2024.txt)
file_pattern = "Exportaciones * *.txt"
file_list = list(downloads_path.glob(file_pattern))

print(f"Archivos encontrados: {len(file_list)}")
for file in file_list:
    print(f"  - {file.name}")

# -------------------------------
# b. Definir las Columnas
# -------------------------------

# Lista de columnas original (incluye duplicados)
original_columns = [
    "FECHAACEPT",
    "NUMEROIDENT",
    "ADUANA",
    "TIPOOPERACION",
    "CODIGORUTEXPORTADORPPAL",
    "NRO_EXPORTADOR",
    "PORCENTAJEEXPPPAL",
    "COMUNAEXPORTADORPPAL",
    "CODIGORUTEXPSEC",
    "NRO_EXPORTADOR_SEC",
    "PORCENTAJEEXPSECUNDARIO",
    "COMUNAEXPSECUNDARIO",
    "PUERTOEMB",
    "GLOSAPUERTOEMB",
    "REGIONORIGEN",
    "TIPOCARGA",
    "VIATRANSPORTE",
    "PUERTODESEMB",
    "GLOSAPUERTODESEMB",
    "PAISDESTINO",
    "GLOSAPAISDESTINO",
    "NOMBRECIATRANSP",
    "PAISCIATRANSP",
    "RUTCIATRANSP",
    "DVRUTCIATRANSP",
    "NOMBREEMISORDOCTRANSP",
    "RUTEMISOR",
    "DVRUTEMISOR",
    "CODIGOTIPOAUTORIZA",
    "NUMEROINFORMEEXPO",
    "DVNUMEROINFORMEEXP",
    "FECHAINFORMEEXP",
    "MONEDA",
    "MODALIDADVENTA",
    "CLAUSULAVENTA",
    "FORMAPAGO",
    "VALORCLAUSULAVENTA",
    "COMISIONESEXTERIOR",
    "OTROSGASTOS",
    "VALORLIQUIDORETORNO",
    "NUMEROREGSUSP",
    "ADUANAREGSUSP",
    "PLAZOVIGENCIAREGSUSP",
    "TOTALITEM",
    "TOTALBULTOS",
    "PESOBRUTOTOTAL",
    "TOTALVALORFOB",
    "VALORFLETE",
    "CODIGOFLETE",
    "VALORSEGURO",
    "CODIGOSEG",
    "VALORCIF",
    "NUMEROPARCIALIDAD",
    "TOTALPARCIALES",
    "PARCIAL",
    "OBSERVACION",
    "NUMERODOCTOCANCELA",
    "FECHADOCTOCANCELA",
    "TIPODOCTOCANCELA",
    "PESOBRUTOCANCELA",
    "TOTALBULTOSCANCELA",
    "CAMPO - DUS - ITEM",
    "NUMEROITEM",
    "NOMBRE",
    "ATRIBUTO1",
    "ATRIBUTO2",
    "ATRIBUTO3",
    "ATRIBUTO4",
    "ATRIBUTO5",
    "ATRIBUTO6",
    "CODIGOARANCEL",
    "UNIDADMEDIDA",
    "CANTIDADMERCANCIA",
    "FOBUNITARIO",
    "FOBUS",
    "CODIGOOBSERVACION1",
    "VALOROBSERVACION1",
    "GLOSAOBSERVACION1",
    "CODIGOOBSERVACION2",
    "VALOROBSERVACION2",
    "GLOSAOBSERVACION2",
    "CODIGOOBSERVACION3",
    "VALOROBSERVACION3",
    "GLOSAOBSERVACION3",
    "PESOBRUTOITEM",
    "BULTOS",
    "NUMEROBULTO",
    "TIPOBULTO",
    "CANTIDADBULTO",
    "IDENTIFICACIONBULTO",
    "DOCTO. TRANSPORTE",
    "NSECDOCTRANSP",
    "NUMERODOCTRANSP",
    "FECHADOCTRANSP",
    "NOMBRENAVE",
    "NUMEROVIAJE",
]

# Eliminar duplicados y hacer nombres únicos
def make_unique_columns(columns):
    seen = {}
    unique_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            unique_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            unique_columns.append(col)
    return unique_columns

columns = make_unique_columns(original_columns)
print(f"Número total de columnas definidas: {len(columns)}")
expected_num_fields = len(columns)
print(f"Número esperado de campos por línea: {expected_num_fields}")

# -------------------------------
# c. Definir Tipos de Datos Inicialmente como 'string'
# -------------------------------

# Asignar todas las columnas como 'string' inicialmente
dtype_dict = {col: 'string' for col in columns}

# Columnas que contienen fechas y deben ser parseadas
date_columns = [
    "FECHAACEPT",
    "FECHAINFORMEEXP",
    "FECHADOCTOCANCELA",
    "FECHADOCTRANSP",
]

# -------------------------------
# d. Función para Procesar un Solo Archivo en Chunks
# -------------------------------
# Configurar logging para registrar errores y advertencias
logging.basicConfig(
    filename='procesamiento_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def detectar_codificacion(file_path, num_bytes=10000) :
    """
    Detecta la codificación de un archivo leyendo los primeros `num_bytes` bytes.
    """
    with open(file_path, 'rb') as f :
        rawdata = f.read(num_bytes)
    resultado = chardet.detect(rawdata)
    return resultado['encoding']


def encontrar_lineas_problema(file_path, expected_num_fields, encoding='utf-8', delimiter=';') :
    """
    Lee el archivo línea por línea y registra las líneas que no tienen el número esperado de campos
    o que no se pueden decodificar.
    """
    bad_lines = []
    with open(file_path, 'r', encoding=encoding, errors='strict') as f :
        for line_num, line in enumerate(f, start=1) :
            try :
                # Remover saltos de línea y dividir por el delimitador
                fields = line.rstrip('\n').split(delimiter)
                if len(fields) != expected_num_fields :
                    bad_lines.append((line_num, line.strip()))
            except UnicodeDecodeError as e :
                bad_lines.append((line_num, f"Error de decodificación: {e}"))
    return bad_lines


def registrar_lineas_problema(file_path, expected_num_fields, encoding='utf-8', delimiter=';') :
    """
    Encuentra y registra las líneas malformadas en el archivo.
    """
    lineas_problema = encontrar_lineas_problema(file_path, expected_num_fields, encoding, delimiter)

    if lineas_problema :
        with open('lineas_problema.log', 'a', encoding='utf-8') as log_file :
            for line_num, content in lineas_problema :
                log_file.write(f"Línea {line_num}: {content}\n")
        print(f"Se han registrado {len(lineas_problema)} líneas problemáticas en 'lineas_problema.log'.")
    else :
        print("No se encontraron líneas problemáticas.")


def process_file_in_chunks(file_path, columns, dtype_dict, date_columns, expected_num_fields, chunksize=100000) :
    """
    Lee un archivo en chunks y devuelve una lista de DataFrames procesados.
    Valida que cada línea tenga el número correcto de campos y maneja errores de decodificación y parsing.
    """
    logging.info(f"Procesando el archivo: {file_path.name}")
    chunks = []

    # Detectar la codificación del archivo
    try :
        encoding = detectar_codificacion(file_path)
        logging.info(f"Codificación detectada para {file_path.name}: {encoding}")
    except Exception as e :
        logging.error(f"No se pudo detectar la codificación para {file_path.name}: {e}")
        return chunks

    try :
        chunk_iter = pd.read_csv(
            file_path,
            sep=';',
            header=None,
            names=columns,
            dtype=dtype_dict,
            decimal=',',
            na_values=[''],
            keep_default_na=True,
            on_bad_lines='warn',  # O 'skip' para omitir líneas malas
            engine='python',
            chunksize=chunksize,
            encoding=encoding,
        )

        # Variable para rastrear el número de línea global
        linea_global = 0

        for i, chunk in enumerate(chunk_iter) :
            # Actualizar el contador de líneas
            linea_global += len(chunk)

            # Validar el número de campos por línea
            actual_num_fields = chunk.shape[1]
            logging.info(
                f"Chunk {i + 1} de {file_path.name}: {actual_num_fields} campos, se esperaban {expected_num_fields}.")

            if actual_num_fields != expected_num_fields :
                logging.warning(
                    f"Chunk {i + 1} de {file_path.name} tiene {actual_num_fields} campos, se esperaban {expected_num_fields}.")
                # Manejar las filas con campos incorrectos
                # Por ejemplo, recortar el DataFrame para tener el número correcto de columnas
                chunk = chunk.iloc[:, :expected_num_fields]

            # Convertir columnas de fecha manualmente
            for columna in date_columns :
                if columna in chunk.columns :
                    try :
                        chunk[columna] = pd.to_datetime(chunk[columna], format='%d%m%Y', errors='coerce')
                    except Exception as e :
                        logging.error(
                            f"Error al convertir la columna {columna} en chunk {i + 1} de {file_path.name}: {e}")

            logging.info(f"Procesando chunk {i + 1} de {file_path.name}")
            chunks.append(chunk)

        logging.info(f"Archivo {file_path.name} procesado y agregado.")

    except pd.errors.ParserError as e :
        logging.error(f"Error de Parser al procesar el archivo {file_path.name}: {e}")
    except UnicodeDecodeError as e :
        logging.error(f"Error de decodificación al procesar el archivo {file_path.name}: {e}")
    except Exception as e :
        logging.error(f"Error inesperado al procesar el archivo {file_path.name}: {e}")

    return chunks

# -------------------------------
# e. Leer y Procesar Todos los Archivos
# -------------------------------

# Inicializar una lista para almacenar los DataFrames procesados
dataframes = []

# Iterar sobre cada archivo y leerlo en bloques
for file_path in file_list:
    file_chunks = process_file_in_chunks(file_path, columns, dtype_dict, date_columns, expected_num_fields=96)
    dataframes.extend(file_chunks)
    print(f"  Archivo {file_path.name} procesado y agregado.")

# -------------------------------
# f. Concatenar Todos los DataFrames en Uno Solo
# -------------------------------

if dataframes:
    print("Concatenando todos los DataFrames en uno solo...")
    df = pd.concat(dataframes, ignore_index=True)
    print(f"DataFrame creado con éxito con {df.shape[0]} filas y {df.shape[1]} columnas.")
else:
    print("No se procesaron DataFrames. Verifica los archivos y el formato.")

# -------------------------------
# g. Convertir Columnas a Tipos de Datos Originales con Manejo de Errores
# -------------------------------

# Define un diccionario con las conversiones deseadas
conversion_dict = {
    "NUMEROIDENT": "Int64",
    "ADUANA": "Int64",
    "TIPOOPERACION": "Int64",
    "CODIGORUTEXPORTADORPPAL": "Int64",
    "NRO_EXPORTADOR": "Int64",
    "PORCENTAJEEXPPPAL": "float32",
    "COMUNAEXPORTADORPPAL": "Int64",
    "CODIGORUTEXPSEC": "Int64",
    "NRO_EXPORTADOR_SEC": "Int64",
    "PORCENTAJEEXPSECUNDARIO": "float32",
    "COMUNAEXPSECUNDARIO": "Int64",
    "PUERTOEMB": "Int64",
    "REGIONORIGEN": "Int64",
    "TIPOCARGA": "category",
    "VIATRANSPORTE": "Int64",
    "PUERTODESEMB": "Int64",
    "PAISDESTINO": "Int64",
    "PAISCIATRANSP": "Int64",
    "RUTCIATRANSP": "Int64",
    "MONEDA": "Int64",
    "MODALIDADVENTA": "Int64",
    "CLAUSULAVENTA": "Int64",
    "FORMAPAGO": "Int64",
    "VALORCLAUSULAVENTA": "float32",
    "COMISIONESEXTERIOR": "float32",
    "OTROSGASTOS": "float32",
    "VALORLIQUIDORETORNO": "float32",
    "NUMEROREGSUSP": "Int64",
    "ADUANAREGSUSP": "Int64",
    "PLAZOVIGENCIAREGSUSP": "Int64",
    "TOTALITEM": "Int64",
    "TOTALBULTOS": "Int64",
    "PESOBRUTOTOTAL": "float32",
    "TOTALVALORFOB": "float32",
    "VALORFLETE": "float32",
    "VALORSEGURO": "float32",
    "VALORCIF": "float32",
    "NUMEROPARCIALIDAD": "Int64",
    "TOTALPARCIALES": "Int64",
    "NUMERODOCTOCANCELA": "Int64",
    "PESOBRUTOCANCELA": "float32",
    "TOTALBULTOSCANCELA": "Int64",
    "UNIDADMEDIDA": "Int64",
    "CANTIDADMERCANCIA": "float32",
    "FOBUNITARIO": "float32",
    "FOBUS": "float32",
    "CODIGOOBSERVACION1": "Int16",
    "CODIGOOBSERVACION2": "Int16",
    "CODIGOOBSERVACION3": "Int16",
    "NUMEROITEM": "Int64",
    "NSECDOCTRANSP": "Int16",
}

# Convertir las columnas según el conversion_dict
for col, dtype in conversion_dict.items():
    if col in df.columns:
        print(f"Convirtiendo columna {col} a {dtype}...")
        if dtype.startswith('Int') or dtype.startswith('float'):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
        elif dtype == 'category':
            df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype(dtype)

print("Conversión de tipos de datos completada.")

# -------------------------------
# h. Guardar el DataFrame Resultante en Formato Parquet (Opcional)
# -------------------------------

# Guardar el DataFrame en formato Parquet para un acceso más rápido en el futuro
output_path = downloads_path / "exportaciones_2024_combinado_clean.parquet"
try:
    print(f"Guardando el DataFrame en {output_path}...")
    df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
    print("Guardado completado exitosamente.")
except Exception as e:
    print(f"Error al guardar el DataFrame: {e}")

# -------------------------------
# i. Limpieza de Memoria (Opcional)
# -------------------------------

# Liberar memoria eliminando la lista de DataFrames si ya no se necesita
del dataframes
