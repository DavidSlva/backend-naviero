import requests
from lxml import html
import pandas as pd
import psycopg2
import numpy as np

# Función para descargar metadata desde la página
def descargar_metadata(url, headers):
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        tree = html.fromstring(response.content)
        data = tree.xpath('//*[@id="ctl00_ContentPlaceHolder1_GridView_Lista"]')
        
        headers = ['E.T.A.', 'Agencia', 'Nave', 'Eslora', 'Terminal', 'Emp. muellaje', 'Carga', 'Detalle', 'Cantidad', 'Operación']
        extracted_data = []
        for table in data:
            for row in table.xpath('.//tr'):
                row_data = [cell.text_content().strip() for cell in row.xpath('.//td')]
                if len(row_data) > 0:
                    extracted_data.append(row_data)
        df = pd.DataFrame(extracted_data, columns=headers)
        return df
    else:
        print("Error al descargar la página")
        return None

# Guardar la metadata en un archivo CSV
def guardar_metadata_csv(df, file_name):
    df.to_csv(file_name, mode='a', index=False, header=False)
    print(f"Datos guardados en {file_name}")

# Conexión a la base de datos PostgreSQL
def conectar_db():
    db = psycopg2.connect(
        host="localhost",
        user="postgres",        # Reemplaza por tu usuario de PostgreSQL
        password="root",            # Contraseña (dejada en blanco según tu indicación)
        database="puerto_naves_amenazas"
    )
    return db

# Insertar datos del CSV en la base de datos
def insertar_datos_db(db, file_name):
    cursor = db.cursor()
    df = pd.read_csv(file_name, header=None)  # Cargar sin encabezados
    # Asignar nombres a las columnas basados en su contenido
    df.columns = ['E.T.A.', 'Agencia', 'Nave', 'Eslora', 'Terminal', 'Emp. muellaje', 'Carga', 'Detalle', 'Cantidad', 'Operación']
    
    # Convertir la columna 'Eslora' a un formato adecuado para PostgreSQL
    df['Eslora'] = df['Eslora'].str.replace(',', '.').astype(float)
    
    # Manejar las columnas de fecha (E.T.A.) correctamente, convirtiendo a datetime o a NULL si es NaN
    df['E.T.A.'] = pd.to_datetime(df['E.T.A.'], errors='coerce')  # Convertir a datetime, 'coerce' pone NaT en valores inválidos
    
    for _, row in df.iterrows():
        # Si las fechas son NaT (not a time), asignamos None (NULL en SQL)
        fecha_entrada = row['E.T.A.'] if pd.notna(row['E.T.A.']) else None
        fecha_salida = row['E.T.A.'] if pd.notna(row['E.T.A.']) else None
        
        sql = """INSERT INTO barcos_recalando 
                 (nombre_barco, puerto, tipo_barco, eslora, agente, carga, detalle_operacion, fecha_entrada, fecha_salida) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (row['Nave'], row['Terminal'], row['Operación'], row['Eslora'], row['Agencia'], row['Carga'], row['Detalle'], fecha_entrada, fecha_salida)
        cursor.execute(sql, values)
    db.commit()
    print(f"Datos insertados en la base de datos desde {file_name}")

# URL de la página para descargar la metadata
url = 'https://gessup.puertosanantonio.com/Planificaciones/general.aspx'

# Encabezados para la solicitud
headers = {
    'Host': 'gessup.puertosanantonio.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36',
    'Accept-Language': 'es-ES',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

# Archivo CSV donde se guarda la metadata
file_name = 'metadata_san_antonio.csv'

# Descargar metadata y guardarla en CSV
df = descargar_metadata(url, headers)
if df is not None:
    guardar_metadata_csv(df, file_name)

    # Conectar a la base de datos e insertar los datos
    db = conectar_db()
    insertar_datos_db(db, file_name)
