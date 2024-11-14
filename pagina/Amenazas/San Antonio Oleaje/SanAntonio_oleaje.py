import requests
import pandas as pd
import csv  # Asegúrate de importar el módulo csv
import psycopg2

# Coordenadas para San Antonio, Chile
latitude = -33.5922
longitude = -71.6210

# Endpoint de la API de Open-Meteo para obtener datos de oleaje
url = f"https://marine-api.open-meteo.com/v1/marine?latitude={latitude}&longitude={longitude}&hourly=wave_height,wave_direction,wave_period"

# Realizar la solicitud HTTP para obtener los datos
response = requests.get(url)

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    data = response.json()  # Obtener los datos en formato JSON
    
    # Extraer los datos de oleaje
    hours = data['hourly']['time']
    wave_heights = data['hourly']['wave_height']
    wave_directions = data['hourly']['wave_direction']
    wave_periods = data['hourly']['wave_period']

    # Combinar los datos en un DataFrame
    wave_data = list(zip(hours, wave_heights, wave_directions, wave_periods))
    df = pd.DataFrame(wave_data, columns=['Hora', 'Altura del Oleaje (m)', 'Dirección del Oleaje (°)', 'Periodo del Oleaje (s)'])

    # Mostrar los datos
    print(df)

    # Exportar a CSV
    csv_filename = "oleaje_san_antonio_open_meteo.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Datos de oleaje exportados a '{csv_filename}'")
else:
    print(f"Error al obtener datos de oleaje: {response.status_code}")

# Función para insertar la información del oleaje desde el archivo CSV en la base de datos PostgreSQL
def insertar_oleaje_db(csv_filename):
    # Conexión a la base de datos PostgreSQL
    db = psycopg2.connect(
        host="localhost",
        user="postgres",        # Reemplaza por tu usuario
        password="root",        # Reemplaza por tu contraseña
        database="puerto_naves_amenazas"
    )
    cursor = db.cursor()

    # Leer los datos desde el archivo CSV
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            hora = row['Hora']
            altura = row['Altura del Oleaje (m)'] if row['Altura del Oleaje (m)'] else 0
            direccion = row['Dirección del Oleaje (°)'] if row['Dirección del Oleaje (°)'] else 0
            periodo = row['Periodo del Oleaje (s)'] if row['Periodo del Oleaje (s)'] else 0

            # Consulta SQL para insertar los datos en la tabla 'oleaje_san_antonio'
            sql = "INSERT INTO oleaje_san_antonio (hora, altura, direccion, periodo) VALUES (%s, %s, %s, %s)"
            values = (hora, altura, direccion, periodo)
            cursor.execute(sql, values)

    # Guardar los cambios
    db.commit()
    cursor.close()
    db.close()

    print("Datos de oleaje insertados correctamente en la base de datos.")

# Insertar los datos del oleaje en la base de datos
insertar_oleaje_db(csv_filename)
