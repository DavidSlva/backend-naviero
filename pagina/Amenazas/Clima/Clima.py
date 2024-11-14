import requests
import csv
import psycopg2

# Tu clave de API de OpenWeather
api_key = "0c6725042528bb4984cdf3da84490b57"

# Coordenadas de las ciudades
ciudades = {
    "San Antonio": {"lat": -33.5922, "lon": -71.6210},
    "Valparaíso": {"lat": -33.0472, "lon": -71.6127}
}

# URL base para la API
base_url = "https://api.openweathermap.org/data/3.0/onecall"

# Función para obtener el clima actual de una ciudad
def get_current_weather(api_key, lat, lon):
    url = f"{base_url}?lat={lat}&lon={lon}&appid={api_key}&units=metric"  # Temperaturas en Celsius
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Lista para almacenar los resultados
resultados = []

# Obtener el clima actual para cada ciudad y almacenar los resultados
for ciudad, coords in ciudades.items():
    data = get_current_weather(api_key, coords["lat"], coords["lon"])
    if data:
        current_weather = data['current']
        temp = current_weather.get('temp')
        humidity = current_weather.get('humidity')
        description = current_weather['weather'][0].get('description')
        resultados.append([ciudad, temp, humidity, description])
    else:
        print(f"Error al obtener datos para {ciudad}")

# Exportar los resultados a un archivo CSV
csv_filename = "clima_actual_san_antonio_valparaiso.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Ciudad", "Temperatura (°C)", "Humedad (%)", "Descripción"])
    writer.writerows(resultados)

print(f"Datos del clima exportados a {csv_filename}")

# Función para insertar la información del clima desde el archivo CSV en la base de datos PostgreSQL
def insertar_clima_db(csv_filename):
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
            ciudad = row['Ciudad']
            temperatura = row['Temperatura (°C)'] if row['Temperatura (°C)'] else 0
            humedad = row['Humedad (%)'] if row['Humedad (%)'] else 0
            descripcion = row['Descripción']

            # Consulta SQL para insertar los datos en la tabla 'clima'
            sql = "INSERT INTO clima (ciudad, temperatura, humedad, descripcion) VALUES (%s, %s, %s, %s)"
            values = (ciudad, temperatura, humedad, descripcion)
            cursor.execute(sql, values)

    # Guardar los cambios
    db.commit()
    cursor.close()
    db.close()

    print("Datos del clima insertados correctamente en la base de datos.")

# Insertar los datos del clima en la base de datos
insertar_clima_db(csv_filename)
