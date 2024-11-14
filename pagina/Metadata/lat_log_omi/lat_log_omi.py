import requests
from bs4 import BeautifulSoup
import csv
import psycopg2

# Función para obtener la información del barco y guardarla en un archivo CSV
def obtener_informacion_barco(imo):
    url = f'https://www.vesselfinder.com/?imo={imo}'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'es-419'
    }
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_description = soup.find('meta', {'name': 'description'})

        if meta_description:
            info_barco = meta_description['content']
            nombre_barco = info_barco.split(" ")[0] + " " + info_barco.split(" ")[1]
            latitud = info_barco.split(" ")[5] + " " + info_barco.split(" ")[6]
            longitud = info_barco.split(" ")[7] + " " + info_barco.split(" ")[8]

            # Guardar los datos en un archivo CSV
            file_name = f'barco_{imo}.csv'
            with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['imo', 'nombre_barco', 'latitud', 'longitud', 'informacion_completa'])
                writer.writerow([imo, nombre_barco, latitud, longitud, info_barco])

            print(f"Datos del barco guardados en {file_name}")
            return file_name
        else:
            print("No se pudo encontrar la información del barco en la página.")
            return None
    else:
        print(f"Error en la solicitud HTTP: {response.status_code}")
        return None

# Función para insertar la información del barco desde el CSV en la base de datos PostgreSQL
def insertar_barcos_db(file_name):
    db = psycopg2.connect(
        host="localhost",
        user="postgres",
        password="root",
        database="puerto_naves_amenazas"
    )
    cursor = db.cursor()

    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sql = """INSERT INTO barcos_lat_long (imo, nombre_barco, latitud, longitud, informacion_completa) 
                     VALUES (%s, %s, %s, %s, %s)"""
            values = (row['imo'], row['nombre_barco'], row['latitud'], row['longitud'], row['informacion_completa'])
            cursor.execute(sql, values)

    db.commit()
    cursor.close()
    db.close()

    print(f"Datos del archivo {file_name} insertados correctamente en la base de datos.")

# Ejecutar el proceso
imo_input = input("Introduce el número IMO del barco: ")
csv_file_name = obtener_informacion_barco(imo_input)

if csv_file_name:
    insertar_barcos_db(csv_file_name)
