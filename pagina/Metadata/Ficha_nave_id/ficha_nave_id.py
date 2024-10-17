import requests
import json
import csv
import psycopg2

# Función para descargar la ficha de la nave desde la API y guardarla en un CSV
def descargar_ficha_nave(id_nave):
    base_url = "https://orion.directemar.cl/sitport/back/users/FichaNave"
    headers = {
        "Host": "orion.directemar.cl",
        "Sec-Ch-Ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "es-419",
        "Sec-Ch-Ua-Mobile": "?0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Sec-Ch-Ua-Platform": '"macOS"',
        "Origin": "https://sitport.directemar.cl",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://sitport.directemar.cl/",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }

    url = f"{base_url}/{id_nave}"
    
    # Realizar la solicitud GET
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        json_response = response.json()
        first_item = json_response[0] if json_response else None
        
        if first_item and 'datosnave' in first_item:
            filtered_response = first_item['datosnave']
            
            # Guardar la información en un archivo CSV
            file_name = f'ficha_nave_{id_nave}.csv'
            with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Escribir encabezados
                writer.writerow(filtered_response.keys())
                # Escribir los datos
                writer.writerow(filtered_response.values())
                
            print(f"Datos de la nave guardados en {file_name}")
            return file_name
        else:
            print(f"La respuesta no contiene 'datosnave'.")
            return None
    else:
        print(f"Error en la solicitud: {response.status_code}")
        return None

# Función para insertar la ficha de la nave desde el CSV en la base de datos PostgreSQL
def insertar_ficha_nave_db(db, file_name):
    cursor = db.cursor()

    # Leer el archivo CSV y preparar los datos para la inserción
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Convertir los diccionarios a cadenas JSON válidas para PostgreSQL
            pais = json.dumps(json.loads(row['pais'].replace("'", '"'))) if row['pais'] else None
            bandera = json.dumps(json.loads(row['bandera'].replace("'", '"'))) if row['bandera'] else None
            estado = json.dumps(json.loads(row['estado'].replace("'", '"'))) if row['estado'] else None

            sql = '''INSERT INTO ficha_nave (id, nombre, eslora, manga, bandera, pais, estado, senal, caladoMax) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
            # Usar las columnas correctas de acuerdo al CSV
            values = (row['id'], row['nombre'], row['eslora'], row['manga'], bandera, pais, estado, row['senal'], row['caladoMax'])
            cursor.execute(sql, values)

    db.commit()
    print(f"Datos del archivo {file_name} insertados correctamente en la base de datos.")

# Conexión a la base de datos PostgreSQL
def conectar_db():
    db = psycopg2.connect(
        host="localhost",
        user="postgres",        # Reemplaza por tu usuario de PostgreSQL
        password="root",            # Contraseña (dejada en blanco según tu indicación)
        database="puerto_naves_amenazas"
    )
    return db

# Solicitar ID de la nave
id_nave = input("Introduce el ID de la nave: ")

# Descargar ficha de la nave y guardarla en CSV
csv_file_name = descargar_ficha_nave(id_nave)

if csv_file_name:
    # Conectar a la base de datos e insertar la ficha de la nave
    db = conectar_db()
    insertar_ficha_nave_db(db, csv_file_name)
    db.close()
