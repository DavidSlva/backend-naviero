import requests
import csv
import psycopg2

# Función para descargar la información de los barcos y mostrar las claves disponibles
def descargar_barcos():
    url = "https://orion.directemar.cl/sitport/back/users/consultaNaveRecalando"
    headers = {
        "Host": "orion.directemar.cl",
        "Content-Length": "2",
        "Sec-Ch-Ua": '"Not(A:Brand";v="24", "Chromium";v="122"',
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Sec-Ch-Ua-Mobile": "?0",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36"
    }
    data = {}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        json_response = response.json()
        barcos = json_response.get("recordsets", [[]])[0]

        if len(barcos) > 0:
            # Imprimir las claves disponibles en el primer barco para depuración
            print("Claves disponibles en los datos de barcos:")
            print(barcos[0].keys())

        # Filtrar los barcos que están en "VALPARAISO" o "SAN ANTONIO"
        # barcos_filtrados = [barco for barco in barcos if barco.get("nombrePuerto") in ["VALPARAISO", "SAN ANTONIO"]]
        # return barcos_filtrados
        return barcos
    else:
        print("Error en la solicitud:", response.status_code)
        return None

# Función para escribir los datos en un CSV
def escribir_csv(barcos_filtrados):
    file_name = 'barcos_valparaiso_san_antonio.csv'
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Escribir los encabezados
        writer.writerow(['nombre_barco', 'puerto', 'tipo_barco', 'eslora', 'bandera', 'agente', 'carga', 'detalle_operacion', 'fecha_entrada', 'fecha_salida'])
        # Escribir los datos
        for barco in barcos_filtrados:
            writer.writerow([
                barco.get('Nombre_2', 'N/A'),         # Ajuste del nombre correcto de la clave 'Nombre_2'
                barco.get('nombrePuerto', 'N/A'),
                barco.get('TipoNave', 'N/A'),         # Clave ajustada a 'TipoNave'
                barco.get('dmEslora', 'N/A'),         # Ajuste de clave 'dmEslora'
                barco.get('bandera', 'N/A'),
                barco.get('agente', 'N/A'),
                barco.get('caracteristica', 'N/A'),   # 'caracteristica' es el campo de carga
                barco.get('detalle_operacion', 'N/A'),  # Revisar si esta clave está correcta en la API
                barco.get('fecha', 'N/A'),            # 'fecha' corresponde a 'fechaEntrada'
                barco.get('fechafin', 'N/A')          # 'fechafin' es 'fechaSalida'
            ])
    
    print(f"Datos de los barcos guardados en {file_name}")
    return file_name

# Función para insertar la información de los barcos desde el CSV en la base de datos PostgreSQL
def insertar_barcos_db(db, barcos_filtrados):
    cursor = db.cursor()

    for barco in barcos_filtrados:
        sql = """INSERT INTO barcos_recalando (nombre_barco, puerto, tipo_barco, eslora, bandera, agente, carga, detalle_operacion, fecha_entrada, fecha_salida) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (
            barco.get('Nombre_2'),
            barco.get('nombrePuerto'),
            barco.get('TipoNave'),
            barco.get('dmEslora'),
            barco.get('bandera'),
            barco.get('agente'),
            barco.get('caracteristica'),
            barco.get('detalle_operacion'),
            barco.get('fecha'),
            barco.get('fechafin')
        )
        cursor.execute(sql, values)

    # Guardar los cambios
    db.commit()
    print("Datos insertados correctamente en la base de datos.")

# Conexión a la base de datos PostgreSQL
def conectar_db():
    db = psycopg2.connect(
        host="localhost",
        user="postgres",        # Reemplaza por tu usuario de PostgreSQL
        password="root",            # Contraseña (dejada en blanco según tu indicación)
        database="puerto_naves_amenazas"
    )
    return db

if __name__ == "__main__":
    # Ejecutar el proceso
    barcos_filtrados = descargar_barcos()

    if barcos_filtrados:
        # Guardar los datos en CSV
        escribir_csv(barcos_filtrados)

        # Conectar a la base de datos e insertar los datos
        db = conectar_db()
        insertar_barcos_db(db, barcos_filtrados)
        db.close()
