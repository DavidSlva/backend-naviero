import requests
from bs4 import BeautifulSoup
import csv
import psycopg2

# URL de la página de sismología
url = "https://www.sismologia.cl/"

# Realizar la solicitud HTTP para obtener el contenido de la página
response = requests.get(url)

# Verifica si la solicitud fue exitosa
if response.status_code == 200:
    html_content = response.text

    # Analizar el contenido HTML usando BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Encontrar la tabla con la clase 'sismologia'
    table = soup.find('table', class_='sismologia')

    # Crear una lista para almacenar los datos extraídos
    data = []

    # Iterar sobre las filas de la tabla
    for row in table.find_all('tr')[1:]:  # Saltar la primera fila de encabezados
        columns = row.find_all('td')
        if columns:
            fecha_lugar = columns[0].get_text(strip=True).replace('\n', ' ')
            profundidad = columns[1].get_text(strip=True)
            magnitud = columns[2].get_text(strip=True)
            data.append([fecha_lugar, profundidad, magnitud])

    # Exportar los datos a un archivo CSV
    csv_filename = "sismos_chile.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Fecha Local / Lugar", "Profundidad", "Magnitud"])  # Encabezados del CSV
        writer.writerows(data)

    print(f"Datos de sismos exportados a '{csv_filename}'")
else:
    print(f"Error al obtener la página web: {response.status_code}")


# Función para insertar la información de los sismos desde el archivo CSV en la base de datos PostgreSQL
def insertar_sismos_db(csv_filename):
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
            fecha_lugar = row['Fecha Local / Lugar']
            profundidad = row['Profundidad'] if row['Profundidad'] else 0
            magnitud = row['Magnitud'] if row['Magnitud'] else 0

            # Consulta SQL para insertar los datos en la tabla 'sismos_chile'
            sql = "INSERT INTO sismos_chile (fecha_lugar, profundidad, magnitud) VALUES (%s, %s, %s)"
            values = (fecha_lugar, profundidad, magnitud)
            cursor.execute(sql, values)

    # Guardar los cambios
    db.commit()
    cursor.close()
    db.close()

    print("Datos de sismos insertados correctamente en la base de datos.")

# Insertar los datos de los sismos en la base de datos
insertar_sismos_db(csv_filename)
