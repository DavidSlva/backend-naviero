import math
import os
import random
import string
import pandas as pd
import requests
import logging
from bs4 import BeautifulSoup
from lxml import html
import networkx as nx
from application.models import Arista, Nodo
from backend import settings
from collection_manager.models import Puerto
from interpreter.models import Registro
import pickle
from django.db import transaction

# Configurar logger
logger = logging.getLogger(__name__)

class ServicioExternoError(Exception):
    """Excepción personalizada para errores en servicios externos."""
    pass

class NoSeEncuentraInformacionError(Exception):
    """Excepción personalizada cuando no se encuentra la información solicitada."""
    pass

def obtener_puertos():
    """
    Obtener todas las rutas de los archivos de importaciones y exportaciones disponibles.
    """
    # Obtener todos los registros de la base de datos
    puertos = Puerto.objects.all()
    # Crear una lista de diccionarios donde cada diccionario representa un registro completo
    return puertos
def calcular_distancia(lat1, lon1, lat2, lon2):
    # Fórmula del haversine para calcular la distancia entre dos coordenadas geográficas
    R = 6371  # Radio de la Tierra en km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Retorna la distancia en kilómetros

GRAFO_FILE_PATH = 'grafo.pickle'  # Define la ruta donde guardarás el archivo del grafo

def construir_grafo():
    # Verificar si el archivo del grafo existe
    if os.path.exists(GRAFO_FILE_PATH):
        with open(GRAFO_FILE_PATH, 'rb') as f:
            G = pickle.load(f)
        print("Grafo cargado desde el archivo.")
        return G

    # Si no hay archivo, construimos el grafo desde la base de datos o desde cero
    if Nodo.objects.exists() and Arista.objects.exists():
        G = nx.Graph()

        # Cargar nodos desde la base de datos
        nodos = Nodo.objects.all()
        for nodo in nodos:
            G.add_node(nodo.codigo, nombre=nodo.nombre, pais=nodo.pais, latitud=nodo.latitud, longitud=nodo.longitud)

        # Cargar aristas desde la base de datos
        aristas = Arista.objects.all()
        for arista in aristas:
            G.add_edge(arista.origen.codigo, arista.destino.codigo, distancia=arista.distancia)

        print("Grafo cargado desde la base de datos.")
    else:
        # Si no existe el grafo en la base de datos, crear un nuevo grafo
        G = nx.Graph()

        # Obtener registros de puertos y sus conexiones
        registros = Registro.objects.values_list('puerto_embarque_id', 'puerto_desembarque_id')

        # Obtener los puertos
        puertos = obtener_puertos()  # Asumimos que retorna un queryset de puertos

        # Añadir nodos (puertos) al grafo y almacenar los nodos en una lista para batch insert
        nodos_db = []
        for puerto in puertos:
            G.add_node(puerto.codigo, nombre=puerto.nombre, pais=puerto.pais, latitud=puerto.latitud, longitud=puerto.longitud)
            nodos_db.append(Nodo(
                codigo=puerto.codigo,
                nombre=puerto.nombre,
                pais=puerto.pais,
                latitud=puerto.latitud,
                longitud=puerto.longitud
            ))

        # Insertar todos los nodos a la base de datos en una sola operación
        Nodo.objects.bulk_create(nodos_db, ignore_conflicts=True)  # `ignore_conflicts` ignora los duplicados

        # Añadir aristas (conexiones entre puertos) al grafo y almacenar en una lista para batch insert
        edges = {
            (puerto_origen, puerto_destino)
            for puerto_origen, puerto_destino in registros
            if puerto_origen and puerto_destino and puerto_origen != 0 and puerto_destino != 0
        }

        aristas_db = []
        for origen_codigo, destino_codigo in edges:
            G.add_edge(origen_codigo, destino_codigo)
            nodo_origen = G.nodes[origen_codigo]
            nodo_destino = G.nodes[destino_codigo]
            distancia = calcular_distancia(
                nodo_origen['latitud'], nodo_origen['longitud'],
                nodo_destino['latitud'], nodo_destino['longitud']
            )
            aristas_db.append(Arista(
                origen=Nodo.objects.get(codigo=origen_codigo),
                destino=Nodo.objects.get(codigo=destino_codigo),
                distancia=distancia
            ))

        Arista.objects.bulk_create(aristas_db, ignore_conflicts=True)

        print("Grafo construido y guardado en la base de datos.")

    # Guardar el grafo en un archivo para uso futuro
    with open(GRAFO_FILE_PATH, 'wb') as f:
        pickle.dump(G, f)

    print("Grafo guardado en el archivo.")
    return G
def inhabilitar_puerto(G, codigo_puerto):
    """
    Inhabilita un puerto en la base de datos.
    """
    G.remove_node(codigo_puerto)
    return G

def obtener_vecinos_puerto(G, codigo_puerto):
    """
    Obtiene los vecinos de un puerto en la base de datos.
    """
    if G.has_node(codigo_puerto):
        vecinos = list(G.neighbors(codigo_puerto))
        vecinos = [int(v) for v in vecinos]
        return vecinos
    else:
        return []   
    
def visualizar_rutas_alternativas(G, puerto_origen, puertos_alternativos):
    import folium
    from folium.plugins import MarkerCluster
    
    centro_lat = -35.6751
    centro_lon = -71.5430
    m = folium.Map(location=[centro_lat, centro_lon], zoom_start=5)
    marker_cluster = MarkerCluster().add_to(m)
    
    # Añadir puertos al mapa
    for node in G.nodes():
        lat = G.nodes[node].get('latitud')
        lon = G.nodes[node].get('longitud')
        nombre_puerto = G.nodes[node].get('nombre', '')
        codigo_puerto = node
        if pd.notna(lat) and pd.notna(lon):
            if codigo_puerto == puerto_origen:
                color = 'green'
            elif codigo_puerto in [pa[0] for pa in puertos_alternativos]:
                color = 'orange'
            else:
                color = 'blue'
            folium.Marker(
                location=[lat, lon],
                popup=f"{nombre_puerto} ({codigo_puerto})",
                icon=folium.Icon(color=color, icon='anchor', prefix='fa')
            ).add_to(marker_cluster)

    
    for puerto_alternativo, camino in puertos_alternativos:
        coordenadas_camino = []
        try:
            for codigo_puerto in camino :
                lat = G.nodes[str(codigo_puerto)].get('latitud')
                lon = G.nodes[str(codigo_puerto)].get('longitud')
                if pd.notna(lat) and pd.notna(lon) :
                    coordenadas_camino.append((lat, lon))
            if coordenadas_camino :
                folium.PolyLine(
                    locations=coordenadas_camino,
                    color='green',
                    weight=3,
                    opacity=0.8
                ).add_to(m)
        except Exception as e:
            print(f"Error al agregar camino de puertos: {e}")
    # Generar nombre aleatorio
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    grafo_html_path = os.path.join(settings.STATIC_ROOT,f'{random_string}.html')

    m.save(grafo_html_path)
    return random_string + '.html'   # Retornar el nombre del archivo HTML
def generar_infraestructura(puerto_origen: Puerto, puerto_destino: Puerto):
    """
    Genera la infraestructura para un puerto cerrado
    """
    G = construir_grafo()
    puerto_destino = puerto_destino.codigo
    puerto_origen = puerto_origen.codigo
    inhabilitar_puerto(G, str(puerto_destino))
    vecinos_origen = obtener_vecinos_puerto(G, str(puerto_origen))

    if puerto_destino in vecinos_origen:
        vecinos_origen.remove(puerto_destino)

    puertos_alternativos = []
    for puerto_alternativo in vecinos_origen:
        # Como son vecinos directos, la ruta es simplemente [puerto_origen, puerto_alternativo]
        camino = [puerto_origen, puerto_alternativo]
        puertos_alternativos.append((puerto_alternativo, camino))

    
    if puertos_alternativos:
        html = visualizar_rutas_alternativas(G, puerto_origen, puertos_alternativos)
        return html
    
    return None
def obtener_restricciones(id_bahia):
    """
    Obtiene las restricciones para una bahía específica.
    Lanza ServicioExternoError si hay algún problema con la solicitud HTTP.
    """
    url_restricciones = "https://orion.directemar.cl/sitport/back/users/consultaRestricciones"
    headers = {
        "Host": "orion.directemar.cl",
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36"
    }
    
    try:
        response = requests.post(url_restricciones, headers=headers, json={})
        response.raise_for_status()  # Lanza excepción si la respuesta no es 200 OK
        restricciones = response.json().get("recordsets", [[]])[0]
        if not restricciones:
            raise NoSeEncuentraInformacionError(f"No se encontraron restricciones para la bahía con ID {id_bahia}")
        return restricciones
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al obtener restricciones para la bahía {id_bahia}: {e}")
        raise ServicioExternoError(f"Error al comunicarse con el servicio de restricciones: {e}")


def obtener_nave(id_nave):
    """
    Obtiene la información de una nave por su ID.
    Lanza ServicioExternoError si hay algún problema con la solicitud HTTP.
    """
    base_url = "https://orion.directemar.cl/sitport/back/users/FichaNave"
    headers = {
        "Host": "orion.directemar.cl",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Origin": "https://sitport.directemar.cl",
        "Referer": "https://sitport.directemar.cl/"
    }

    url = f"{base_url}/{id_nave}"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lanza excepción si la respuesta no es 200 OK
        json_response = response.json()
        if json_response and 'datosnave' in json_response[0]:
            return json_response[0]['datosnave']
        else:
            raise NoSeEncuentraInformacionError(f"No se encontraron datos para la nave con ID {id_nave}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al obtener información de la nave {id_nave}: {e}")
        raise ServicioExternoError(f"Error al comunicarse con el servicio de naves: {e}")

def convertir_coordenada(coordenada_str):
    """
    Convierte una coordenada en formato '35 N' o '17 E' a formato decimal.
    """
    # Dividir el número de la dirección (N, S, E, W)
    valor, direccion = coordenada_str.split(" ")
    valor = float(valor)
    
    # Convertir dependiendo de la dirección
    if direccion in ['S', 'W']:
        valor = -valor
    
    return valor

def obtener_ubicacion_barco(imo):
    """
    Obtiene la información de un barco dado su IMO desde vesselfinder.com.
    Lanza ServicioExternoError si hay algún problema con la solicitud HTTP.
    """
    url = f'https://www.vesselfinder.com/?imo={imo}'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'es-419'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lanza excepción si la respuesta no es 200 OK
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_description = soup.find('meta', {'name': 'description'})

        if meta_description:
            info_barco = meta_description['content']
            
            # Suponiendo que el formato está en posiciones fijas en la cadena
            nombre_barco = info_barco.split(" ")[0] + " " + info_barco.split(" ")[1]
            
            # Obtener coordenadas
            latitud_str = info_barco.split(" ")[5] + " " + info_barco.split(" ")[6]
            longitud_str = info_barco.split(" ")[7] + " " + info_barco.split(" ")[8]
            
            # Convertir coordenadas a formato decimal
            latitud_decimal = convertir_coordenada(latitud_str)
            longitud_decimal = convertir_coordenada(longitud_str)
            
            return {
                'nombre_barco': nombre_barco,
                'latitud': latitud_decimal,
                'longitud': longitud_decimal,
                'description': info_barco
            }
        else:
            raise NoSeEncuentraInformacionError(f"No se encontró información del barco con IMO {imo}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al obtener información del barco con IMO {imo}: {e}")
        raise ServicioExternoError(f"Error al comunicarse con vesselfinder: {e}")
    

def get_naves_recalando():
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

        return barcos
    else:
        print("Error en la solicitud:", response.status_code)
        return None

def get_planificacion_san_antonio():
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
        return extracted_data
    else:
        print("Error al descargar la página")
        return None
    
#  Función para obtener el clima actual de una ciudad
def get_current_weather(lat, lon):
    api_key = os.getenv('OPEN_WEATHER_API_KEY')
    base_url = "https://api.openweathermap.org/data/3.0/onecall"
    url = f"{base_url}?lat={lat}&lon={lon}&appid={api_key}&units=metric"  # Temperaturas en Celsius
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_current_wave(lat, lon):
    # Endpoint de la API de Open-Meteo para obtener datos de oleaje
    url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,wave_direction,wave_period"

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
        wave_data = [
            {
                'hour': hour,
                'wave_height': wave_height,
                'wave_direction': wave_direction,
                'wave_period': wave_period
            }
            for hour, wave_height, wave_direction, wave_period in zip(hours, wave_heights, wave_directions, wave_periods)
        ]

        return wave_data
    else:
        print(f"Error al obtener datos de oleaje: {response}")
        raise Exception(f"Error al obtener datos de oleaje: {response.status_code}")
def obtener_sismos_chile():
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
                data.append({
                    'fecha_ubicacion': fecha_lugar,
                    'profundidad': profundidad,
                    'magnitud': magnitud
                })
        return data
    else:
        print(f"Error al obtener datos de sismos chilenos: {response.status_code}")
        return None


def obtener_datos_nave(nombre_nave):
    url = "https://maritimeoptima.com/public/vessels/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Accept-Language": "es-ES",
        "Accept": "*/*",
        # Añade más cabeceras si es necesario
    }
    
    # Realiza la solicitud GET
    response = requests.get(url, headers=headers, params={'q': nombre_nave})
    if response.status_code == 200:
        data = response.json()
        # Filtrar el resultado más relevante
        for item in data:
            if nombre_nave.isdigit() and len(nombre_nave) == 7 and item['url'].split('imo:')[1].split('/')[0] == nombre_nave:
                return {
                    'IMO': item['url'].split('imo:')[1].split('/')[0],
                    'nombre': item['name'],
                    'url': item['url']
                }


        # Si no hay coincidencia exacta, buscar nombre parcial
        for item in data:
            if item['name'].lower() == nombre_nave.lower().replace('%', ' '):
                return {
                    'IMO': item['url'].split('imo:')[1].split('/')[0],
                    'nombre': item['name'],
                    'url': item['url']
                }
            if nombre_nave.lower() in item['name'].lower().replace('%', ' '):
                return {
                    'IMO': item['url'].split('imo:')[1].split('/')[0],
                    'nombre': item['name'],
                    'url': item['url']
                }
        
        raise Exception(f"No se encontró el IMO para el nombre de la nave: {nombre_nave}")
    else:
        raise Exception(f"Error al obtener la información del IMO: {response.status_code}")
def obtener_sesion_id():
    url = "http://comext.aduana.cl:7001/ManifestacionMaritima/"
    response = requests.get(url)
    if response.status_code == 200:
        # Obtener el nuevo JSESSIONID de las cookies
        return response.cookies.get('JSESSIONID'), response.cookies.get('AWSALB')
    else:
        raise Exception(f"Error al obtener el JSESSIONID: {response.status_code}")
    
def consultar_datos_manifiesto(programacion):
    url = f"http://comext.aduana.cl:7001/ManifestacionMaritima/limpiarListaProgramacionNaves2.do;jsessionid=K9YWPJ8AS1irScQd0rOn2+7j"
    
    headers = {
        "Host": "comext.aduana.cl:7001",
        "Content-Length": "36",
        "Cache-Control": "max-age=0",
        "Accept-Language": "es-ES",
        "Upgrade-Insecure-Requests": "1",
        "Origin": "http://comext.aduana.cl:7001",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Referer": "http://comext.aduana.cl:7001/ManifestacionMaritima/",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    
    cookies = {
        "JSESSIONID": "K9YWPJ8AS1irScQd0rOn2+7j",
        "AWSALB": "Mr+kOsI07FrQU7Rl0gg9g3tG5oYzcYHwEh5vj9wJGv5NTDFuWeT3dAuWvQO5Q+aKNuq+JsHW6oDbRZIqP6vPMfmgI7CD6HiVb9o7lnXTaQF49FM1qoMiVmHEIBawi2bJbRULcDS4shzuZF2kZ9kztZUc41z8mnx71SI2ba43+MJvg9xDLTULUIcd+fyJSw=="
    }

    # Los datos deben estar codificados en formato URL
    data = "%7BactionForm.programacion%7D=" + str(programacion)

    # Hacer la solicitud POST
    response = requests.post(url, headers=headers, cookies=cookies, data=data)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        # Parsear el HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extraer los datos de la tabla (columna 1 a 11)
        fila = soup.select_one('table[border="1"] > tr:nth-of-type(3)')
        if fila:
            datos = []
            for i in range(1, 12):  # Extraer del td1 al td11
                celda = fila.select_one(f'td:nth-of-type({i}) > label')
                datos.append(celda.get_text(strip=True) if celda else "")
            
            # Crear un diccionario con los datos extraídos
            resultado = {
                "numero_programacion": datos[0],
                "nombre_puerto": datos[1],
                "nombre_nave": datos[2],
                "numero_viaje": datos[3],
                "nombre_agente": datos[4],
                "fecha_arribo_zarpe_estimado": datos[5],
                "fecha_registro_incorporacion": datos[6],
                "fecha_arribo_zarpe_efectivo": datos[7],
                "registro_arribo_zarpe_efectivo": datos[8],
                "tipo": datos[9],
                "estado": datos[10]
            }
            
            return resultado
        else:
            raise Exception("No se encontró la fila de datos en el HTML.")
    else:
        raise Exception(f"Error en la solicitud: {response.status_code}")



def obtener_datos_nave_por_nombre_o_imo(entrada):
    # Intentar obtener la URL a partir del nombre o IMO
    try:
        # Si el valor es IMO, usarlo directamente
        data = obtener_datos_nave(entrada)
        
        return scrape_nave_data(data['url'])
    
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def scrape_nave_data(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Accept-Language": "es-ES",
        "Accept": "*/*",
    }
    htmlData = requests.get(url, headers=headers)
    soup = BeautifulSoup(htmlData.text, 'html.parser')
    
    # Localizar la tabla con los datos
    tabla = soup.select_one('body > div:nth-of-type(1) > div:nth-of-type(2) > section:nth-of-type(4) > table')
    
    # Extraer los datos
    nave_data = {}
    if tabla:
        rows = tabla.find_all('tr')
        for row in rows:
            key = row.find('td').get_text(strip=True)
            value = row.find_all('td')[1].get_text(strip=True)
            nave_data[key] = value
    
    return nave_data