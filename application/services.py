import math
import os
import random
import string
import pandas as pd
import requests
import logging
from bs4 import BeautifulSoup
from geopy.distance import geodesic
from lxml import html
import networkx as nx
from application.models import Arista, Nodo
from backend import settings
from collection_manager.models import Puerto, Ruta
from interpreter.models import Registro
import pickle
from django.db import transaction, connection
import folium
from folium.plugins import MarkerCluster
import searoute as sr

# Configurar logger
logger = logging.getLogger(__name__)


class ServicioExternoError(Exception) :
    """Excepción personalizada para errores en servicios externos."""
    pass


class NoSeEncuentraInformacionError(Exception) :
    """Excepción personalizada cuando no se encuentra la información solicitada."""
    pass


def obtener_puertos() :
    """
    Obtener todas las rutas de los archivos de importaciones y exportaciones disponibles.
    """
    # Obtener todos los registros de la base de datos
    puertos = Puerto.objects.all()
    # Crear una lista de diccionarios donde cada diccionario representa un registro completo
    return puertos


def calcular_distancia(lat1, lon1, lat2, lon2) :
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


def construir_grafo() :
    # Verificar si el archivo del grafo existe
    if os.path.exists(GRAFO_FILE_PATH) :
        with open(GRAFO_FILE_PATH, 'rb') as f :
            G = pickle.load(f)
        print("Grafo cargado desde el archivo.")
        return G

    # Si no hay archivo, construimos el grafo desde la base de datos o desde cero
    if Nodo.objects.exists() and Arista.objects.exists() :
        G = nx.Graph()

        # Cargar nodos desde la base de datos
        nodos = Nodo.objects.all()
        for nodo in nodos :
            G.add_node(nodo.codigo, nombre=nodo.nombre, pais=nodo.pais, latitud=nodo.latitud, longitud=nodo.longitud)

        # Cargar aristas desde la base de datos
        aristas = Arista.objects.all()
        for arista in aristas :
            G.add_edge(arista.origen.codigo, arista.destino.codigo, distancia=arista.distancia)

        print("Grafo cargado desde la base de datos.")
    else :
        # Si no existe el grafo en la base de datos, crear un nuevo grafo
        G = nx.Graph()

        # Obtener registros de puertos y sus conexiones
        registros = Registro.objects.values_list('puerto_embarque_id', 'puerto_desembarque_id')

        # Obtener los puertos
        puertos = obtener_puertos()  # Asumimos que retorna un queryset de puertos

        # Añadir nodos (puertos) al grafo y almacenar los nodos en una lista para batch insert
        nodos_db = []
        for puerto in puertos :
            G.add_node(puerto.codigo, nombre=puerto.nombre, pais=puerto.pais, latitud=puerto.latitud,
                       longitud=puerto.longitud)
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
        for origen_codigo, destino_codigo in edges :
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
    with open(GRAFO_FILE_PATH, 'wb') as f :
        pickle.dump(G, f)

    print("Grafo guardado en el archivo.")
    return G


def inhabilitar_puerto(G, codigo_puerto) :
    """
    Inhabilita un puerto en la base de datos.
    """
    G.remove_node(codigo_puerto)
    return G


def obtener_vecinos_puerto(G, codigo_puerto) :
    """
    Obtiene los vecinos de un puerto en la base de datos.
    """
    if G.has_node(codigo_puerto) :
        vecinos = list(G.neighbors(codigo_puerto))
        vecinos = [int(v) for v in vecinos]
        return vecinos
    else :
        return []


def visualizar_rutas_alternativas(G, puerto_origen, puertos_alternativos) :
    # Función para extraer coordenadas de la ruta
    def extract_route_coordinates(route) :
        coordinates = []
        if not route :
            print("La ruta es None o está vacía.")
            return coordinates

        if 'geometry' in route and 'coordinates' in route['geometry'] :
            coords = route['geometry']['coordinates']
            for point in coords :
                coordinates.append([point[1], point[0]])  # Invertir el orden para obtener [latitud, longitud]
        else :
            print("La estructura de la ruta no es la esperada.")
        return coordinates

    try :
        # Crear el mapa centrado en Chile
        centro_lat = -35.6751
        centro_lon = -71.5430
        m = folium.Map(location=[centro_lat, centro_lon], zoom_start=5)
        marker_cluster = MarkerCluster().add_to(m)

        # Añadir puertos al mapa
        for node in G.nodes() :
            try :
                lat = G.nodes[node].get('latitud')
                lon = G.nodes[node].get('longitud')
                nombre_puerto = G.nodes[node].get('nombre', '')
                codigo_puerto = node
                if pd.notna(lat) and pd.notna(lon) :
                    color = 'green' if codigo_puerto == puerto_origen else 'orange' if codigo_puerto in [pa[0] for pa in
                                                                                                         puertos_alternativos] else 'blue'
                    folium.Marker(
                        location=[lat, lon],
                        popup=f"{nombre_puerto} ({codigo_puerto})",
                        icon=folium.Icon(color=color, icon='anchor', prefix='fa')
                    ).add_to(marker_cluster)
            except Exception as e :
                print(f"Error al añadir el puerto {node}: {e}")

        # Calcular y añadir rutas al mapa
        for puerto_alternativo, _ in puertos_alternativos :
            try :
                origin = [G.nodes[str(puerto_origen)]['longitud'], G.nodes[str(puerto_origen)]['latitud']]
                destination = [G.nodes[str(puerto_alternativo)]['longitud'],
                               G.nodes[str(puerto_alternativo)]['latitud']]
                print(f'Calculando ruta de {origin} a {destination}')

                # Calcular la ruta marítima
                route = sr.searoute(
                    origin,
                    destination,
                    append_orig_dest=True,
                    restrictions=['northwest'],
                    include_ports=True,
                    port_params={'only_terminals' : True}
                )

                route_coords = extract_route_coordinates(route)

                # Añadir la ruta al mapa
                if route_coords :
                    folium.PolyLine(
                        locations=route_coords,
                        color='green',
                        weight=3,
                        opacity=0.8
                    ).add_to(m)
                else :
                    print(
                        f"No se pudieron extraer las coordenadas de la ruta entre {puerto_origen} y {puerto_alternativo}")
            except Exception as e :
                print(f"Ocurrió un error al procesar la ruta entre {puerto_origen} y {puerto_alternativo}: {e}")

        # Guardar el mapa en un archivo HTML con un nombre aleatorio
        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        grafo_html_path = f'{random_string}.html'
        m.save(f"staticfiles/{grafo_html_path}")
        print(f"Mapa guardado en '{grafo_html_path}'")
        return grafo_html_path

    except Exception as e :
        print(f"Ocurrió un error general en la función: {e}")
        return None


def generar_infraestructura(puerto_origen: Puerto, puerto_destino: Puerto) :
    """
    Genera la infraestructura para un puerto cerrado
    """
    G = construir_grafo()
    puerto_destino = puerto_destino.codigo
    puerto_origen = puerto_origen.codigo
    print(puerto_destino, 'puerto origen')
    inhabilitar_puerto(G, str(puerto_destino))
    vecinos_origen = obtener_vecinos_puerto(G, str(puerto_origen))

    if puerto_destino in vecinos_origen :
        vecinos_origen.remove(puerto_destino)

    puertos_alternativos = []
    for puerto_alternativo in vecinos_origen :
        # Como son vecinos directos, la ruta es simplemente [puerto_origen, puerto_alternativo]
        camino = [puerto_origen, puerto_alternativo]
        puertos_alternativos.append((puerto_alternativo, camino))

    if puertos_alternativos :
        html = visualizar_rutas_alternativas(G, puerto_origen, puertos_alternativos)
        return html

    return None


def obtener_restricciones(id_bahia) :
    """
    Obtiene las restricciones para una bahía específica.
    Lanza ServicioExternoError si hay algún problema con la solicitud HTTP.
    """
    url_restricciones = "https://orion.directemar.cl/sitport/back/users/consultaRestricciones"
    headers = {
        "Host" : "orion.directemar.cl",
        "Accept" : "application/json, text/plain, */*",
        "Content-Type" : "application/json",
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36"
    }

    try :
        response = requests.post(url_restricciones, headers=headers, json={})
        response.raise_for_status()  # Lanza excepción si la respuesta no es 200 OK
        restricciones = response.json().get("recordsets", [[]])[0]
        if not restricciones :
            raise NoSeEncuentraInformacionError(f"No se encontraron restricciones para la bahía con ID {id_bahia}")
        return restricciones
    except requests.exceptions.RequestException as e :
        logger.error(f"Error al obtener restricciones para la bahía {id_bahia}: {e}")
        raise ServicioExternoError(f"Error al comunicarse con el servicio de restricciones: {e}")


def obtener_nave(id_nave) :
    """
    Obtiene la información de una nave por su ID.
    Lanza ServicioExternoError si hay algún problema con la solicitud HTTP.
    """
    base_url = "https://orion.directemar.cl/sitport/back/users/FichaNave"
    headers = {
        "Host" : "orion.directemar.cl",
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Origin" : "https://sitport.directemar.cl",
        "Referer" : "https://sitport.directemar.cl/"
    }

    url = f"{base_url}/{id_nave}"

    try :
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lanza excepción si la respuesta no es 200 OK
        json_response = response.json()
        if json_response and 'datosnave' in json_response[0] :
            return json_response[0]['datosnave']
        else :
            raise NoSeEncuentraInformacionError(f"No se encontraron datos para la nave con ID {id_nave}")
    except requests.exceptions.RequestException as e :
        logger.error(f"Error al obtener información de la nave {id_nave}: {e}")
        raise ServicioExternoError(f"Error al comunicarse con el servicio de naves: {e}")


def convertir_coordenada(coordenada_str) :
    """
    Convierte una coordenada en formato '35 N' o '17 E' a formato decimal.
    """
    # Dividir el número de la dirección (N, S, E, W)
    valor, direccion = coordenada_str.split(" ")
    valor = float(valor)

    # Convertir dependiendo de la dirección
    if direccion in ['S', 'W'] :
        valor = -valor

    return valor


def obtener_ubicacion_barco(imo) :
    """
    Obtiene la información de un barco dado su IMO desde vesselfinder.com.
    Lanza ServicioExternoError si hay algún problema con la solicitud HTTP.
    """
    url = f'https://www.vesselfinder.com/?imo={imo}'

    headers = {
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36',
        'Accept' : 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language' : 'es-419'
    }

    try :
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lanza excepción si la respuesta no es 200 OK
        soup = BeautifulSoup(response.text, 'html.parser')
        meta_description = soup.find('meta', {'name' : 'description'})

        if meta_description :
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
                'nombre_barco' : nombre_barco,
                'latitud' : latitud_decimal,
                'longitud' : longitud_decimal,
                'description' : info_barco
            }
        else :
            raise NoSeEncuentraInformacionError(f"No se encontró información del barco con IMO {imo}")
    except requests.exceptions.RequestException as e :
        logger.error(f"Error al obtener información del barco con IMO {imo}: {e}")
        raise ServicioExternoError(f"Error al comunicarse con vesselfinder: {e}")


def get_naves_recalando() :
    url = "https://orion.directemar.cl/sitport/back/users/consultaNaveRecalando"
    headers = {
        "Host" : "orion.directemar.cl",
        "Content-Length" : "2",
        "Sec-Ch-Ua" : '"Not(A:Brand";v="24", "Chromium";v="122"',
        "Accept" : "application/json, text/plain, */*",
        "Content-Type" : "application/json",
        "Sec-Ch-Ua-Mobile" : "?0",
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36"
    }
    data = {}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200 :
        json_response = response.json()
        barcos = json_response.get("recordsets", [[]])[0]

        return barcos
    else :
        print("Error en la solicitud:", response.status_code)
        return None


def get_planificacion_san_antonio() :
    # URL de la página para descargar la metadata
    url = 'https://gessup.puertosanantonio.com/Planificaciones/general.aspx'

    # Encabezados para la solicitud
    headers = {
        'Host' : 'gessup.puertosanantonio.com',
        'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36',
        'Accept-Language' : 'es-ES',
        'Accept-Encoding' : 'gzip, deflate, br',
        'Connection' : 'keep-alive'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200 :
        tree = html.fromstring(response.content)
        data = tree.xpath('//*[@id="ctl00_ContentPlaceHolder1_GridView_Lista"]')

        headers = ['E.T.A.', 'Agencia', 'Nave', 'Eslora', 'Terminal', 'Emp. muellaje', 'Carga', 'Detalle', 'Cantidad',
                   'Operación']
        extracted_data = []
        for table in data :
            for row in table.xpath('.//tr') :
                row_data = [cell.text_content().strip() for cell in row.xpath('.//td')]
                if len(row_data) > 0 :
                    extracted_data.append(row_data)
        return extracted_data
    else :
        print("Error al descargar la página")
        return None


#  Función para obtener el clima actual de una ciudad
def get_current_weather(lat, lon) :
    api_key = os.getenv('OPEN_WEATHER_API_KEY')
    base_url = "https://api.openweathermap.org/data/3.0/onecall"
    url = f"{base_url}?lat={lat}&lon={lon}&appid={api_key}&units=metric"  # Temperaturas en Celsius
    response = requests.get(url)
    if response.status_code == 200 :
        return response.json()
    else :
        return None


def get_current_wave(lat, lon) :
    # Endpoint de la API de Open-Meteo para obtener datos de oleaje
    url = f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}&hourly=wave_height,wave_direction,wave_period"

    # Realizar la solicitud HTTP para obtener los datos
    response = requests.get(url)

    # Verifica si la solicitud fue exitosa
    if response.status_code == 200 :
        data = response.json()  # Obtener los datos en formato JSON

        # Extraer los datos de oleaje
        hours = data['hourly']['time']
        wave_heights = data['hourly']['wave_height']
        wave_directions = data['hourly']['wave_direction']
        wave_periods = data['hourly']['wave_period']

        # Combinar los datos en un DataFrame
        wave_data = [
            {
                'hour' : hour,
                'wave_height' : wave_height,
                'wave_direction' : wave_direction,
                'wave_period' : wave_period
            }
            for hour, wave_height, wave_direction, wave_period in
            zip(hours, wave_heights, wave_directions, wave_periods)
        ]

        return wave_data
    else :
        print(f"Error al obtener datos de oleaje: {response}")
        raise Exception(f"Error al obtener datos de oleaje: {response.status_code}")


def obtener_sismos_chile() :
    # URL de la página de sismología
    url = "https://www.sismologia.cl/"

    # Realizar la solicitud HTTP para obtener el contenido de la página
    response = requests.get(url)

    # Verifica si la solicitud fue exitosa
    if response.status_code == 200 :
        html_content = response.text

        # Analizar el contenido HTML usando BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')

        # Encontrar la tabla con la clase 'sismologia'
        table = soup.find('table', class_='sismologia')

        # Crear una lista para almacenar los datos extraídos
        data = []

        # Iterar sobre las filas de la tabla
        for row in table.find_all('tr')[1 :] :  # Saltar la primera fila de encabezados
            columns = row.find_all('td')
            if columns :
                fecha_lugar = columns[0].get_text(strip=True).replace('\n', ' ')
                profundidad = columns[1].get_text(strip=True)
                magnitud = columns[2].get_text(strip=True)
                data.append({
                    'fecha_ubicacion' : fecha_lugar,
                    'profundidad' : profundidad,
                    'magnitud' : magnitud
                })
        return data
    else :
        print(f"Error al obtener datos de sismos chilenos: {response.status_code}")
        return None


def obtener_datos_nave(nombre_nave) :
    url = "https://maritimeoptima.com/public/vessels/search"
    headers = {
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Accept-Language" : "es-ES",
        "Accept" : "*/*",
        # Añade más cabeceras si es necesario
    }

    # Realiza la solicitud GET
    response = requests.get(url, headers=headers, params={'q' : nombre_nave})
    if response.status_code == 200 :
        data = response.json()
        # Filtrar el resultado más relevante
        for item in data :
            if nombre_nave.isdigit() and len(nombre_nave) == 7 and item['url'].split('imo:')[1].split('/')[
                0] == nombre_nave :
                return {
                    'IMO' : item['url'].split('imo:')[1].split('/')[0],
                    'nombre' : item['name'],
                    'url' : item['url']
                }

        # Si no hay coincidencia exacta, buscar nombre parcial
        for item in data :
            if item['name'].lower() == nombre_nave.lower().replace('%', ' ') :
                return {
                    'IMO' : item['url'].split('imo:')[1].split('/')[0],
                    'nombre' : item['name'],
                    'url' : item['url']
                }
            if nombre_nave.lower() in item['name'].lower().replace('%', ' ') :
                return {
                    'IMO' : item['url'].split('imo:')[1].split('/')[0],
                    'nombre' : item['name'],
                    'url' : item['url']
                }

        raise Exception(f"No se encontró el IMO para el nombre de la nave: {nombre_nave}")
    else :
        raise Exception(f"Error al obtener la información del IMO: {response.status_code}")


def obtener_sesion_id() :
    url = "http://comext.aduana.cl:7001/ManifestacionMaritima/"
    response = requests.get(url)
    if response.status_code == 200 :
        # Obtener el nuevo JSESSIONID de las cookies
        return response.cookies.get('JSESSIONID'), response.cookies.get('AWSALB')
    else :
        raise Exception(f"Error al obtener el JSESSIONID: {response.status_code}")


def consultar_datos_manifiesto(programacion) :
    url = f"http://comext.aduana.cl:7001/ManifestacionMaritima/limpiarListaProgramacionNaves2.do;jsessionid=K9YWPJ8AS1irScQd0rOn2+7j"

    headers = {
        "Host" : "comext.aduana.cl:7001",
        "Content-Length" : "36",
        "Cache-Control" : "max-age=0",
        "Accept-Language" : "es-ES",
        "Upgrade-Insecure-Requests" : "1",
        "Origin" : "http://comext.aduana.cl:7001",
        "Content-Type" : "application/x-www-form-urlencoded",
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Accept" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Referer" : "http://comext.aduana.cl:7001/ManifestacionMaritima/",
        "Accept-Encoding" : "gzip, deflate, br",
        "Connection" : "keep-alive"
    }

    cookies = {
        "JSESSIONID" : "K9YWPJ8AS1irScQd0rOn2+7j",
        "AWSALB" : "Mr+kOsI07FrQU7Rl0gg9g3tG5oYzcYHwEh5vj9wJGv5NTDFuWeT3dAuWvQO5Q+aKNuq+JsHW6oDbRZIqP6vPMfmgI7CD6HiVb9o7lnXTaQF49FM1qoMiVmHEIBawi2bJbRULcDS4shzuZF2kZ9kztZUc41z8mnx71SI2ba43+MJvg9xDLTULUIcd+fyJSw=="
    }

    # Los datos deben estar codificados en formato URL
    data = "%7BactionForm.programacion%7D=" + str(programacion)

    # Hacer la solicitud POST
    response = requests.post(url, headers=headers, cookies=cookies, data=data)

    # Verificar si la solicitud fue exitosa
    if response.status_code == 200 :
        # Parsear el HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extraer los datos de la tabla (columna 1 a 11)
        fila = soup.select_one('table[border="1"] > tr:nth-of-type(3)')
        if fila :
            datos = []
            for i in range(1, 12) :  # Extraer del td1 al td11
                celda = fila.select_one(f'td:nth-of-type({i}) > label')
                datos.append(celda.get_text(strip=True) if celda else "")

            # Crear un diccionario con los datos extraídos
            resultado = {
                "numero_programacion" : datos[0],
                "nombre_puerto" : datos[1],
                "nombre_nave" : datos[2],
                "numero_viaje" : datos[3],
                "nombre_agente" : datos[4],
                "fecha_arribo_zarpe_estimado" : datos[5],
                "fecha_registro_incorporacion" : datos[6],
                "fecha_arribo_zarpe_efectivo" : datos[7],
                "registro_arribo_zarpe_efectivo" : datos[8],
                "tipo" : datos[9],
                "estado" : datos[10]
            }

            return resultado
        else :
            raise Exception("No se encontró la fila de datos en el HTML.")
    else :
        raise Exception(f"Error en la solicitud: {response.status_code}")


def obtener_datos_nave_por_nombre_o_imo(entrada) :
    # Intentar obtener la URL a partir del nombre o IMO
    try :
        # Si el valor es IMO, usarlo directamente
        data = obtener_datos_nave(entrada)

        return scrape_nave_data(data['url'])

    except Exception as e :
        raise Exception(f"Error: {str(e)}")


def scrape_nave_data(url) :
    headers = {
        "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36",
        "Accept-Language" : "es-ES",
        "Accept" : "*/*",
    }
    htmlData = requests.get(url, headers=headers)
    soup = BeautifulSoup(htmlData.text, 'html.parser')

    # Localizar la tabla con los datos
    tabla = soup.select_one('body > div:nth-of-type(1) > div:nth-of-type(2) > section:nth-of-type(4) > table')

    # Extraer los datos
    nave_data = {}
    if tabla :
        rows = tabla.find_all('tr')
        for row in rows :
            key = row.find('td').get_text(strip=True)
            value = row.find_all('td')[1].get_text(strip=True)
            nave_data[key] = value

    return nave_data


def calcular_distancia_total(coordenadas) :
    """
    Calcula la distancia total en kilómetros a partir de una lista de coordenadas.
    """
    distancia_total = 0
    for i in range(len(coordenadas) - 1) :
        punto_inicio = coordenadas[i]
        punto_fin = coordenadas[i + 1]
        distancia_total += geodesic(punto_inicio, punto_fin).kilometers
    return distancia_total


def create_rutas(puerto_origen, puertos_destino) :
    """
    Crea rutas entre los puertos origen y destino y calcula la distancia total para cada ruta.
    """
    rutas = []
    distancias_totales = []

    # Verificar que el puerto origen tenga latitud y longitud válidas
    if puerto_origen.latitud is None or puerto_origen.longitud is None :
        return rutas, distancias_totales  # Retorna vacío si el puerto origen no tiene coordenadas

    # Preparar la tupla de coordenadas del origen
    origin = (puerto_origen.latitud, puerto_origen.longitud)

    for puerto_destino in puertos_destino :
        # Verificar que el puerto destino tenga latitud y longitud válidas
        try :
            lat = puerto_destino.latitud
            lon = puerto_destino.longitud

            if (
                    lat is None
                    or lon is None
                    or (isinstance(lat, float) and math.isnan(lat))
                    or (isinstance(lon, float) and math.isnan(lon))
            ) :
                continue

            # Preparar la tupla de coordenadas del destino
            destination = (puerto_destino.latitud, puerto_destino.longitud)

            # Llamada a searoute con las coordenadas en el formato correcto
            ruta = sr.searoute(
                origin,
                destination,
                append_orig_dest=True,
                restrictions=['northwest'],
                include_ports=True,
                port_params={'only_terminals' : True}
            )
            # Extraer las coordenadas de la ruta generada
            coordenadas = [(punto[0], punto[1]) for punto in ruta['geometry']['coordinates']]

            # Calcular la distancia total de la ruta
            distancia_total = calcular_distancia_total(coordenadas)

            # Crear o obtener el objeto Ruta
            Ruta.objects.get_or_create(origen=puerto_origen, destino=puerto_destino, distancia=distancia_total)

            rutas.append(ruta)
            distancias_totales.append(distancia_total)
        except Exception as e :
            print(f"Error al calcular la ruta entre {puerto_origen} y {puerto_destino}: {e}")

    return rutas, distancias_totales



def get_best_routes(origin_puerto, destination_puertos):
    """
    Calcula las rutas más cortas desde el puerto de origen a los puertos de destino utilizando el algoritmo de Dijkstra de NetworkX.

    :param origin_puerto: Objeto Puerto que representa el puerto de origen.
    :param destination_puertos: QuerySet de objetos Puerto que representan los puertos de destino.
    :return: Lista de diccionarios con 'destination', 'total_cost' y opcionalmente 'path'.
    """
    try:
        # Crear un grafo dirigido o no dirigido según tus necesidades
        # Aquí asumimos que las rutas son bidireccionales; si no, cambia a nx.DiGraph()
        G = nx.Graph()

        # Obtener todas las rutas de la base de datos
        rutas = Ruta.objects.all()

        # Agregar aristas al grafo con las distancias como pesos
        for ruta in rutas:
            G.add_edge(
                ruta.origen.codigo,
                ruta.destino.codigo,
                weight=ruta.distancia
            )

        # Obtener los códigos de los puertos de destino
        destination_codes = [puerto.codigo for puerto in destination_puertos]

        best_routes = []

        # Calcular la ruta más corta a cada destino
        for dest_code in destination_codes:
            if dest_code == origin_puerto.codigo:
                # La distancia al mismo puerto es 0
                best_routes.append({
                    'destination': dest_code,
                    'total_cost': 0,
                    'path': [dest_code],
                })
                continue

            try:
                # Obtener la distancia más corta
                total_cost = nx.dijkstra_path_length(G, origin_puerto.codigo, dest_code, weight='weight')

                # Obtener el camino más corto (opcional)
                path = nx.dijkstra_path(G, origin_puerto.codigo, dest_code, weight='weight')

                best_routes.append({
                    'destination': dest_code,
                    'total_cost': total_cost,
                    'path': path,  # Puedes eliminar esta línea si no necesitas el camino completo
                })
            except nx.NetworkXNoPath:
                # No existe una ruta entre el origen y el destino
                best_routes.append({
                    'destination': dest_code,
                    'total_cost': None,  # O algún indicador de que no hay ruta
                    'path': None,
                })

        return best_routes

    except Exception as e:
        logger.error(f"Error al calcular las rutas más cortas con NetworkX: {e}")
        raise


def get_best_route(origin_puerto, destination_puertos) :
    """
    Calcula la ruta más corta desde el puerto de origen a los puertos de destino utilizando el algoritmo de Dijkstra de NetworkX.

    :param origin_puerto: Objeto Puerto que representa el puerto de origen.
    :param destination_puertos: QuerySet de objetos Puerto que representan los puertos de destino.
    :return: Diccionario con 'destination', 'total_cost' y 'path' de la ruta más corta.
    """
    try :
        # Crear un grafo no dirigido; si tus rutas son unidireccionales, utiliza nx.DiGraph()
        G = nx.Graph()

        # Obtener todas las rutas de la base de datos
        rutas = Ruta.objects.all()

        # Agregar aristas al grafo con las distancias como pesos
        for ruta in rutas :
            G.add_edge(
                ruta.origen.codigo,
                ruta.destino.codigo,
                weight=ruta.distancia
            )

        # Obtener los códigos de los puertos de destino
        destination_codes = set(puerto.codigo for puerto in destination_puertos)

        # Calcular todas las distancias y caminos desde el origen
        # Esto es más eficiente que calcular por separado para cada destino
        lengths, paths = nx.single_source_dijkstra(G, origin_puerto.codigo, weight='weight')

        # Filtrar solo los destinos que están en destination_codes
        # y que tienen una ruta disponible
        available_routes = {
            dest : (lengths[dest], paths[dest])
            for dest in destination_codes
            if dest in lengths
        }

        if not available_routes :
            # No hay rutas disponibles hacia ninguno de los destinos
            return {
                'destination' : None,
                'total_cost' : None,
                'path' : None,
            }

        # Encontrar la ruta con la menor distancia total
        min_dest, (min_cost, min_path) = min(
            available_routes.items(),
            key=lambda item : item[1][0]
        )

        return {
            'destination' : min_dest,
            'total_cost' : min_cost,
            'path' : min_path,
        }

    except Exception as e :
        logger.error(f"Error al calcular la ruta más corta con NetworkX: {e}")
        raise

def get_best_route_metaheuristic(origin_puerto, destination_puertos):
    """
    Calcula la ruta óptima desde el puerto de origen a los puertos de destino utilizando
    el algoritmo de Optimización por Colonia de Hormigas (Ant Colony Optimization - ACO).
    
    :param origin_puerto: Objeto Puerto que representa el puerto de origen.
    :param destination_puertos: QuerySet de objetos Puerto que representan los puertos de destino.
    :return: Diccionario con 'destination', 'total_cost' y 'path' de la mejor ruta encontrada.
    """
    try:
        # Crear un grafo no dirigido
        G = nx.Graph()

        # Obtener todas las rutas de la base de datos
        rutas = Ruta.objects.all()

        # Agregar aristas al grafo con las distancias como pesos
        for ruta in rutas:
            G.add_edge(
                ruta.origen.codigo,
                ruta.destino.codigo,
                weight=ruta.distancia
            )

        # Obtener los códigos de los puertos de destino
        destination_codes = set(puerto.codigo for puerto in destination_puertos)

        # Obtener las probabilidades de falla de los puertos
        puertos = Puerto.objects.all()
        port_failure_probabilities = {}
        for puerto in puertos:
            failure_probability = calcular_probabilidad_falla_puerto(puerto)
            port_failure_probabilities[puerto.codigo] = failure_probability

        # Parámetros del ACO
        num_ants = 10
        num_iterations = 100
        evaporation_rate = 0.5
        alpha = 1  # Importancia de la feromona
        beta = 2   # Importancia de la heurística (1/distancia)

        # Inicializar las feromonas en todas las aristas
        pheromone = {}
        for edge in G.edges():
            pheromone[edge] = 1.0

        best_path = None
        best_cost = float('inf')
        best_destination = None

        for iteration in range(num_iterations):
            paths = []
            for ant in range(num_ants):
                # Cada hormiga construye una solución
                path = [origin_puerto.codigo]
                current_node = origin_puerto.codigo
                visited = set()
                visited.add(current_node)

                while current_node not in destination_codes:
                    neighbors = [n for n in G.neighbors(current_node) if n not in visited]
                    if not neighbors:
                        break  # No hay camino disponible
                    probabilities = []
                    for neighbor in neighbors:
                        edge = (current_node, neighbor)
                        edge_pheromone = pheromone.get(edge, 1.0)
                        edge_weight = G[current_node][neighbor]['weight']
                        heuristic = (1.0 / edge_weight) ** beta

                        # Considerar la probabilidad de falla del puerto
                        failure_penalty = (1 - port_failure_probabilities.get(neighbor, 0.0))
                        probability = (edge_pheromone ** alpha) * heuristic * failure_penalty
                        probabilities.append(probability)
                    # Normalizar las probabilidades
                    total = sum(probabilities)
                    probabilities = [p / total for p in probabilities]

                    # Seleccionar el siguiente nodo basado en las probabilidades
                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current_node = next_node

                # Evaluar el costo del camino
                if current_node in destination_codes:
                    total_cost = 0
                    for i in range(len(path) - 1):
                        total_cost += G[path[i]][path[i+1]]['weight']
                        # Añadir penalización por probabilidad de falla
                        total_cost += port_failure_probabilities.get(path[i+1], 0.0) * 1000  # Peso ajustable
                    paths.append((path, total_cost, current_node))
                    # Actualizar la mejor solución
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_path = path
                        best_destination = current_node

            # Actualizar las feromonas
            for edge in pheromone:
                pheromone[edge] *= (1 - evaporation_rate)  # Evaporación

            for path_info in paths:
                path, total_cost, dest = path_info
                # Depositar feromonas inversamente proporcional al costo
                deposit = 1.0 / total_cost
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    pheromone[edge] += deposit

        return {
            'destination': best_destination,
            'total_cost': best_cost,
            'path': best_path,
        }

    except Exception as e:
        logger.error(f"Error al calcular la ruta con ACO: {e}")
        raise

def get_best_route_cplex(origin_puerto, destination_puertos):
    """
    Calcula la ruta óptima desde el puerto de origen a los puertos de destino utilizando
    el modelado de optimización con CPLEX, considerando las variables de los metadatos y amenazas,
    y las condiciones del usuario como restricciones.

    :param origin_puerto: Objeto Puerto que representa el puerto de origen.
    :param destination_puertos: QuerySet de objetos Puerto que representan los puertos de destino.
    :return: Diccionario con 'destination', 'total_cost' y 'path' de la mejor ruta encontrada.
    """
    try:
        from docplex.mp.model import Model

        # Crear un modelo de optimización
        mdl = Model(name='RouteOptimization')

        # Crear un grafo no dirigido
        G = nx.Graph()

        # Obtener todas las rutas de la base de datos
        rutas = Ruta.objects.all()

        # Agregar aristas al grafo con las distancias como pesos
        for ruta in rutas:
            G.add_edge(
                ruta.origen.codigo,
                ruta.destino.codigo,
                weight=ruta.distancia
            )

        # Obtener los códigos de los puertos de destino
        destination_codes = set(puerto.codigo for puerto in destination_puertos)

        # Variables de decisión: x[i,j] = 1 si la ruta (i,j) está en el camino óptimo
        x = mdl.binary_var_dict(G.edges(), name='x')

        # Función objetivo: minimizar la suma de las distancias de las rutas seleccionadas
        mdl.minimize(mdl.sum(G[i][j]['weight'] * x[(i, j)] for i, j in G.edges()))

        # Restricciones de flujo
        for node in G.nodes():
            flujo_entrada = mdl.sum(x[(i, node)] for i in G.neighbors(node) if (i, node) in x)
            flujo_salida = mdl.sum(x[(node, j)] for j in G.neighbors(node) if (node, j) in x)
            if node == origin_puerto.codigo:
                mdl.add_constraint(flujo_salida - flujo_entrada == 1)
            elif node in destination_codes:
                mdl.add_constraint(flujo_entrada - flujo_salida == 1)
            else:
                mdl.add_constraint(flujo_entrada - flujo_salida == 0)

        # Obtener las probabilidades de falla de los puertos
        puertos = Puerto.objects.all()
        port_failure_probabilities = {}
        for puerto in puertos:
            failure_probability = calcular_probabilidad_falla_puerto(puerto)
            port_failure_probabilities[puerto.codigo] = failure_probability

        # Condiciones del usuario como restricciones
        failure_threshold = 0.5  # Por ejemplo, evitar puertos con más del 50% de probabilidad de falla
        for node in G.nodes():
            if port_failure_probabilities.get(node, 0.0) > failure_threshold:
                # Eliminar todas las aristas conectadas a este nodo
                for neighbor in list(G.neighbors(node)):
                    if (node, neighbor) in x:
                        mdl.add_constraint(x[(node, neighbor)] == 0)
                    if (neighbor, node) in x:
                        mdl.add_constraint(x[(neighbor, node)] == 0)

        # Resolver el modelo
        solution = mdl.solve()

        if solution:
            # Obtener la ruta óptima
            path_edges = [edge for edge in x if x[edge].solution_value > 0.5]
            # Construir el camino a partir de las aristas seleccionadas
            path_graph = nx.Graph()
            path_graph.add_edges_from(path_edges)
            paths = []
            for dest in destination_codes:
                try:
                    path = nx.shortest_path(path_graph, source=origin_puerto.codigo, target=dest)
                    paths.append((path, dest))
                except nx.NetworkXNoPath:
                    continue

            if not paths:
                return {
                    'destination': None,
                    'total_cost': None,
                    'path': None,
                }

            # Seleccionar el camino más corto
            best_path_info = min(paths, key=lambda p: sum(G[p[0][i]][p[0][i+1]]['weight'] for i in range(len(p[0])-1)))
            best_path, best_destination = best_path_info
            total_cost = sum(G[best_path[i]][best_path[i+1]]['weight'] for i in range(len(best_path)-1))
            return {
                'destination': best_destination,
                'total_cost': total_cost,
                'path': best_path,
            }
        else:
            logger.error("No se encontró solución al modelo de optimización con CPLEX")
            return {
                'destination': None,
                'total_cost': None,
                'path': None,
            }

    except Exception as e:
        logger.error(f"Error al calcular la ruta con CPLEX: {e}")
        raise


def calcular_probabilidad_falla_puerto(puerto):
    """
    Calcula la probabilidad de falla del puerto basado en factores como sismos, lluvias, oleaje y restricciones.
    """
    try:
        if puerto.latitud and puerto.longitud:
            weather = get_current_weather(puerto.latitud, puerto.longitud)
            wave_data = get_current_wave(puerto.latitud, puerto.longitud)
            sismos = obtener_sismos_chile()

            hourly_data = weather.get('hourly', {})
            maxWaveHeight = max(oleaje.get('wave_height', 0) for oleaje in wave_data)

            # Distancia máxima para sismos
            distancia_maxima_km = 500
            ubicacion_puerto = (puerto.latitud, puerto.longitud)
            probabilidadFallaSismo_final = 0

            # Cálculo de las probabilidades para Bahía
            if puerto.sector:
                bahia = puerto.sector.id
                restricciones = obtener_restricciones(bahia)

                restricciones_filtradas = [
                    restriccion for restriccion in restricciones
                    if restriccion['bahia'] == bahia
                ]
                probabilidadFallaBahia = 1.0 if restricciones_filtradas else 0.0
            else:
                probabilidadFallaBahia = 0.0

            # Cálculo de la probabilidad para Sismos
            for sismo in sismos:
                epicentro = (sismo.get('latitud'), sismo.get('longitud'))
                try:
                    distancia = geodesic(epicentro, ubicacion_puerto).kilometers
                    esta_cerca = distancia <= distancia_maxima_km
                except ValueError:
                    esta_cerca = False

                if esta_cerca:
                    magnitud = sismo.get('magnitud')
                    if magnitud is None:
                        probabilidadFallaSismo = 0.0
                    elif magnitud <= 5:
                        probabilidadFallaSismo = 0.0
                    elif magnitud >= 7:
                        probabilidadFallaSismo = 1.0
                    else:
                        probabilidadFallaSismo = ((magnitud - 5)/(7 - 5))
                    probabilidadFallaSismo_final = max(probabilidadFallaSismo_final, probabilidadFallaSismo)

            # Cálculo de la probabilidad para Lluvia
            precipTotal = sum(hora.get('rain', {}).get('1h', 0) for hora in hourly_data)
            precipMax = min(precipTotal, 150)
            probabilidadFallaLluvia = (precipMax / 150)

            # Cálculo de la probabilidad para el Oleaje
            if maxWaveHeight >= 1.8:
                probabilidadFallaOleaje = 1.0
            elif maxWaveHeight >= 1.5:
                probabilidadFallaOleaje = ((maxWaveHeight - 1.5) / 0.3)
            else:
                probabilidadFallaOleaje = 0.0

            # Combinar las probabilidades
            probabilidadFallaTotal = max(
                probabilidadFallaBahia,
                probabilidadFallaSismo_final,
                probabilidadFallaLluvia,
                probabilidadFallaOleaje
            )

            return probabilidadFallaTotal

        else:
            return 0.0

    except Exception as e:
        logger.error(f"Error al calcular la probabilidad de falla del puerto {puerto.codigo}: {e}")
        return 0.0
