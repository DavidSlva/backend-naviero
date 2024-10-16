import os
import requests
import logging
from bs4 import BeautifulSoup
from lxml import html


# Configurar logger
logger = logging.getLogger(__name__)

class ServicioExternoError(Exception):
    """Excepción personalizada para errores en servicios externos."""
    pass

class NoSeEncuentraInformacionError(Exception):
    """Excepción personalizada cuando no se encuentra la información solicitada."""
    pass


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