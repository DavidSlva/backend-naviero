from collection_manager.models import Sector

import requests
import csv
import pandas as pd
import psycopg2
from time import sleep
from datetime import datetime

# Definir las URLs y headers
url_bahias = "https://orion.directemar.cl/sitport/back/users/consultaBahias"
url_restricciones = "https://orion.directemar.cl/sitport/back/users/consultaRestricciones"

headers = {
    "Host": "orion.directemar.cl",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
    "Sec-Ch-Ua": '"Chromium";v="127", "Not)A;Brand";v="99"',
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Accept-Language": "es-419",
    "Sec-Ch-Ua-Mobile": "?0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.6533.100 Safari/537.36"
}

# Función para obtener la fecha de hoy en el formato adecuado
def obtener_fecha_hoy():
    return datetime.now().strftime('%d %b %Y')

# Solicitar información de bahías
def obtener_bahias():
    response = requests.post(url_bahias, headers=headers, json={})
    if response.status_code == 200:
        return response.json()['recordsets'][0]  # Primer conjunto de resultados
    else:
        print(f"Error al obtener bahías: {response.status_code}")
        return []
def cargar_sectores():
    bahias = obtener_bahias()

# Solicitar restricciones por bahía
def obtener_restricciones(id_bahia):
    response = requests.post(url_restricciones, headers=headers, json={})
    if response.status_code == 200:
        return response.json().get("recordsets", [[]])[0]
    else:
        print(f"Error al obtener restricciones para la bahía {id_bahia}: {response.status_code}")
        return []
