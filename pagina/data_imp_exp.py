import requests
import os
import rarfile  # Importar rarfile para descomprimir
from lxml import html
from time import sleep
from requests.exceptions import RequestException
import concurrent.futures
from tqdm import tqdm  

# Función para obtener los enlaces formateados de importaciones
def obtener_enlaces_importaciones():
    url_importaciones = "https://datos.gob.cl/dataset/registro-de-importacion-2024/resource/c4941a06-34b9-47d0-a477-ff2aa021e442?inner_span=True"
    
    # Realizamos la solicitud HTTP para obtener el HTML
    response = requests.get(url_importaciones)
    page_content = response.content
    
    # Parseamos el contenido HTML
    tree = html.fromstring(page_content)
    
    # XPath que selecciona los <a> dentro de <ul>
    xpath_query = '//*[@id="content"]/div[3]/aside/section/ul/li/a'
    
    # Extraemos todos los elementos <a> y sus atributos href y title
    links = tree.xpath(xpath_query)
    
    # Código arbitrario para reemplazar el año en la URL
    arbitrary_code_importaciones = "096c3946-657e-420f-ae74-2337c00b5ba2"
    
    # Lista para almacenar los enlaces formateados
    formatted_urls_importaciones = []
    
    # Procesamos cada <a>
    for link in links:
        href = link.get('href')  # Extraemos el href
        title = link.get('title')  # Extraemos el title para obtener el mes y la parte

        # Saltar si es metadata
        if 'Metadata' in title:
            continue

        # Extraemos el ID de la etiqueta 'a' desde el href
        resource_id = href.split('/resource/')[1].split('?')[0]

        # Extraemos la información del mes y la parte del title
        title_parts = title.split(' ')
        month = title_parts[2].lower()  # El mes (ejemplo: 'enero')
        part_info = title_parts[-1]  # La parte (ejemplo: '1/5')
        part_number = part_info.split('/')[0]  # Número de parte (ejemplo: '1')

        # Formamos la nueva URL
        formatted_url = f"https://datos.gob.cl/dataset/{arbitrary_code_importaciones}/resource/{resource_id}/download/importaciones-{month}-2024.part0{part_number}.rar"

        # Añadimos la URL generada a la lista
        formatted_urls_importaciones.append(formatted_url)

    return formatted_urls_importaciones

# Función para obtener los enlaces formateados de exportaciones
def obtener_enlaces_exportaciones():
    url_exportaciones = "https://datos.gob.cl/dataset/registro-de-exportaciones-2024/resource/118906a7-a117-4aa0-829a-a7a9633e3acd"
    
    # Realizamos la solicitud HTTP para obtener el HTML
    response = requests.get(url_exportaciones)
    page_content = response.content
    
    # Parseamos el contenido HTML
    tree = html.fromstring(page_content)
    
    # XPath que selecciona los <a> dentro de <ul>
    xpath_query = '//*[@id="content"]/div[3]/aside/section/ul/li/a'
    
    # Extraemos todos los elementos <a> y sus atributos href y title
    links = tree.xpath(xpath_query)
    
    # Código arbitrario para la URL de exportaciones
    arbitrary_code_exportaciones = "1545f888-3490-466b-b815-60a0cd02ad23"
    
    # Lista para almacenar los enlaces formateados
    formatted_urls_exportaciones = []
    
    # Procesamos cada <a>
    for link in links:
        href = link.get('href')  # Extraemos el href
        title = link.get('title')  # Extraemos el title para filtrar exportaciones

        # Saltar los que no son "Exportaciones {mes} 2024"
        if "Exportaciones" not in title or "2024" not in title:
            continue

        # Extraemos el ID de la etiqueta 'a' desde el href
        resource_id = href.split('/resource/')[1].split('?')[0]

        # Extraemos la información del mes del title
        title_parts = title.split(' ')
        month = title_parts[1].lower()  # El mes (ejemplo: 'enero')

        # Formamos la nueva URL
        formatted_url = f"https://datos.gob.cl/dataset/{arbitrary_code_exportaciones}/resource/{resource_id}/download/exportaciones-{month}-2024.rar"

        # Añadimos la URL generada a la lista
        formatted_urls_exportaciones.append(formatted_url)

    return formatted_urls_exportaciones

download_dir = "downloads/"

# Asegurarse de que la carpeta exista
os.makedirs(download_dir, exist_ok=True)

# Función para descargar archivos en "chunks" con tqdm para visualizar progreso
def descargar_archivo(url, output_path, retries=3):
    if os.path.exists(output_path):
        print(f"{output_path} ya existe. Saltando descarga.")
        return

    for intento in range(retries):
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()  # Verifica si la solicitud fue exitosa
                total_size = int(r.headers.get('content-length', 0))  # Tamaño total del archivo
                block_size = 8192  # Tamaño de bloque para descargar

                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Descargando {os.path.basename(output_path)}") as t:
                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:  # Filtra chunks vacíos
                                f.write(chunk)
                                t.update(len(chunk))  # Actualiza la barra de progreso
            print(f"Descargado: {output_path}")
            return
        except RequestException as e:
            print(f"Error al descargar {output_path}: {e}. Reintentando... ({intento+1}/{retries})")
            sleep(5) 

    print(f"Error persistente. No se pudo descargar el archivo {output_path} después de {retries} intentos.")

def descargar_archivos_paralelo(urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(lambda url: descargar_archivo(url, os.path.join(download_dir, url.split("/")[-1])), urls)

def es_archivo_completo(filepath, min_size=1048576):
    if os.path.getsize(filepath) < min_size:
        print(f"El archivo {filepath} parece estar incompleto.")
        return False
    return True

def verificar_partes_multipart(archivo_base, max_parts=5):
    for i in range(1, max_parts + 1):
        parte = f"{archivo_base}.part{i:02d}.rar"
        if not os.path.exists(parte):
            print(f"Parte faltante: {parte}")
            return False
    return True

def descomprimir_y_convertir(download_dir):
    rar_files = sorted([f for f in os.listdir(download_dir) if f.endswith('.part01.rar') or f.endswith('.rar')])

    for rar_file in rar_files:
        rar_path = os.path.join(download_dir, rar_file)
        if not es_archivo_completo(rar_path):
            continue

        try:
            if ".part01.rar" in rar_file or ".rar" in rar_file:
                print(f"Descomprimiendo {rar_file} usando rarfile...")

                with rarfile.RarFile(rar_path) as rf:
                    file_list = rf.namelist()  # 
                    with tqdm(total=len(file_list), desc=f"Descomprimiendo {rar_file}") as t:
                        for file in file_list:
                            rf.extract(file, download_dir)
                            t.update(1)  

                print(f"Descomprimido: {rar_file}")

                extracted_files = [f for f in os.listdir(download_dir) if not f.endswith('.rar')]
                for file in extracted_files:
                    old_path = os.path.join(download_dir, file)
                    new_file_name = os.path.splitext(file)[0] + ".txt" 
                    new_path = os.path.join(download_dir, new_file_name)
                    os.rename(old_path, new_path) 
                    print(f"Renombrado: {new_file_name}")

        except Exception as e:
            e
def download():
    formatted_urls_importaciones = obtener_enlaces_importaciones()
    descargar_archivos_paralelo(formatted_urls_importaciones)

    formatted_urls_exportaciones = obtener_enlaces_exportaciones()
    descargar_archivos_paralelo(formatted_urls_exportaciones)
    
    descomprimir_y_convertir(download_dir)

if __name__ == "__main__":
    download()