import requests
from lxml import html

# URL de la página que contiene los enlaces
url = "https://datos.gob.cl/dataset/registro-de-importacion-2024/resource/c4941a06-34b9-47d0-a477-ff2aa021e442?inner_span=True"

# Realizamos la solicitud HTTP para obtener el HTML
response = requests.get(url)
page_content = response.content

# Parseamos el contenido HTML
tree = html.fromstring(page_content)

# XPath que selecciona los <a> dentro de <ul> como especificaste
xpath_query = '//*[@id="content"]/div[3]/aside/section/ul/li/a'

# Extraemos todos los elementos <a> y sus atributos href y title
links = tree.xpath(xpath_query)

# Código arbitrario para reemplazar el año en la URL
arbitrary_code = "096c3946-657e-420f-ae74-2337c00b5ba2"

# Lista para almacenar los enlaces formateados
formatted_urls = []

# Procesamos cada <a>
for link in links:
    href = link.get('href')  # Extraemos el href
    title = link.get('title')  # Extraemos el title para obtener el mes y la parte

    # Extraemos el ID de la etiqueta 'a' desde el href
    # El ID es la parte del href después de "/resource/"
    resource_id = href.split('/resource/')[1].split('?')[0]
    # Extraemos la información del mes y la parte del title
    # Por ejemplo: "Importaciones - enero 2024 1/5"
    # Separamos para obtener el mes y la parte
    if('Metadata' in title):
        continue
    title_parts = title.split(' ')
    month = title_parts[2].lower()  # El mes (ejemplo: 'enero')
    part_info = title_parts[-1]  # La parte (ejemplo: '1/5')
    part_number = part_info.split('/')[0]  # Número de parte (ejemplo: '1')

    # Formamos la nueva URL
    formatted_url = f"https://datos.gob.cl/dataset/{arbitrary_code}/resource/{resource_id}/download/importaciones-{month}-2024.part0{part_number}.rar"

    # Añadimos la URL generada a la lista
    formatted_urls.append(formatted_url)


