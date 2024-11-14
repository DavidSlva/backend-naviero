import subprocess
import sys
import os
import json
from pathlib import Path
import threading
import webbrowser

def run_command(command, show_output=False):
    try:
        if show_output:
            result = subprocess.run(command, shell=True, check=True)
        else:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        sys.exit(1)

# Leer el Pipfile
pipfile_path = Path("Pipfile")
if not pipfile_path.exists():
    print("Error: Pipfile no encontrado en el directorio actual.")
    sys.exit(1)

# Extraer la versión de Python necesaria del Pipfile
with open(pipfile_path, "r") as pipfile:
    pipfile_content = pipfile.read()
    if "python_version" not in pipfile_content:
        print("Error: Pipfile no tiene la especificación de 'python_version'.")
        sys.exit(1)
    
    # Buscar la versión de Python en el Pipfile
    for line in pipfile_content.splitlines():
        if "python_version" in line:
            version_line = line.strip().split("=")
            if len(version_line) > 1:
                python_version_required = version_line[1].strip().strip("\"")
                break

# Verificar la versión de Python actual
current_python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
if current_python_version != python_version_required:
    print(f"Error: Se necesita Python {python_version_required} pero se tiene Python {current_python_version}.")
    sys.exit(1)

# Verificar si pipenv está instalado
try:
    subprocess.run(["pipenv", "--version"], check=True, capture_output=True, text=True)
except subprocess.CalledProcessError:
    # Si no está instalado, solicitar confirmación para instalar pipenv
    install_pipenv = input("Pipenv no está instalado. ¿Desea instalarlo ahora? [s/N]: ").lower()
    if install_pipenv == 's':
        print("Instalando pipenv...")
        run_command("pip install pipenv", show_output=True)
    else:
        print("Pipenv es necesario para continuar. Abortando.")
        sys.exit(1)

# Crear el entorno virtual con pipenv e instalar dependencias
print("Creando el entorno virtual e instalando dependencias del Pipfile...")
run_command("pipenv install", show_output=True)

# Ejecutar script en pagina/main.py con dos parámetros
print("Ejecutando pagina/main.py con los parámetros 644 y 906...")
run_command("pipenv run python pagina/main.py 644 906", show_output=True)

# Iniciar servidor Django en el puerto 8080 en un hilo separado
print("Iniciando servidor Django en el puerto 8080...")
django_thread = threading.Thread(target=lambda: run_command("pipenv run python manage.py runserver 8080", show_output=True))
django_thread.daemon = True
django_thread.start()

# Esperar unos segundos para asegurarse de que el servidor esté en funcionamiento
import time
time.sleep(5)

# Abrir la página admin en el navegador
admin_url = "http://127.0.0.1:8080/admin"
print(f"Abriendo la página admin en el navegador: {admin_url}")
webbrowser.open(admin_url)
print("Credenciales de acceso:")
print("Usuario: diegoportales")
print("Contraseña: c!8fr7^5XQg#8@&J")

# Ejecutar servidor HTTP para mostrar el archivo index.html en el puerto 8000 en un hilo separado
index_html_path = Path("pagina/index.html")
if not index_html_path.exists():
    print("Error: index.html no encontrado en la ruta especificada.")
    sys.exit(1)

print("Iniciando servidor HTTP en el puerto 8000 para mostrar index.html...")
os.chdir("pagina")
http_thread = threading.Thread(target=lambda: run_command("python -m http.server 8000", show_output=True))
http_thread.daemon = True
http_thread.start()

# Esperar unos segundos para asegurarse de que el servidor esté en funcionamiento
time.sleep(5)

# Abrir la página localhost:8000 en el navegador
localhost_url = "http://127.0.0.1:8000"
print(f"Abriendo la página en el navegador: {localhost_url}")
webbrowser.open(localhost_url)

# Mantener el script en ejecución para permitir que los hilos trabajen
django_thread.join()
http_thread.join()
