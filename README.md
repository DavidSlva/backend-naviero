# README - Backend Naviero Setup

## Clonar el Repositorio

Para comenzar, debes clonar este repositorio en tu máquina local. Ejecuta el siguiente comando en tu terminal:

```sh
 git clone https://github.com/DavidSlva/backend-naviero.git
```

Luego, navega dentro del directorio clonado:

```sh
cd backend-naviero
```

## Requisitos del Sistema

Este proyecto requiere **Python 3.12**. Es importante que tengas esta versión específica de Python para garantizar que todo funcione correctamente. Si no tienes Python 3.12, debes instalarlo antes de continuar. Puedes verificar tu versión actual de Python con el siguiente comando:

```sh
python --version
```

Si tu versión de Python no coincide con la requerida, por favor instálala desde [python.org](https://www.python.org/downloads/).

## Configuración del Entorno

### Agregar el Archivo `.env`

En la carpeta `backend`, debes agregar un archivo `.env` que contiene configuraciones específicas para el proyecto. Si eres un usuario de la universidad, el archivo `.env` debería venir incluido como comentario en la entrega del proyecto.

Asegúrate de que el archivo `.env` esté en la carpeta `backend` antes de continuar, ya que este archivo contiene configuraciones críticas para la aplicación.

## Ejecución del Script Principal (`main.py`)

El archivo `main.py` automatiza todos los pasos necesarios para configurar y ejecutar el proyecto. A continuación se detalla lo que sucede al ejecutar este script:

1. **Verificación de la Versión de Python**: El script revisa el `Pipfile` y verifica si la versión de Python instalada es la correcta. Si la versión no coincide, el script detiene su ejecución y solicita que se instale la versión adecuada.

2. **Instalación de Pipenv**: El script verifica si `pipenv` está instalado. Si no está presente, ofrece instalarlo automáticamente. `Pipenv` es necesario para gestionar el entorno virtual y las dependencias del proyecto.

3. **Creación del Entorno Virtual e Instalación de Dependencias**: Con `pipenv`, el script crea un entorno virtual y descarga todas las dependencias necesarias que se especifican en el `Pipfile`.

4. **Ejecución de `pagina/main.py` con Parámetros**: El script ejecuta `pagina/main.py` con dos parámetros específicos (`644` y `906`). Esto es parte de la lógica del proyecto y se realiza de manera automática, mostrando los logs para que puedas ver el progreso de la ejecución.

5. **Inicio del Servidor Django**: El script inicia el servidor de Django en el puerto `8080`. Esto permite acceder a la interfaz administrativa de Django, donde podrás gestionar los datos del proyecto. Puedes acceder a la página de administración usando la siguiente URL:

   [http://127.0.0.1:8080/admin](http://127.0.0.1:8080/admin)

   Las credenciales de acceso para el superusuario son:

   - **Usuario**: `diegoportales`
   - **Contraseña**: `c!8fr7^5XQg#8@&J`

6. **Servidor HTTP para el Archivo `index.html`**: Además del servidor Django, el script también inicia un servidor HTTP simple para servir el archivo `index.html` ubicado en la carpeta `pagina`. Este servidor se ejecuta en el puerto `8000` y la página se abrirá automáticamente en tu navegador en la siguiente URL:

   [http://127.0.0.1:8000](http://127.0.0.1:8000)

Este script automatiza por completo el proceso de instalación, configuración y ejecución del proyecto, lo que hace que sea muy sencillo para ti levantar todo el entorno necesario con un solo comando.

## Documentación de la API

La documentación de la API está disponible en la ruta local `/docs/swagger`. Puedes acceder a ella después de iniciar el servidor Django en la siguiente URL:

[http://127.0.0.1:8080/docs/swagger](http://127.0.0.1:8080/docs/swagger)

Aquí encontrarás información sobre cómo usar la API, incluyendo detalles sobre la **metadata** y **amenazas** que se manejan en el proyecto.

## Reporte de Errores

Si encuentras algún error o tienes alguna sugerencia, por favor repórtalo en la página del repositorio de GitHub:

[https://github.com/DavidSlva/backend-naviero.git](https://github.com/DavidSlva/backend-naviero.git)

## Notas Importantes

- Este script ha sido probado en **Windows**. Si estás en otro sistema operativo, algunos comandos pueden necesitar adaptaciones.
- Asegúrate de tener una conexión a Internet estable, ya que la descarga de dependencias y la ejecución del script requieren acceso a recursos en línea.
- Algunas partes del proceso (como la instalación de dependencias o la ejecución del servidor) pueden tardar un poco. Se recomienda tener paciencia durante estas etapas.

¡Gracias por usar el backend naviero! Siéntete libre de contribuir o de comunicar cualquier problema que encuentres.
