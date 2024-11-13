from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from collection_manager.models import Sector, Puerto
from application.services import consultar_datos_manifiesto, generar_infraestructura, get_current_wave, get_current_weather, get_planificacion_san_antonio, obtener_datos_nave_por_nombre_o_imo, obtener_sismos_chile, obtener_ubicacion_barco, obtener_restricciones, obtener_nave, get_naves_recalando, scrape_nave_data
from application.services import obtener_restricciones
from application.serializers import GrafoInfraestructuraSerializer, SectorSerializer, SismoSerializer, WaveSerializer
from drf_spectacular.utils import extend_schema
import logging
from django.http import HttpResponse
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, OpenApiRequest
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse
import requests
from geopy.distance import geodesic

from django.http import JsonResponse
import os
import csv
import json

logger = logging.getLogger(__name__)

class GetGrafoInfraestructuraView(APIView):
    """
    Vista para obtener la información de los grafos de infraestructura.
    """
    @extend_schema(
        description="Obtiene la información de los grafos de infraestructura.",
        methods=['POST'],
        request=GrafoInfraestructuraSerializer, 
        responses={
            200: OpenApiResponse(description="Datos del grafo de infraestructura."),
            404: OpenApiResponse(description="No se encontró la información del grafo de infraestructura."),
            500: OpenApiResponse(description="Error al obtener la información del grafo de infraestructura.")
        }
    )
    def post(self, request, format=None):
        try:
            body = request.data
            body_puerto_origen = body.get('puerto_origen')
            body_puerto_destino = body.get('puerto_destino')
            if body_puerto_destino and body_puerto_origen:
                puerto_origen = Puerto.objects.get(codigo=body_puerto_origen)
                puerto_destino = Puerto.objects.get(codigo=body_puerto_destino)
                grafo_name = generar_infraestructura(puerto_origen, puerto_destino)
                # !Revisar 
                return Response({'grafo_url': grafo_name}, status=status.HTTP_200_OK)
            else:
                raise Exception("Faltan los parámetros 'puerto_origen' y 'puerto_destino' en el body")
        except Puerto.DoesNotExist:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información de los grafos de infraestructura: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información de los grafos de infraestructura.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información de los grafos de infraestructura: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información de los grafos de infraestructura.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class GetSismosChileView(APIView):
    """
    Vista para obtener la información de los sismos chilenos.
    """
    @extend_schema(
        description="Obtiene la información de los sismos chilenos.",
        responses={
            200: SismoSerializer(many=True),
        }
    )
    def get(self, request, format=None):
        try:
            # Obtener la nave de la bahía
            sismos = obtener_sismos_chile()
            
            # Retornar los datos de la nave en formato JSON
            return Response(sismos, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información de los sismos chilenos: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información de los sismos chilenos.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class GetDatosManifiesto(APIView):
    """
    Vista para obtener la información de un manifiesto de embarque.
    """
    @extend_schema(
        description="Obtiene la información de un manifiesto de embarque.",
        parameters=[
            OpenApiParameter(
                name='programacion',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description="Código del programación.",
                required=True
            )
        ],
        responses={
            200: OpenApiResponse(description="Datos del manifiesto de embarque."),
            404: OpenApiResponse(description="No se encontró la información del manifiesto de embarque."),
            500: OpenApiResponse(description="Error al obtener la información del manifiesto de embarque.")
        }
    )
    def get(self, request, programacion, format=None):
        try:            
            # Retornar los datos de la nave en formato JSON
            return Response(consultar_datos_manifiesto(programacion), status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información del manifiesto de embarque: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información del manifiesto de embarque.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class GetCurrentWaveView(APIView):
    """
    Vista para obtener la información actual de oleaje de un puerto.
    """
    @extend_schema(
        description="Obtiene la información actual de oleaje de un puerto.",
        parameters=[
            OpenApiParameter(
                name='codigo_puerto',
                type=OpenApiTypes.INT,
                location=OpenApiParameter.QUERY,
                description="Código del puerto.",
                required=True
            )
        ],
        responses={
            200: WaveSerializer(many=True),
            404: OpenApiResponse(description="No se encontró la información del puerto."),
            500: OpenApiResponse(description="Error al obtener la información del puerto.")
        }
    )
    def get(self, request, codigo_puerto, format=None):
        try:
            # Obtener la nave de la bahía
            puerto = Puerto.objects.get(codigo=codigo_puerto)
            if(not puerto.latitud or not puerto.longitud):
                raise Exception("No se encontró la latitud y la longitud del puerto")
            wave_data = get_current_wave(puerto.latitud, puerto.longitud)
            
            # Retornar los datos de la nave en formato JSON
            return Response(wave_data, status=status.HTTP_200_OK)
        except ValueError as e:
            # Registrar el error y retornar un mensaje genérico
            print(e)
            logger.error(f"Error al obtener la información actual de oleaje de un puerto: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información actual de oleaje de un puerto. No existen las coordenadas del puerto'},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Puerto.DoesNotExist:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información actual de oleaje de un puerto: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información actual de oleaje de un puerto.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información actual de oleaje de un puerto: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información actual de oleaje de un puerto.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class GetCurrentWeatherView(APIView):
    """
    Vista para obtener la información actual del clima de un puerto.
    """
    def get(self, request, codigo_puerto, format=None):
        try:
            # Obtener la nave de la bahía
            puerto = Puerto.objects.get(codigo=codigo_puerto)
            if(not puerto.latitud or not puerto.longitud):
                raise Exception("No se encontró la latitud y la longitud del puerto")
 
            weather = get_current_weather(puerto.latitud, puerto.longitud)
            
            # Retornar los datos de la nave en formato JSON
            return Response(weather, status=status.HTTP_200_OK)
        except Puerto.DoesNotExist:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información actual del clima de un puerto: ")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información actual del clima de un puerto.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información actual del clima de un puerto: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información actual del clima de un puerto.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PuertoSanAntonioView(APIView):
    """
    Vista para obtener la información de los planes de navegación del puerto de San Antonio.
    """
    def get(self, request, format=None):
        try:
            # Obtener la nave de la bahía
            naves = get_planificacion_san_antonio()
            
            # Retornar los datos de la nave en formato JSON
            return Response(naves, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la información de los planes de navegación del puerto de San Antonio: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la información de los planes de navegación del puerto de San Antonio.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PuertoNavesRecalandoView(APIView):
    """
    Vista para obtener las naves recalando 
    """
    def get(self, request, format=None):
        try:
            # Obtener las naves de la bahía
            naves = get_naves_recalando()
            
            # Retornar los datos de las naves en formato JSON
            return Response(naves, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener las naves recalando : {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener las naves recalando del puerto.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class UbicacionApiView(APIView):
    """
    Vista para obtener la ubicación de nave por su IMO.
    """
    def get(self, request, imo, format=None):
        try:
            # Obtener la nave de la bahía
            nave = obtener_ubicacion_barco(imo)
            
            # Retornar los datos de la nave en formato JSON
            return Response(nave, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la nave con IMO {imo}: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la nave.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class ObtenerNaveView(APIView):
    """
    Vista para obtener la información de una nave por su nombre o IMO.
    """
    def get(self, request, nombre_nave, format=None):
        try:
            # Obtener la url de la nave
            nave = obtener_datos_nave_por_nombre_o_imo(nombre_nave)
            print(nave)
            
            # Retornar los datos de la nave en formato JSON
            return Response(nave, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener la nave {nombre_nave}: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener la nave.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
class ObtenerNavesView(APIView):
    """
    Vista para obtener la nave por su ID.
    """
    def get(self, request, id_nave, format=None):
        try:
            # Obtener las naves de la bahía
            naves = obtener_nave(id_nave)
            
            # Retornar los datos de las naves en formato JSON
            return Response(naves, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener los datos de la nave {id_nave}: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener los datos de las naves.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    

class ObtenerRestriccionesView(APIView):
    """
    Vista para obtener las restricciones de una bahía por su ID.
    """
    def get(self, request, id_bahia=None, format=None):
        try:
            if not id_bahia:
                id_bahia = request.GET.get('id_bahia')
            # Intentar obtener la bahía
            bahia = Sector.objects.get(id=id_bahia)
            
            # Obtener las restricciones de la bahía
            restricciones = obtener_restricciones(bahia.id)
            
            # Serializar la información de la bahía
            bahia_serializer = SectorSerializer(bahia)
            
            return Response({
                'bahia': bahia_serializer.data,
                'restricciones': restricciones
            }, status=status.HTTP_200_OK)
        except Sector.DoesNotExist:
            logger.error(f"Error al obtener restricciones para la bahía {id_bahia}: No se encontró la bahía.")
            return Response(
                {'status': 'error', 'message': 'No se encontró la bahía'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        except Exception as e:
            logger.error(f"Error al obtener restricciones para la bahía {id_bahia}: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener los datos de la restricción.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GuardarView(APIView) :

    def get(self, request) :
        # Filtrar los puertos de Chile que son puertos marítimos
        puertos_chile = Puerto.objects.filter(pais__codigo='997', tipo='Puerto marítimo')
        
        # Obtener las restricciones de la bahía
        #restricciones = obtener_restricciones(bahia.id)

        # Lista para almacenar las líneas del archivo de texto
        contenido = []
        

        for puerto in puertos_chile :
            if puerto.latitud and puerto.longitud :
                try :
                    #bahias = puerto.sector
                    #print(f"ID BAHIAS: {bahias}")
                    
                    weather = get_current_weather(puerto.latitud, puerto.longitud)
                    wave_data = get_current_wave(puerto.latitud, puerto.longitud)
                    sismos = obtener_sismos_chile()
                    
                    climaHoy = weather.get('current', {})
                    hourly_data = weather.get('hourly', {})
                    maxWaveHeight = max(oleaje.get('wave_height', 0) for oleaje in wave_data)
                    
                    
                    #Distancia maxima para sismos
                    distancia_maxima_km = 500
                    ubicacion_puerto = (puerto.latitud, puerto.longitud)
                    probabilidadFallaSismo_inicial = 0
                    probabilidadFallaSismo_final = 0
                    
                    #Calculo de la probabilidad para Sismos
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
                                probabilidadFallaSismo = 0
                            elif magnitud <= 5:
                                probabilidadFallaSismo = 0
                            elif magnitud >= 7:
                                probabilidadFallaSismo = 100
                            else:
                                probabilidadFallaSismo = ((magnitud - 5)/(7 - 5))*100
                                
                            probabilidadFallaSismo_final = max(probabilidadFallaSismo_inicial, probabilidadFallaSismo) 
                                
                    
                    #Calculo de la probabilidad para LLuvia
                    precipTotal = sum(hora.get('rain', {}).get('1h', 0) for hora in hourly_data)
                    precipMax = min(precipTotal, 150)
                    probabilidadFalla = (precipMax/150)*100
                    
                    #Calculo de la probabilidad para el Oleaje
                    if maxWaveHeight >= 1.8:
                        probabilidadFallaOleaje = 100
                    elif maxWaveHeight >= 1.5:
                        probabilidadFallaOleaje = ((maxWaveHeight - 1.5)/0.3)*100
                    else:
                        probabilidadFallaOleaje = 0
                    

                    # Añadir información al contenido
                    contenido.append(f"Puerto: {puerto.nombre}\n")
                    contenido.append(f"País: {puerto.pais.nombre}\n")
                    contenido.append(f"Clima: {climaHoy}\n")
                    contenido.append(f"Probabilidad de Falla para LLuvia: {probabilidadFalla:.2f}%\n")
                    contenido.append(f"Probabilidad de Falla para Oleaje: {probabilidadFallaOleaje:.2f}%\n")
                    contenido.append(f"Probabilidad de Falla para Sismos: {probabilidadFallaSismo_final:.2f}%\n")
                    contenido.append("-" * 40 + "\n")  # Separador entre puertos
                except Exception as e :
                    # Manejar excepciones y registrar el error en el contenido
                    contenido.append(f"Puerto: {puerto.nombre}\n")
                    contenido.append(f"Error al obtener datos: {str(e)}\n")
                    contenido.append("-" * 40 + "\n")
            else :
                # Manejar casos donde faltan latitud o longitud
                contenido.append(f"Puerto: {puerto.nombre}\n")
                contenido.append("Error: Latitud o longitud faltante.\n")
                contenido.append("-" * 40 + "\n")

        # Unir todas las líneas en una sola cadena de texto
        texto_final = "\n".join(contenido)

        # Crear la respuesta HTTP con el contenido de texto
        respuesta = HttpResponse(texto_final, content_type='text/plain')
        respuesta['Content-Disposition'] = 'attachment; filename="datos_puertos.txt"'

        return respuesta




    # def guardar_probabilidades(request):
    #
    #
    #     if request.method == 'POST':
    #
    #         try:
    #             data = json.loads(request.body)
    #             sismos = data.get('sismos', [])
    #
    #             if not sismos or not isinstance(sismos, list):
    #                 return JsonResponse({'message': 'Datos inválidos'}, status=400)
    #
    #             # Definir la ruta donde se guardará el archivo
    #             ruta_carpeta = os.path.join(os.path.dirname(__file__), 'archivos')
    #             if not os.path.exists(ruta_carpeta):
    #                 os.makedirs(ruta_carpeta)
    #
    #             ruta_archivo = os.path.join(ruta_carpeta, 'sismologia_probabilidades.csv')
    #
    #             # Escribir el archivo CSV
    #             with open(ruta_archivo, mode='w', newline='', encoding='utf-8') as archivo_csv:
    #                 escritor_csv = csv.writer(archivo_csv)
    #                 # Escribir encabezados
    #                 escritor_csv.writerow(['Fecha y Ubicación', 'Profundidad (km)', 'Magnitud', 'Probabilidad de Falla'])
    #                 # Escribir datos de sismos
    #                 for sismo in sismos:
    #                     escritor_csv.writerow([
    #                         sismo.get('fecha_ubicacion', 'N/A'),
    #                         sismo.get('profundidad', 'N/A'),
    #                         sismo.get('magnitud', 'N/A'),
    #                         sismo.get('probabilidadFalla', 'N/A')
    #                     ])
    #
    #             print('Archivo guardado en:', ruta_archivo)
    #             return JsonResponse({'message': 'Archivo generado exitosamente'})
    #
    #         except Exception as e:
    #             print('Error al procesar la solicitud:', e)
    #             return JsonResponse({'message': 'Error al procesar la solicitud'}, status=500)
    #     else:
    #         return JsonResponse({'message': 'Método no permitido'}, status=405)
