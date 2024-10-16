from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from collection_manager.models import Sector, Puerto
from application.services import get_current_wave, get_current_weather, get_planificacion_san_antonio, obtener_sismos_chile, obtener_ubicacion_barco, obtener_restricciones, obtener_nave, get_naves_recalando
from application.services import obtener_restricciones
from application.serializers import SectorSerializer, SismoSerializer, WaveSerializer
from drf_spectacular.utils import extend_schema
import logging
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse

logger = logging.getLogger(__name__)

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
            if(puerto.latitud or not puerto.longitud or float(puerto.latitud) == 0 or float(puerto.longitud) == 0):
                raise Exception("No se encontró la latitud y la longitud del puerto")
            wave_data = get_current_wave(puerto.latitud, puerto.longitud)
            
            # Retornar los datos de la nave en formato JSON
            return Response(wave_data, status=status.HTTP_200_OK)
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
            logger.error(f"Error al obtener la información actual del clima de un puerto: {e}")
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
    def get(self, request, id_bahia, format=None):
        try:
            # Intentar obtener la bahía
            bahia = get_object_or_404(Sector, id=id_bahia)
            
            # Obtener las restricciones de la bahía
            restricciones = obtener_restricciones(bahia.id)
            
            # Serializar la información de la bahía
            bahia_serializer = SectorSerializer(bahia)
            
            # Retornar los datos de la bahía y las restricciones en formato JSON
            return Response({
                'bahia': bahia_serializer.data,
                'restricciones': restricciones
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al obtener restricciones para la bahía {id_bahia}: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al obtener los datos de la restricción.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
