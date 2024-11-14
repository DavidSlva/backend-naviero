from venv import logger
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from interpreter.services import cargar_registros_importacion


# Create your views here.
class CargarRegistrosView(APIView):
    """
    View para cargar los registros en general a la base de datos.
    """
    def get(self, request, format=None):
        try:
            # Cargar los registros en la base de datos
            cargar_registros_importacion()
            
            # Retornar un mensaje de éxito
            return Response(
                {'status': 'success', 'message': 'Registros cargados correctamente.'},
                status=status.HTTP_200_OK
            )
        
        except Exception as e:
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al cargar los registros: {e}")
            return Response(
                {'status': 'error', 'message': 'Error al cargar los registros.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )