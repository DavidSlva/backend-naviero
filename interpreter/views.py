from venv import logger

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.viewsets import ModelViewSet

from interpreter.models import Registro
from interpreter.serializer import RegistroSerializer
from interpreter.services import cargar_registros_importacion
from rest_framework.pagination import PageNumberPagination


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
class CustomPagination(PageNumberPagination):
    """
    Paginador personalizado para definir un límite de registros por solicitud.
    """
    page_size = 100  # Tamaño predeterminado (100 registros)
    page_size_query_param = 'page_size'  # Permite al cliente definir cuántos registros quiere recibir
    max_page_size = 1000  # Limita el máximo número de registros por solicitud


class RegistrosViewSet(ModelViewSet):
    """
    ViewSet para manejar las operaciones CRUD de los registros con soporte de filtros y paginación.
    """
    queryset = Registro.objects.all()  # Consulta principal de los registros
    serializer_class = RegistroSerializer
    filter_backends = [DjangoFilterBackend]  # Habilita los filtros
    pagination_class = CustomPagination  # Usa el paginador personalizado