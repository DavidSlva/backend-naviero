from venv import logger

from rest_framework.views import APIView

from rest_framework.viewsets import ModelViewSet


from interpreter.models import Registro, VolumenPredicho
from interpreter.serializer import RegistroSerializer, VolumenTotalSerializer, VolumenPorPuertoSerializer, \
    VolumenTotalAnualSerializer, VolumenPorPuertoAnualSerializer, VolumenPredichoSerializer
from interpreter.services import cargar_registros_importacion
from rest_framework.pagination import PageNumberPagination

from interpreter.utils import procesar_y_cargar_datos

from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action
from django.db.models import Sum, F, Value
from django.db.models.functions import ExtractYear
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters

from .filters import VolumenPorPuertoFilter, VolumenPredichoFilter
from .models import VolumenTotal, VolumenPorPuerto

class PrediccionesViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las predicciones.
    """
    queryset = VolumenPredicho.objects.all()  # El modelo VolumenPredicho representa las predicciones
    serializer_class = VolumenPredichoSerializer
    filter_backends = [DjangoFilterBackend]  # Habilita los filtros
    filterset_class = VolumenPredichoFilter

class VolumenPorPuertoViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de los volumenes por puerto.
    """
    queryset = VolumenPorPuerto.objects.all()  # El modelo VolumenPorPuerto representa los volumenes por puerto
    serializer_class = VolumenPorPuertoSerializer
    filter_backends = [DjangoFilterBackend]  # Habilita los filtros
    filterset_class = VolumenPorPuertoFilter

class VolumenAnualViewSet(viewsets.ViewSet):
    """
    ViewSet para obtener el volumen total por año y el volumen por puerto en un año específico.
    """
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['anio', 'puerto__codigo']
    ordering_fields = ['anio', 'volumen_total']

    @action(detail=False, methods=['get'], url_path='total-por-anio')
    def total_por_anio(self, request):
        """
        Endpoint para obtener el volumen total por año.
        Se puede filtrar por año utilizando el parámetro 'anio'.
        """
        anio = request.query_params.get('anio', None)

        queryset = VolumenTotal.objects.annotate(
            anio=ExtractYear('semana')
        )

        if anio:
            queryset = queryset.filter(anio=anio)

        queryset = queryset.values('anio').annotate(
            volumen_total=Sum('volumen_total')
        ).order_by('-anio')

        serializer = VolumenTotalAnualSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], url_path='por-puerto-anio')
    def por_puerto_anio(self, request):
        """
        Endpoint para obtener el volumen por puerto en un año específico.
        Se deben proporcionar los parámetros 'codigo_puerto' y 'anio'.
        """
        codigo_puerto = request.query_params.get('codigo_puerto', None)
        anio = request.query_params.get('anio', None)
        print(codigo_puerto, anio)

        if not codigo_puerto or not anio:
            return Response(
                {"error": "Por favor, proporciona los parámetros 'codigo_puerto' y 'anio'."},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            codigo_puerto = int(codigo_puerto)
            anio = int(anio)
        except ValueError:
            return Response(
                {"error": "Los parámetros 'codigo_puerto' y 'anio' deben ser enteros."},
                status=status.HTTP_400_BAD_REQUEST
            )

        queryset = VolumenPorPuerto.objects.filter(
            puerto__codigo=codigo_puerto
        ).annotate(
            anio=ExtractYear('semana')
        ).filter(
            anio=anio
        ).values(
            puerto_codigo=F('puerto__codigo'),
            puerto_nombre=F('puerto__nombre'),
            anio=F('anio')  # Aliasando 'anio' correctamente
        ).annotate(
            volumen_total=Sum('volumen')
        ).order_by('-volumen_total')

        serializer = VolumenPorPuertoAnualSerializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

class CargarRegistrosView(APIView) :
    """
    View para cargar los registros en general a la base de datos.
    """

    def get(self, request, format=None) :
        try :
            # Cargar los registros en la base de datos
            cargar_registros_importacion()

            # Retornar un mensaje de éxito
            return Response(
                {'status' : 'success', 'message' : 'Registros cargados correctamente.'},
                status=status.HTTP_200_OK
            )

        except Exception as e :
            # Registrar el error y retornar un mensaje genérico
            logger.error(f"Error al cargar los registros: {e}")
            return Response(
                {'status' : 'error', 'message' : 'Error al cargar los registros.'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CustomPagination(PageNumberPagination) :
    """
    Paginador personalizado para definir un límite de registros por solicitud.
    """
    page_size = 100  # Tamaño predeterminado (100 registros)
    page_size_query_param = 'page_size'  # Permite al cliente definir cuántos registros quiere recibir
    max_page_size = 1000  # Limita el máximo número de registros por solicitud


class RegistrosViewSet(ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD de los registros con soporte de filtros y paginación.
    """
    queryset = Registro.objects.all()  # Consulta principal de los registros
    serializer_class = RegistroSerializer
    filter_backends = [DjangoFilterBackend]  # Habilita los filtros
    pagination_class = CustomPagination  # Usa el paginador personalizado

class VolumenTotalViewSet(ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD de los datos de volumen total con soporte de filtros y paginación.
    """
    queryset = VolumenTotal.objects.all()  # Consulta principal de los registros
    serializer_class = VolumenTotalSerializer
    filter_backends = [DjangoFilterBackend]  # Habilita los filtros
    pagination_class = CustomPagination  # Usa el paginador personalizado

class VolumenPorPuertoViewSet(viewsets.ModelViewSet):
    """
    ViewSet para manejar las operaciones CRUD de los datos de volumen por puerto con soporte de filtros y paginación.
    """
    queryset = VolumenPorPuerto.objects.all()
    serializer_class = VolumenPorPuertoSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = VolumenPorPuertoFilter
    pagination_class = CustomPagination

class DatosGeneralesViewSet(ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD de los datos generales con soporte de filtros y paginación.
    """

    # queryset = Registro.objects.all()  # Consulta principal de los registros
    # serializer_class = RegistroSerializer
    # filter_backends = [DjangoFilterBackend]  # Habilita los filtros
    # pagination_class = CustomPagination  # Usa el paginador personalizado

    @action(detail=False, methods=['get'], url_path='general')
    def general(self, request) :
        # Cargar parquet
        # df = get_dataframe()
        path = r"C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet"
        procesar_y_cargar_datos(path)
        return Response({'status': 'success', 'message': 'Datos cargados correctamente.'})
