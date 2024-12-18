import numpy as np
import pandas as pd
from django.http import JsonResponse
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.decorators import api_view, action

from application.services import create_rutas
from collection_manager.filters import SectorFilter, PuertoFilter
#
from collection_manager.serializer import PaisSerializer, PuertoSerializer, TipoOperacionSerializer, AduanaSerializer, \
    TipoCargaSerializer, ViaTransporteSerializer, RegimenImportacionSerializer, ModalidadVentaSerializer, \
    RegionSerializer, UnidadMedidaSerializer, TipoMonedaSerializer, ClausulaSerializer, SectorSerializer, \
    MuelleSerializer
from collection_manager.models import (Pais, Puerto, TipoOperacion, Aduana, TipoCarga, ViaTransporte,
                                       RegimenImportacion,
                                       ModalidadVenta, Region, UnidadMedida, TipoMoneda, Clausula, Sector, Muelle)
from collection_manager.services import obtener_bahias
from rest_framework import viewsets
from rest_framework import status
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiExample


@api_view(['GET'])
def cargar_rutas_importantes(request) :
    puertos_chile = Puerto.objects.filter(pais__codigo='997', tipo='Puerto marítimo')
    puertos_importantes = Puerto.objects.filter(important=True)
    for importante in puertos_importantes :
        if importante.latitud and importante.longitud :
            rutas, distancias_totales = create_rutas(importante, puertos_chile)
    return Response({"message" : "Rutas actualizadas."}, status=status.HTTP_200_OK)


@api_view(['POST'])
def cargar_aduanas(request) :
    file_path = 'collection_manager/data/tablas_de_codigos.xlsx'
    data = pd.ExcelFile(file_path)
    # 4. Cargar Aduanas
    print("Cargando datos de Aduanas...")
    aduanas_df = pd.read_excel(data, 'Aduanas', skiprows=3)
    for _, row in aduanas_df.iterrows() :
        if pd.notna(row['COD_ADUANA_TRAMITACION']) :
            try :
                aduana, created = Aduana.objects.update_or_create(
                    codigo=row['COD_ADUANA_TRAMITACION'],
                    defaults={
                        'nombre' : row['NOMBRE_ADUANA'],
                        'latitud' : row['LATITUD'] if 'LATITUD' in row else None,
                        'longitud' : row['LONGITUD'] if 'LONGITUD' in row else None
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Aduana {aduana.nombre} ({aduana.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar aduana {row['NOMBRE_ADUANA']}: {e}")
    return JsonResponse({'status' : 'success', 'message' : 'Datos cargados correctamente desde el archivo Excel.'})


@api_view(['POST'])
def cargar_codigos(request) :
    file_path = 'collection_manager/data/tablas_de_codigos.xlsx'
    data = pd.ExcelFile(file_path)
    print(f"Se encontraron las siguientes hojas: {data.sheet_names}")

    # 1. Cargar Países
    print("Cargando datos de Países...")
    paises_df = pd.read_excel(data, 'Países', skiprows=4)
    for _, row in paises_df.iterrows() :
        if pd.notna(row['COD_PAIS']) :
            try :
                pais, created = Pais.objects.update_or_create(
                    codigo=row['COD_PAIS'],
                    defaults={
                        'nombre' : row['NOMBRE_PAIS'],
                        'continente' : row['NOMBRE_CONTINENTE']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"País {pais.nombre} ({pais.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar país {row['NOMBRE_PAIS']}: {e}")

    # 2. Cargar Puertos

    puertos_df = pd.read_excel(data, 'Puertos', skiprows=4)
    print(puertos_df)
    for _, row in puertos_df.iterrows() :
        if pd.notna(row['COD_PAIS']) and pd.notna(row['COD_PUERTO']) :
            try :
                pais_obj = Pais.objects.get(codigo=int(row['COD_PAIS']))
                puerto, created = Puerto.objects.update_or_create(
                    codigo=int(row['COD_PUERTO']),
                    defaults={
                        'nombre' : row['NOMBRE_PUERTO'],
                        'tipo' : row['TIPO_PUERTO'],
                        'pais' : pais_obj,
                        'latitud' : row['LATITUD'] if 'LATITUD' in row else None,
                        'longitud' : row['LONGITUD'] if 'LONGITUD' in row else None,
                        # 'longitud': row['LONGITUD'],
                        # 'zona_geografica': row['ZONA_GEOGRAFICA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Puerto {puerto.nombre} ({puerto.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar puerto {row['NOMBRE_PUERTO']}: {e}")

    # 3. Cargar Tipos de Operación
    print("Cargando datos de Tipos de Operación...")
    tipos_operacion_df = pd.read_excel(data, 'Tipos de Operación', skiprows=3)
    for _, row in tipos_operacion_df.iterrows() :
        if pd.notna(row['COD_TIPO_OPERACION']) :
            try :
                tipo_operacion, created = TipoOperacion.objects.update_or_create(
                    codigo=row['COD_TIPO_OPERACION'],
                    defaults={
                        'nombre' : row['NOMBRE_TIPO_OPERACION'],
                        'nomber_a_consignar' : row['NOMBRE_A_CONSIGNAR'],
                        'ind_ingreso' : row['INGRESO/SALIDA'] == 'Ingreso',
                        'ind_salida' : row['INGRESO/SALIDA'] == 'Salida',
                        'operacion' : row['OPERACION']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Tipo de Operación {tipo_operacion.nombre} ({tipo_operacion.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar tipo de operación {row['NOMBRE_TIPO_OPERACION']}: {e}")

    # 4. Cargar Aduanas
    print("Cargando datos de Aduanas...")
    aduanas_df = pd.read_excel(data, 'Aduanas', skiprows=3)
    for _, row in aduanas_df.iterrows() :
        if pd.notna(row['COD_ADUANA_TRAMITACION']) :
            try :
                aduana, created = Aduana.objects.update_or_create(
                    codigo=row['COD_ADUANA_TRAMITACION'],
                    defaults={
                        'nombre' : row['NOMBRE_ADUANA'],
                        'latitud' : row['LATITUD'],
                        'longitud' : row['LONGITUD']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Aduana {aduana.nombre} ({aduana.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar aduana {row['NOMBRE_ADUANA']}: {e}")

    # 5. Cargar Tipos de Carga
    print("Cargando datos de Tipos de Carga...")
    tipos_carga_df = pd.read_excel(data, 'Tipos de Carga', skiprows=5)
    print(tipos_carga_df)
    for _, row in tipos_carga_df.iterrows() :
        if pd.notna(row['COD_TIPO_CARGA']) :
            try :
                tipo_carga, created = TipoCarga.objects.update_or_create(
                    codigo=row['COD_TIPO_CARGA'],
                    defaults={
                        'nombre' : row['NOMBRE_TIPO_CARGA'],
                        'descripcion' : row['DESCRIPCION_TIPO_CARGA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Tipo de Carga {tipo_carga.nombre} ({tipo_carga.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar tipo de carga {row['NOMBRE_TIPO_CARGA']}: {e}")

    # 6. Cargar Vías de Transporte
    print("Cargando datos de Vías de Transporte...")
    vias_transporte_df = pd.read_excel(data, 'Vías de Transporte', skiprows=3)
    for _, row in vias_transporte_df.iterrows() :
        if pd.notna(row['COD_VIA_TRANSPORTE']) and row['COD_VIA_TRANSPORTE'] != np.nan :
            try :
                via_transporte, created = ViaTransporte.objects.update_or_create(
                    codigo=row['COD_VIA_TRANSPORTE'],
                    defaults={'nombre' : row['NOMBRE_VIA_TRANSPORTE']}
                )
                action = "creado" if created else "actualizado"
                print(f"Vía de Transporte {via_transporte.nombre} ({via_transporte.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar vía de transporte {row['NOMBRE_VIA_TRANSPORTE']}: {e}")

    # 7. Cargar Régimen de Importación
    print("Cargando datos de Régimen de Importación...")
    regimen_importacion_df = pd.read_excel(data, 'Régimen de Importación', skiprows=3)
    print(regimen_importacion_df)
    for _, row in regimen_importacion_df.iterrows() :
        if pd.notna(row['COD_RÉGIMEN_IMPORTACION']) :
            try :
                regimen, created = RegimenImportacion.objects.update_or_create(
                    codigo=row['COD_RÉGIMEN_IMPORTACION'],
                    defaults={
                        'nombre' : row['NOMBRE_RÉGIMEN_IMPORTACION'],
                        'sigla' : row['SIGLA_RÉGIMEN_IMPORTACION'],
                        'active' : True
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Régimen de Importación {regimen.nombre} ({regimen.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar régimen de importación {row['NOMBRE_RÉGIMEN_IMPORTACION']}: {e}")

    # 8. Cargar Modalidades de Venta
    print("Cargando datos de Modalidades de Venta...")
    modalidades_venta_df = pd.read_excel(data, 'Modalidades de Venta', skiprows=2)
    print(modalidades_venta_df)
    for _, row in modalidades_venta_df.iterrows() :
        if pd.notna(row['COD_MODALIDAD_VENTA']) :
            try :
                modalidad, created = ModalidadVenta.objects.update_or_create(
                    codigo=row['COD_MODALIDAD_VENTA'],
                    defaults={
                        'nombre' : row['NOMBRE_MODALIDAD_VENTA'],
                        'descripcion' : row['DESCRIPCION_MODALIDAD_VENTA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Modalidad de Venta {modalidad.nombre} ({modalidad.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar modalidad de venta {row['NOMBRE_MODALIDAD_VENTA']}: {e}")

    # 9. Cargar Regiones
    print("Cargando datos de Regiones...")
    regiones_df = pd.read_excel(data, 'Regiones', skiprows=3)
    print(regiones_df)
    for _, row in regiones_df.iterrows() :
        if pd.notna(row['COD_REGION_ORIGEN']) :
            try :
                region, created = Region.objects.update_or_create(
                    codigo=row['COD_REGION_ORIGEN'],
                    defaults={'nombre' : row['NOMBRE_REGION']}
                )
                action = "creado" if created else "actualizado"
                print(f"Región {region.nombre} ({region.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar región {row['NOMBRE_REGION']}: {e}")

    # 10. Cargar Unidades de Medida
    print("Cargando datos de Unidades de Medida...")
    unidades_medida_df = pd.read_excel(data, 'Unidades de Medida', skiprows=3)
    for _, row in unidades_medida_df.iterrows() :
        if pd.notna(row['COD_UNIDAD_MEDIDA']) :
            try :
                unidad_medida, created = UnidadMedida.objects.update_or_create(
                    codigo=row['COD_UNIDAD_MEDIDA'],
                    defaults={
                        'nombre' : row['NOMBRE_UNIDAD_MEDIDA'],
                        'unidad' : row['UNIDAD_MEDIDA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Unidad de Medida {unidad_medida.nombre} ({unidad_medida.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar unidad de medida {row['NOMBRE_UNIDAD_MEDIDA']}: {e}")

    # 11. Cargar Moneda
    print("Cargando datos de Moneda...")
    moneda_df = pd.read_excel(data, 'Moneda', skiprows=4)
    print(moneda_df)
    for _, row in moneda_df.iterrows() :
        if pd.notna(row['MONEDA']) :
            try :
                # Neecsito comparar nombres pero ambos deben estar en lower
                pais_obj = Pais.objects.filter(nombre__iexact=row['PAIS_MONEDA'].lower()).first()
                tipo_moneda, created = TipoMoneda.objects.update_or_create(
                    codigo=row['MONEDA'],
                    defaults={
                        'nombre' : row['MONEDA.1'],
                        'pais' : pais_obj
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Tipo de Moneda {tipo_moneda.nombre} ({tipo_moneda.codigo}) {action}.")
            except Exception as e :
                print(f"Error al cargar tipo de moneda {row['MONEDA.1']}: {e}")

    # 12. Cargar Cláusulas de Compra Venta
    clausulas_df = pd.read_excel(data, 'Cláusulas de Compra Venta', skiprows=3)
    for _, row in clausulas_df.iterrows() :
        if pd.notna(row['CL_COMPRA']) :
            try :
                Clausula.objects.update_or_create(
                    codigo=row['CL_COMPRA'],
                    defaults={
                        'nombre' : row['NOMBRE_CLAUSULA'],
                        'sigla' : row['SIGLA_CLAUSULA']
                    }
                )
            except Exception as e :
                print(f"Error al cargar cláusula de compra venta {row['NOMBRE_CLAUSULA']}: {e}")

    return JsonResponse({'status' : 'success', 'message' : 'Datos cargados correctamente desde el archivo Excel.'})


def view_cargar_bahias(request) :
    bahias = obtener_bahias()
    for bahia in bahias :
        Sector.objects.update_or_create(
            id=bahia.get('IDBahia'),
            defaults={
                'cd_reparticion' : bahia.get('CdReparticion'),
                'nombre' : bahia.get('NMBahia'),
                'sitport_valor' : bahia.get('Valor'),
                'sitport_nom' : bahia.get('Nom')
            }
        )
        print('Sector cargado correctamente', bahia.get('NMBahia'))
    return JsonResponse({'status' : 'success', 'message' : 'Datos cargados correctamente desde el archivo Excel.'})


@extend_schema(
    tags=['bahías'],
    parameters=[
        OpenApiParameter(name='pais', description='Código del país para filtrar las bahías', required=False, type=int),
        OpenApiParameter(name='nombre', description='Nombre de la bahía (búsqueda parcial)', required=False, type=str),
        OpenApiParameter(name='sitport_valor', description='Valor específico del campo sitport para filtrar',
                         required=False, type=str),
    ]
)
class BahiaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las bahías.
    """
    queryset = Sector.objects.all()
    serializer_class = SectorSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = SectorFilter

class MuelleViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las muelles.
    """
    queryset = Muelle.objects.all()
    serializer_class = MuelleSerializer
    filter_backends = [DjangoFilterBackend]
    # filterset_class = MuelleFilter

@extend_schema(
    tags=['puertos'],
    parameters=[
        OpenApiParameter(name='pais', description='Código del país para filtrar los puertos', required=False, type=int),
    ]
)
class PuertoViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de los puertos.
    """
    queryset = Puerto.objects.filter(activo=True)
    serializer_class = PuertoSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_class = PuertoFilter

    @extend_schema(
        operation_id="cargar_puertos",
        tags=['puertos'],
        description="Carga de puertos desde un archivo JSON y asignación de sectores correspondientes",
        request={
            'application/json' : {
                'type' : 'array',
                'items' : {
                    'type' : 'object',
                    'properties' : {
                        'codigo' : {'type' : 'integer', 'description' : 'Código único del puerto'},
                        'codigo_bahia' : {'type' : 'integer',
                                          'description' : 'Código del sector (bahía) al que pertenece el puerto'},
                    },
                    'required' : ['codigo']
                }
            }
        },
        responses={
            200 : OpenApiExample(
                "Puertos actualizados correctamente",
                value={"message" : "Puertos actualizados correctamente."}
            ),
            400 : OpenApiExample(
                "Error: Sector no encontrado",
                value={"error" : "El sector con id X no existe."}
            ),
        },
        summary="Cargar y asociar puertos",
        deprecated=False
    )
    @action(detail=False, methods=['post'])  # Cambiamos detail=True a detail=False
    def cargar_puertos(self, request) :
        data = request.data

        for item in data :
            codigo = item.get('codigo')
            codigo_bahia = item.get('codigo_bahia')

            # Obtener o crear el puerto
            puerto, created = Puerto.objects.get_or_create(codigo=codigo)

            # Si el código de la bahía no es nulo, intenta obtener el sector (bahía)
            if codigo_bahia is not None :
                try :
                    sector = Sector.objects.get(id=codigo_bahia)
                except Sector.DoesNotExist :
                    return Response(
                        {"error" : f"El sector con id {codigo_bahia} no existe."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
            else :
                sector = None  # Si no hay sector, asignamos None

            # Actualizar los campos del puerto
            puerto.sector = sector
            puerto.save()

        return Response({"message" : "Puertos actualizados correctamente."}, status=status.HTTP_200_OK)


@extend_schema(tags=['tipos_operacion'])
class TipoOperacionViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las operaciones.
    """
    queryset = TipoOperacion.objects.all()  # El modelo TipoOperacion representa las operaciones
    serializer_class = TipoOperacionSerializer

    def list(self, request) :
        """Obtener todas las operaciones."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una operación específica."""
        operacion = self.get_object()
        serializer = self.get_serializer(operacion)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva operación."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una operación existente."""
        operacion = self.get_object()
        serializer = self.get_serializer(operacion, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una operación."""
        operacion = self.get_object()
        operacion.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['aduanas'])
class AduanaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las aduanas.
    """
    queryset = Aduana.objects.all()  # El modelo Aduana representa las aduanas
    serializer_class = AduanaSerializer

    def list(self, request) :
        """Obtener todas las aduanas."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una aduana específica."""
        aduana = self.get_object()
        serializer = self.get_serializer(aduana)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva aduana."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una aduana existente."""
        aduana = self.get_object()
        serializer = self.get_serializer(aduana, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una aduana."""
        aduana = self.get_object()
        aduana.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['tipos_operacion'])
class TipoOperacionViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las tipos de operaciones.
    """
    queryset = TipoOperacion.objects.all()  # El modelo TipoOperacion representa las tipos de operaciones
    serializer_class = TipoOperacionSerializer

    def list(self, request) :
        """Obtener todas las tipos de operaciones."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una tipo de operación específica."""
        tipo_operacion = self.get_object()
        serializer = self.get_serializer(tipo_operacion)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva tipo de operación."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una tipo de operación existente."""
        tipo_operacion = self.get_object()
        serializer = self.get_serializer(tipo_operacion, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una tipo de operación."""
        tipo_operacion = self.get_object()
        tipo_operacion.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['aduanas'])
class AduanaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las aduanas.
    """
    queryset = Aduana.objects.all()  # El modelo Aduana representa las aduanas
    serializer_class = AduanaSerializer

    def list(self, request) :
        """Obtener todas las aduanas."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una aduana específica."""
        aduana = self.get_object()
        serializer = self.get_serializer(aduana)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva aduana."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una aduana existente."""
        aduana = self.get_object()
        serializer = self.get_serializer(aduana, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una aduana."""
        aduana = self.get_object()
        aduana.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['tipos_carga'])
class TipoCargaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las tipos de carga.
    """
    queryset = TipoCarga.objects.all()  # El modelo TipoCarga representa las tipos de carga
    serializer_class = TipoCargaSerializer

    def list(self, request) :
        """Obtener todas las tipos de carga."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una tipo de carga específica."""
        tipo_carga = self.get_object()
        serializer = self.get_serializer(tipo_carga)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva tipo de carga."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una tipo de carga existente."""
        tipo_carga = self.get_object()
        serializer = self.get_serializer(tipo_carga, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una tipo de carga."""
        tipo_carga = self.get_object()
        tipo_carga.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['vias_transporte'])
class ViaTransporteViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las vias de transporte.
    """
    queryset = ViaTransporte.objects.all()  # El modelo ViaTransporte representa las vias de transporte
    serializer_class = ViaTransporteSerializer

    def list(self, request) :
        """Obtener todas las vias de transporte."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una via de transporte específica."""
        via_transporte = self.get_object()
        serializer = self.get_serializer(via_transporte)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva via de transporte."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una via de transporte existente."""
        via_transporte = self.get_object()
        serializer = self.get_serializer(via_transporte, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una via de transporte."""
        via_transporte = self.get_object()
        via_transporte.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['regimen_importacion'])
class RegimenImportacionViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de los regimenes de importación.
    """
    queryset = RegimenImportacion.objects.all()  # El modelo RegimenImportacion representa los regimenes de importación
    serializer_class = RegimenImportacionSerializer

    def list(self, request) :
        """Obtener todos los regimenes de importación."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener un regimen de importación específico."""
        regimen_importacion = self.get_object()
        serializer = self.get_serializer(regimen_importacion)
        return Response(serializer.data)

    def create(self, request) :
        """Crear un nuevo regimen de importación."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar un regimen de importación existente."""
        regimen_importacion = self.get_object()
        serializer = self.get_serializer(regimen_importacion, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar un regimen de importación."""
        regimen_importacion = self.get_object()
        regimen_importacion.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['modalidades_venta'])
class ModalidadVentaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las modalidades de venta.
    """
    queryset = ModalidadVenta.objects.all()  # El modelo ModalidadVenta representa las modalidades de venta
    serializer_class = ModalidadVentaSerializer

    def list(self, request) :
        """Obtener todas las modalidades de venta."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una modalidad de venta específica."""
        modalidad_venta = self.get_object()
        serializer = self.get_serializer(modalidad_venta)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva modalidad de venta."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una modalidad de venta existente."""
        modalidad_venta = self.get_object()
        serializer = self.get_serializer(modalidad_venta, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una modalidad de venta."""
        modalidad_venta = self.get_object()
        modalidad_venta.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['regiones'])
class RegionViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las regiones.
    """
    queryset = Region.objects.all()  # El modelo Region representa las regiones
    serializer_class = RegionSerializer

    def list(self, request) :
        """Obtener todas las regiones."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una region específica."""
        region = self.get_object()
        serializer = self.get_serializer(region)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva region."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una region existente."""
        region = self.get_object()
        serializer = self.get_serializer(region, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una region."""
        region = self.get_object()
        region.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['unidades_medida'])
class UnidadMedidaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las unidades de medida.
    """
    queryset = UnidadMedida.objects.all()  # El modelo UnidadMedida representa las unidades de medida
    serializer_class = UnidadMedidaSerializer

    def list(self, request) :
        """Obtener todas las unidades de medida."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una unidad de medida específica."""
        unidad_medida = self.get_object()
        serializer = self.get_serializer(unidad_medida)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva unidad de medida."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una unidad de medida existente."""
        unidad_medida = self.get_object()
        serializer = self.get_serializer(unidad_medida, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una unidad de medida."""
        unidad_medida = self.get_object()
        unidad_medida.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['tipos_moneda'])
class TipoMonedaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de los tipos de moneda.
    """
    queryset = TipoMoneda.objects.all()  # El modelo TipoMoneda representa los tipos de moneda
    serializer_class = TipoMonedaSerializer

    def list(self, request) :
        """Obtener todos los tipos de moneda."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener un tipo de moneda específico."""
        tipo_moneda = self.get_object()
        serializer = self.get_serializer(tipo_moneda)
        return Response(serializer.data)

    def create(self, request) :
        """Crear un nuevo tipo de moneda."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar un tipo de moneda existente."""
        tipo_moneda = self.get_object()
        serializer = self.get_serializer(tipo_moneda, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar un tipo de moneda."""
        tipo_moneda = self.get_object()
        tipo_moneda.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['clausulas'])
class ClausulaViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de las cláusulas.
    """
    queryset = Clausula.objects.all()  # El modelo Clausula representa las cláusulas
    serializer_class = ClausulaSerializer

    def list(self, request) :
        """Obtener todas las cláusulas."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener una cláusula específica."""
        clausula = self.get_object()
        serializer = self.get_serializer(clausula)
        return Response(serializer.data)

    def create(self, request) :
        """Crear una nueva cláusula."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar una cláusula existente."""
        clausula = self.get_object()
        serializer = self.get_serializer(clausula, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar una cláusula."""
        clausula = self.get_object()
        clausula.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=['paises'])
class PaisViewSet(viewsets.ModelViewSet) :
    """
    ViewSet para manejar las operaciones CRUD (GET, POST, PUT, DELETE) de los países.
    """
    queryset = Pais.objects.all()  # El modelo Pais representa los países
    serializer_class = PaisSerializer

    def list(self, request) :
        """Obtener todos los países."""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None) :
        """Obtener un país específico."""
        pais = self.get_object()
        serializer = self.get_serializer(pais)
        return Response(serializer.data)

    def create(self, request) :
        """Crear un nuevo país."""
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, pk=None) :
        """Actualizar un país existente."""
        pais = self.get_object()
        serializer = self.get_serializer(pais, data=request.data)
        if serializer.is_valid() :
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, pk=None) :
        """Eliminar un país."""
        pais = self.get_object()
        pais.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
