import numpy as np
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from collection_manager.models import (Pais, Puerto, TipoOperacion, Aduana, TipoCarga, ViaTransporte, RegimenImportacion, 
                     ModalidadVenta, Region, UnidadMedida, TipoMoneda, Clausula)

@api_view(['POST'])
def cargar_codigos(request):
    file_path = 'collection_manager/data/tablas_de_codigos.xlsx'
    data = pd.ExcelFile(file_path)
    print(f"Se encontraron las siguientes hojas: {data.sheet_names}")

    # 1. Cargar Países
    print("Cargando datos de Países...")
    paises_df = pd.read_excel(data, 'Países', skiprows=4)
    for _, row in paises_df.iterrows():
        if pd.notna(row['COD_PAIS']):
            try:
                pais, created = Pais.objects.update_or_create(
                    codigo=row['COD_PAIS'],
                    defaults={
                        'nombre': row['NOMBRE_PAIS'],
                        'continente': row['NOMBRE_CONTINENTE']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"País {pais.nombre} ({pais.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar país {row['NOMBRE_PAIS']}: {e}")

    # 2. Cargar Puertos
    print("Cargando datos de Puertos...")
    puertos_df = pd.read_excel(data, 'Puertos', skiprows=4)
    print(puertos_df)
    for _, row in puertos_df.iterrows():
        if pd.notna(row['COD_PAIS']) and pd.notna(row['COD_PUERTO']):
            try:
                pais_obj = Pais.objects.get(codigo=int(row['COD_PAIS']))
                puerto, created = Puerto.objects.update_or_create(
                    codigo=int(row['COD_PUERTO']),
                    defaults={
                        'nombre': row['NOMBRE_PUERTO'],
                        'tipo': row['TIPO_PUERTO'],
                        'pais': pais_obj,
                        'latitud': row['LATITUD'] if 'LATITUD' in row else None,
                        'longitud': row['LONGITUD'] if 'LONGITUD' in row else None,
                        # 'longitud': row['LONGITUD'],
                        # 'zona_geografica': row['ZONA_GEOGRAFICA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Puerto {puerto.nombre} ({puerto.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar puerto {row['NOMBRE_PUERTO']}: {e}")

    # 3. Cargar Tipos de Operación
    print("Cargando datos de Tipos de Operación...")
    tipos_operacion_df = pd.read_excel(data, 'Tipos de Operación', skiprows=3)
    for _, row in tipos_operacion_df.iterrows():
        if pd.notna(row['COD_TIPO_OPERACION']):
            try:
                tipo_operacion, created = TipoOperacion.objects.update_or_create(
                    codigo=row['COD_TIPO_OPERACION'],
                    defaults={
                        'nombre': row['NOMBRE_TIPO_OPERACION'],
                        'nomber_a_consignar': row['NOMBRE_A_CONSIGNAR'],
                        'ind_ingreso': row['INGRESO/SALIDA'] == 'Ingreso',
                        'ind_salida': row['INGRESO/SALIDA'] == 'Salida',
                        'operacion': row['OPERACION']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Tipo de Operación {tipo_operacion.nombre} ({tipo_operacion.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar tipo de operación {row['NOMBRE_TIPO_OPERACION']}: {e}")

    # 4. Cargar Aduanas
    print("Cargando datos de Aduanas...")
    aduanas_df = pd.read_excel(data, 'Aduanas', skiprows=3)
    for _, row in aduanas_df.iterrows():
        if pd.notna(row['COD_ADUANA_TRAMITACION']):
            try:
                aduana, created = Aduana.objects.update_or_create(
                    codigo=row['COD_ADUANA_TRAMITACION'],
                    defaults={
                        'nombre': row['NOMBRE_ADUANA'],
                        'latitud': row['LATITUD'],
                        'longitud': row['LONGITUD']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Aduana {aduana.nombre} ({aduana.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar aduana {row['NOMBRE_ADUANA']}: {e}")

    # 5. Cargar Tipos de Carga
    print("Cargando datos de Tipos de Carga...")
    tipos_carga_df = pd.read_excel(data, 'Tipos de Carga', skiprows=5)
    print(tipos_carga_df)
    for _, row in tipos_carga_df.iterrows():
        if pd.notna(row['COD_TIPO_CARGA']):
            try:
                tipo_carga, created = TipoCarga.objects.update_or_create(
                    codigo=row['COD_TIPO_CARGA'],
                    defaults={
                        'nombre': row['NOMBRE_TIPO_CARGA'],
                        'descripcion': row['DESCRIPCION_TIPO_CARGA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Tipo de Carga {tipo_carga.nombre} ({tipo_carga.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar tipo de carga {row['NOMBRE_TIPO_CARGA']}: {e}")

    # 6. Cargar Vías de Transporte
    print("Cargando datos de Vías de Transporte...")
    vias_transporte_df = pd.read_excel(data, 'Vías de Transporte', skiprows=3)
    for _, row in vias_transporte_df.iterrows():
        if pd.notna(row['COD_VIA_TRANSPORTE']) and row['COD_VIA_TRANSPORTE'] != np.nan:
            try:
                via_transporte, created = ViaTransporte.objects.update_or_create(
                    codigo=row['COD_VIA_TRANSPORTE'],
                    defaults={'nombre': row['NOMBRE_VIA_TRANSPORTE']}
                )
                action = "creado" if created else "actualizado"
                print(f"Vía de Transporte {via_transporte.nombre} ({via_transporte.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar vía de transporte {row['NOMBRE_VIA_TRANSPORTE']}: {e}")

    # 7. Cargar Régimen de Importación
    print("Cargando datos de Régimen de Importación...")
    regimen_importacion_df = pd.read_excel(data, 'Régimen de Importación', skiprows=3)
    print(regimen_importacion_df)
    for _, row in regimen_importacion_df.iterrows():
        if pd.notna(row['COD_RÉGIMEN_IMPORTACION']):
            try:
                regimen, created = RegimenImportacion.objects.update_or_create(
                    codigo=row['COD_RÉGIMEN_IMPORTACION'],
                    defaults={
                        'nombre': row['NOMBRE_RÉGIMEN_IMPORTACION'],
                        'sigla': row['SIGLA_RÉGIMEN_IMPORTACION'],
                        'active': True
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Régimen de Importación {regimen.nombre} ({regimen.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar régimen de importación {row['NOMBRE_RÉGIMEN_IMPORTACION']}: {e}")

    # 8. Cargar Modalidades de Venta
    print("Cargando datos de Modalidades de Venta...")
    modalidades_venta_df = pd.read_excel(data, 'Modalidades de Venta', skiprows=2)
    print(modalidades_venta_df)
    for _, row in modalidades_venta_df.iterrows():
        if pd.notna(row['COD_MODALIDAD_VENTA']):
            try:
                modalidad, created = ModalidadVenta.objects.update_or_create(
                    codigo=row['COD_MODALIDAD_VENTA'],
                    defaults={
                        'nombre': row['NOMBRE_MODALIDAD_VENTA'],
                        'descripcion': row['DESCRIPCION_MODALIDAD_VENTA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Modalidad de Venta {modalidad.nombre} ({modalidad.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar modalidad de venta {row['NOMBRE_MODALIDAD_VENTA']}: {e}")

    # 9. Cargar Regiones
    print("Cargando datos de Regiones...")
    regiones_df = pd.read_excel(data, 'Regiones', skiprows=3)
    print(regiones_df)
    for _, row in regiones_df.iterrows():
        if pd.notna(row['COD_REGION_ORIGEN']):
            try:
                region, created = Region.objects.update_or_create(
                    codigo=row['COD_REGION_ORIGEN'],
                    defaults={'nombre': row['NOMBRE_REGION']}
                )
                action = "creado" if created else "actualizado"
                print(f"Región {region.nombre} ({region.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar región {row['NOMBRE_REGION']}: {e}")

    # 10. Cargar Unidades de Medida
    print("Cargando datos de Unidades de Medida...")
    unidades_medida_df = pd.read_excel(data, 'Unidades de Medida', skiprows=3)
    for _, row in unidades_medida_df.iterrows():
        if pd.notna(row['COD_UNIDAD_MEDIDA']):
            try:
                unidad_medida, created = UnidadMedida.objects.update_or_create(
                    codigo=row['COD_UNIDAD_MEDIDA'],
                    defaults={
                        'nombre': row['NOMBRE_UNIDAD_MEDIDA'],
                        'unidad': row['UNIDAD_MEDIDA']
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Unidad de Medida {unidad_medida.nombre} ({unidad_medida.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar unidad de medida {row['NOMBRE_UNIDAD_MEDIDA']}: {e}")

    # 11. Cargar Moneda
    print("Cargando datos de Moneda...")
    moneda_df = pd.read_excel(data, 'Moneda', skiprows=4)
    print(moneda_df)
    for _, row in moneda_df.iterrows():
        if pd.notna(row['MONEDA']):
            try:
                # Neecsito comparar nombres pero ambos deben estar en lower
                pais_obj = Pais.objects.filter(nombre__iexact=row['PAIS_MONEDA'].lower()).first()
                tipo_moneda, created = TipoMoneda.objects.update_or_create(
                    codigo=row['MONEDA'],
                    defaults={
                        'nombre': row['MONEDA.1'],
                        'pais': pais_obj
                    }
                )
                action = "creado" if created else "actualizado"
                print(f"Tipo de Moneda {tipo_moneda.nombre} ({tipo_moneda.codigo}) {action}.")
            except Exception as e:
                print(f"Error al cargar tipo de moneda {row['MONEDA.1']}: {e}")

    # 12. Cargar Cláusulas de Compra Venta
    clausulas_df = pd.read_excel(data, 'Cláusulas de Compra Venta', skiprows=3)
    for _, row in clausulas_df.iterrows():
        if pd.notna(row['CL_COMPRA']):
            try:
                Clausula.objects.update_or_create(
                    codigo=row['CL_COMPRA'],
                    defaults={
                        'nombre': row['NOMBRE_CLAUSULA'],
                        'sigla': row['SIGLA_CLAUSULA']
                    }
                )
            except Exception as e:
                print(f"Error al cargar cláusula de compra venta {row['NOMBRE_CLAUSULA']}: {e}")

    return JsonResponse({'status': 'success', 'message': 'Datos cargados correctamente desde el archivo Excel.'})

