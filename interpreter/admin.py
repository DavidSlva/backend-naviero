from django.contrib import admin
from django.db.models import Count
from interpreter.models import AgenciaTransporte, Registro, VolumenTotal, VolumenPorPuerto, VolumenPredicho
from collection_manager.models import Aduana, Puerto, TipoCarga


@admin.register(AgenciaTransporte)
class AgenciaTransporteAdmin(admin.ModelAdmin) :
    list_display = ['nombre', 'rut', 'dig_v']
    search_fields = ['nombre', 'rut']
    list_filter = ['dig_v']
    ordering = ['nombre']

@admin.register(VolumenTotal)
class VolumenTotalAdmin(admin.ModelAdmin) :
    list_display = ['semana', 'volumen_total']
@admin.register(VolumenPorPuerto)
class VolumenPorPuertoAdmin(admin.ModelAdmin) :
    list_filter = ['puerto']
    search_fields = ['glosapuertoemb', 'semana']
    list_display = ['glosapuertoemb', 'semana', 'volumen', 'puerto']

@admin.register(Registro)
class RegistroAdmin(admin.ModelAdmin) :
    # Campos a mostrar en la lista
    list_display = ['num_registro', 'fecha_aceptacion', 'aduana', 'tipo_carga', 'agencia_transporte', 'puerto_embarque',
                    'puerto_desembarque']

    # Campos por los que se puede buscar.
    # Evitar búsquedas pesadas en muchos campos. Idealmente, tener índices en BD.
    search_fields = ['num_registro']

    # Filtros por relaciones y fecha. Esto permitirá al admin filtrar por puertos, aduanas, tipo de carga.
    list_filter = [
        'aduana',
        'tipo_carga',
        'agencia_transporte',
        'puerto_embarque',
        'puerto_desembarque',
        # Si es posible, filtrar por rangos de fechas también ayuda.
        ('fecha_aceptacion', admin.DateFieldListFilter),
    ]

    # Navegación por fecha en la parte superior
    date_hierarchy = 'fecha_aceptacion'

    # Para evitar cargar todos los objetos en menús desplegables, usamos raw_id_fields o autocomplete_fields
    # asegurándonos de haber configurado propermente los URLs y el search en esos modelos.
    raw_id_fields = ['puerto_embarque', 'puerto_desembarque', 'aduana', 'tipo_carga', 'agencia_transporte']

    # Sobrescribimos changelist_view para mostrar estadísticas interesantes:
    def changelist_view(self, request, extra_context=None) :
        # Llamar vista de lista normal
        response = super().changelist_view(request, extra_context=extra_context)

        # Si la tabla ya se ha cargado (normalmente en response.context_data)
        try :
            qs = response.context_data['cl'].queryset
        except (AttributeError, KeyError) :
            # Si no tenemos queryset (por ejemplo, error) no mostramos estadísticas
            return response

        # Obtener algunas estadísticas:
        # Conteo total de registros (post-filtros)
        total_registros = qs.count()

        # Conteo por Aduana (top 5)
        aduanas_top = qs.values('aduana__nombre').annotate(count=Count('aduana')).order_by('-count')[:5]

        # Conteo por Tipo de Carga (top 5)
        tipos_carga_top = qs.values('tipo_carga__nombre').annotate(count=Count('tipo_carga')).order_by('-count')[:5]

        # Conteo por Puerto de Embarque (top 5)
        puertos_top = qs.values('puerto_embarque__nombre').annotate(count=Count('puerto_embarque')).order_by('-count')[
                      :5]

        extra_context = extra_context or {}
        extra_context['total_registros'] = total_registros
        extra_context['aduanas_top'] = aduanas_top
        extra_context['tipos_carga_top'] = tipos_carga_top
        extra_context['puertos_top'] = puertos_top

        # Actualizar el contexto con nuestras estadísticas
        response.context_data.update(extra_context)

        return response



@admin.register(VolumenPredicho)
class VolumenPredichoAdmin(admin.ModelAdmin) :
    list_display = ['puerto', 'semana', 'volumen_predicho']