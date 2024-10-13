from django.contrib import admin
from simple_history.admin import SimpleHistoryAdmin
from .models import (
    Pais, Puerto, TipoOperacion, Aduana, TipoCarga, ViaTransporte, RegimenImportacion, 
    ModalidadVenta, Region, UnidadMedida, TipoMoneda, Clausula
)

# Registrar cada modelo en el panel de administración utilizando SimpleHistoryAdmin
@admin.register(Pais)
class PaisAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'continente']
    search_fields = ['nombre', 'codigo']

@admin.register(Puerto)
class PuertoAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'pais', 'tipo']
    search_fields = ['nombre', 'codigo']
    list_filter = ['pais']

@admin.register(TipoOperacion)
class TipoOperacionAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'ind_ingreso', 'ind_salida']
    search_fields = ['nombre', 'codigo']

@admin.register(Aduana)
class AduanaAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'latitud', 'longitud']
    search_fields = ['nombre', 'codigo']

@admin.register(TipoCarga)
class TipoCargaAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'descripcion']
    search_fields = ['nombre', 'codigo']

@admin.register(ViaTransporte)
class ViaTransporteAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre']
    search_fields = ['nombre', 'codigo']

@admin.register(RegimenImportacion)
class RegimenImportacionAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'sigla', 'active']
    search_fields = ['nombre', 'codigo']
    list_filter = ['active']

@admin.register(ModalidadVenta)
class ModalidadVentaAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'descripcion']
    search_fields = ['nombre', 'codigo']

@admin.register(Region)
class RegionAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre']
    search_fields = ['nombre', 'codigo']

@admin.register(UnidadMedida)
class UnidadMedidaAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'unidad']
    search_fields = ['nombre', 'codigo']

@admin.register(TipoMoneda)
class TipoMonedaAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'pais']
    search_fields = ['nombre', 'codigo']
    list_filter = ['pais']

@admin.register(Clausula)
class ClausulaAdmin(SimpleHistoryAdmin):
    list_display = ['codigo', 'nombre', 'sigla']
    search_fields = ['nombre', 'codigo']
