from django.contrib import admin

from interpreter.models import AgenciaTransporte, Registro

@admin.register(AgenciaTransporte)
class AgenciaTransporteAdmin(admin.ModelAdmin):
    list_display = ['nombre', 'rut', 'dig_v']
    search_fields = ['nombre', 'rut', 'dig_v']
    list_filter = ['dig_v']
    ordering = ['nombre']

@admin.register(Registro)
class RegistroAdmin(admin.ModelAdmin):
    list_display = ['num_registro', 'fecha_aceptacion', 'aduana', 'tipo_carga', 'agencia_transporte']
    search_fields = ['num_registro', 'fecha_aceptacion', 'aduana', 'tipo_carga', 'agencia_transporte']