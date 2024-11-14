from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from application.models import Arista, Nodo, Test



admin.site.site_header = "Bienvenido a Marine Analytics"
admin.site.site_title = "Panel de Control"
admin.site.index_title = "Bienvenido al Panel de Administración"

@admin.register(Nodo)
class NodosAdmin(admin.ModelAdmin):
    list_display = ('codigo', 'nombre', 'pais', 'latitud', 'longitud')
    search_fields = ('codigo', 'nombre', 'pais')
    list_filter = ('pais',)
    ordering = ('codigo',)

@admin.register(Arista)
class AristasAdmin(admin.ModelAdmin):
    list_display = ('origen', 'destino', 'distancia')
    search_fields = ('origen__codigo', 'origen__nombre', 'destino__codigo', 'destino__nombre')
    list_filter = ('origen__pais', 'destino__pais')
    ordering = ('origen__codigo', 'destino__codigo')

# class APIExampleAdmin(admin.ModelAdmin):
#     change_list_template = "admin/api_example_list.html"

#     def get_urls(self):
#         from django.urls import path
#         urls = super().get_urls()
#         custom_urls = [
#             path('api-examples/', self.admin_site.admin_view(self.api_examples_view), name='api-examples')
#         ]
#         return custom_urls + urls

#     def api_examples_view(self, request):
#         # Aquí puedes manejar lógica personalizada si es necesario, pero en este caso
#         # los formularios simplemente redirigen a las rutas API.
#         pass

# admin.site.register(Test, APIExampleAdmin)
