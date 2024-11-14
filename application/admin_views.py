from django.urls import path
from django.contrib import admin
from django.shortcuts import render, redirect
from django.contrib import messages

from collection_manager.models import Sector
from application.services import obtener_restricciones

# Vista personalizada para obtener restricciones
def obtener_restricciones_admin_view(request):
    if request.method == "POST":
        bahia_id = request.POST.get("bahia")
        bahia = Sector.objects.get(id=bahia_id)
        restricciones = obtener_restricciones(bahia.id)

        # Aquí puedes hacer lo que necesites con las restricciones obtenidas
        messages.success(request, f"Restricciones obtenidas para {bahia.nombre}: {restricciones}")

        # Redirige a la misma página después de procesar el formulario
        return redirect("admin:obtener_restricciones")

    # Obtiene todas las bahías para mostrarlas en un formulario
    bahias = Sector.objects.all()
    return render(request, "admin/obtener_restricciones.html", {"bahias": bahias})

# Registrar la vista en el admin
class CustomAdminSite(admin.AdminSite):
    site_header = "Mi Admin Personalizado"  # Puedes cambiar los títulos
    site_title = "Título de Mi Admin"
    index_title = "Panel de Administración Personalizado"
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path("obtener-restricciones/", self.admin_view(obtener_restricciones_admin_view), name="obtener_restricciones"),
        ]
        return custom_urls + urls

# Instancia personalizada de admin
custom_admin_site = CustomAdminSite(name="custom_admin")
