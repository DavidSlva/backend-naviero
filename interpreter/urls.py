import os
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()

router.register(r'registros', views.RegistrosViewSet, basename='registros')
router.register(r'datos_generales', views.DatosGeneralesViewSet, basename='datos_generales')
router.register(r'volumen_total', views.VolumenTotalViewSet, basename='volumen_total')
router.register(r'volumen_por_puerto', views.VolumenPorPuertoViewSet, basename='volumen_por_puerto')
router.register(r'volumen_anual', views.VolumenAnualViewSet, basename='volumen_anual')
router.register(r'predicciones', views.PrediccionesViewSet, basename='predicciones')


urlpatterns = [
    path('cargar_registros/', views.CargarRegistrosView.as_view(), name='cargar_registros'),
    path('', include(router.urls)),
]
