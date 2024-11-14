from django.urls import include, path
from . import views
from rest_framework.routers import DefaultRouter
router = DefaultRouter()
router.register(r'bahias', views.BahiaViewSet, basename='bahia')
router.register(r'paises', views.PaisViewSet, basename='pais')
router.register(r'puertos', views.PuertoViewSet, basename='puertos')
router.register(r'tipos_operacion', views.TipoOperacionViewSet, basename='tipo_operacion')
router.register(r'aduanas', views.AduanaViewSet, basename='aduana')
router.register(r'tipos_carga', views.TipoCargaViewSet, basename='tipo_carga')
router.register(r'vias_transporte', views.ViaTransporteViewSet, basename='via_transporte')
router.register(r'regimen_importacion', views.RegimenImportacionViewSet, basename='regimen_importacion')
router.register(r'modalidades_venta', views.ModalidadVentaViewSet, basename='modalidad_venta')
router.register(r'regiones', views.RegionViewSet, basename='region')
router.register(r'unidades_medida', views.UnidadMedidaViewSet, basename='unidad_medida')
router.register(r'tipos_moneda', views.TipoMonedaViewSet, basename='tipo_moneda')
router.register(r'clausulas', views.ClausulaViewSet, basename='clausula')

urlpatterns = [
    path('cargar_codigos/', views.cargar_codigos, name='cargar_codigos'),
    path('cargar_aduanas/', views.cargar_aduanas, name='cargar_aduanas'),   
    path('cargar_bahias/', views.view_cargar_bahias, name='cargar_bahias'),
    path('cargar_rutas/', views.cargar_rutas_importantes, name='cargar_rutas'),
    path('', include(router.urls)),
    
]