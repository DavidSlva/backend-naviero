
from django.urls import path
from .views import GetCurrentWaveView, GetCurrentWeatherView, GetDatosManifiesto, GetSismosChileView, ObtenerNaveView, PuertoNavesRecalandoView, PuertoSanAntonioView, UbicacionApiView, ObtenerRestriccionesView,ObtenerNavesView

urlpatterns = [
    path('puerto/naves/san_antonio/', PuertoSanAntonioView.as_view(), name='obtener_naves_san_antonio'),
    path('restricciones/bahias/', ObtenerRestriccionesView.as_view(), name='obtener_restricciones de una bahia'),
    path('restricciones/bahias/<int:id_bahia>/', ObtenerRestriccionesView.as_view(), name='obtener_restricciones de una bahia'),
    path('naves/<int:id_nave>/', ObtenerNavesView.as_view(), name='obtener_nave por id de sitport'),
    path('naves/ubicacion/imo/<int:imo>/', UbicacionApiView.as_view(), name='obtener_ubicacion_nave'),
    # path('puerto/<int:id_puerto>/naves/recalando/', PuertoNavesRecalandoView.as_view(), name='obtener_naves_recalando_puerto'),
    path('puerto/naves/recalando/', PuertoNavesRecalandoView.as_view(), name='obtener_naves_recalando_puerto'),
    path('puerto/<int:codigo_puerto>/clima/', GetCurrentWeatherView.as_view(), name='obtener_clima_puerto'),
    # REVISAR
    path('puerto/<int:codigo_puerto>/oleaje/', GetCurrentWaveView.as_view(), name='obtener_oleaje_puerto'),
    path('sismologia/', GetSismosChileView.as_view(), name='obtener_sismos_chile'),
    path('nave/<str:nombre_nave>/', ObtenerNaveView.as_view(), name='obtener_nave por coincidencia'),
    path('nave/manifiesto/<int:programacion>/', GetDatosManifiesto.as_view(), name='obtener_nave por manifiesto'),
]