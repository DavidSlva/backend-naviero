from django.urls import path
from . import views
urlpatterns = [
    path('cargar_codigos/', views.cargar_codigos, name='cargar_codigos'),
    # path('barcos_recalando/', views.barcos_recalando, name='barcos_recalando'),
    # path('barcos/<int:id>', views.barco, name='barco'),
    # path('barcos/<int:id>/naves/', views.naves, name='naves'),
    # path('barcos/<int:id>/naves/<int:nave_id>', views.nave, name='nave'),
    # path('naves/', views.naves, name='naves'),
    # path('naves/<int:id>', views.nave, name='nave'),
]