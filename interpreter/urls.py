import os
from django.urls import path
from . import views

urlpatterns = [
    path('cargar_registros/', views.CargarRegistrosView.as_view(), name='cargar_registros'),
]