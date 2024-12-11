import os
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from . import views

router = DefaultRouter()

router.register(r'registros', views.RegistrosViewSet, basename='registros')

urlpatterns = [
    path('cargar_registros/', views.CargarRegistrosView.as_view(), name='cargar_registros'),
    path('', include(router.urls)),
]
