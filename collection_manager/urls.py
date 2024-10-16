from django.urls import include, path
from . import views
from rest_framework.routers import DefaultRouter
router = DefaultRouter()
router.register(r'bahias', views.BahiaViewSet, basename='bahia')
urlpatterns = [
    path('cargar_codigos/', views.cargar_codigos, name='cargar_codigos'),
    path('cargar_bahias/', views.view_cargar_bahias, name='cargar_bahias'),
    path('', include(router.urls)),
    
]