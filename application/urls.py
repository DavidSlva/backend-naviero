
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('naves_recalando/', views.obtener_recalando, name='naves_recalando'),
]