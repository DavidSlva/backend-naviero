# filters.py

import django_filters
from .models import Sector, Puerto


class SectorFilter(django_filters.FilterSet):
    pais = django_filters.NumberFilter(field_name='cd_reparticion')
    nombre = django_filters.CharFilter(field_name='nombre', lookup_expr='icontains')
    sitport_valor = django_filters.CharFilter(field_name='sitport_valor')

    class Meta:
        model = Sector
        fields = ['pais', 'nombre', 'sitport_valor']  # Campos que se pueden filtrar

class PuertoFilter(django_filters.FilterSet):
    pais = django_filters.NumberFilter(field_name='pais__codigo')  # Filtrado por el código de país relacionado

    class Meta:
        model = Puerto
        fields = ['pais']  # Campos que se pueden filtrar


