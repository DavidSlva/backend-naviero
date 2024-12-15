# interpreter/filters.py
import django_filters

from collection_manager.models import Puerto
from .models import VolumenPorPuerto, VolumenPredicho


class VolumenPorPuertoFilter(django_filters.FilterSet):
    anio = django_filters.NumberFilter(field_name='semana', lookup_expr='year', label='Año')
    puerto = django_filters.NumberFilter(field_name='puerto__codigo', lookup_expr='exact', label='Código de Puerto')

    class Meta:
        model = VolumenPorPuerto
        fields = ['puerto', 'anio']


class VolumenPredichoFilter(django_filters.FilterSet) :
    # Filtrar por año de la semana (campo 'semana')
    año = django_filters.NumberFilter(field_name='semana', lookup_expr='year')

    # Filtrar por puerto (relación ForeignKey)
    puerto = django_filters.ModelChoiceFilter(queryset=Puerto.objects.all())

    class Meta :
        model = VolumenPredicho
        fields = ['año', 'puerto']