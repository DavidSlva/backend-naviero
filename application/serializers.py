from rest_framework import serializers

from collection_manager.models import Sector

class SectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sector
        fields = ['id', 'cd_reparticion', 'nombre', 'sitport_valor', 'sitport_nom']

class SismoSerializer(serializers.Serializer):
    """
    Serializer para representar la información de un sismo.
    """
    fecha_ubicacion = serializers.CharField(max_length=255)
    profundidad = serializers.CharField(max_length=10)
    magnitud = serializers.DecimalField(max_digits=3, decimal_places=1)

class WaveSerializer(serializers.Serializer):
    """
    Serializer para representar la información de una ola.
    """
    hour = serializers.IntegerField()
    wave_height = serializers.DecimalField(max_digits=3, decimal_places=1)
    wave_direction = serializers.IntegerField()
    wave_period = serializers.DecimalField(max_digits=3, decimal_places=1)

class GrafoInfraestructuraSerializer(serializers.Serializer):
    puerto_origen = serializers.IntegerField()
    puerto_destino = serializers.IntegerField()