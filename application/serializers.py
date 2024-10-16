from rest_framework import serializers

from collection_manager.models import Sector

class SectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sector
        fields = ['id', 'cd_reparticion', 'nombre', 'sitport_valor', 'sitport_nom']
