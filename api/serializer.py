from api.models import BarcosRecalando
from rest_framework import serializers

class BarcosRecalandoSerializer(serializers.ModelSerializer):
    class Meta:
        model = BarcosRecalando
        fields = '__all__'