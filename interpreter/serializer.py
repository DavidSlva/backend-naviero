from rest_framework import serializers
from interpreter.models import Registro, VolumenTotal, VolumenPorPuerto, VolumenPredicho


class RegistroSerializer(serializers.ModelSerializer):
    class Meta:
        model = Registro
        fields = '__all__'

class VolumenTotalSerializer(serializers.ModelSerializer):
    class Meta:
        model = VolumenTotal
        fields = '__all__'

class VolumenPorPuertoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VolumenPorPuerto
        fields = '__all__'

class VolumenTotalAnualSerializer(serializers.Serializer):
    anio = serializers.IntegerField()
    volumen_total = serializers.FloatField()

class VolumenPorPuertoAnualSerializer(serializers.Serializer):
    puerto_codigo = serializers.IntegerField()
    puerto_nombre = serializers.CharField()
    anio = serializers.IntegerField()
    volumen_total = serializers.FloatField()

class VolumenPredichoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VolumenPredicho
        fields = '__all__'