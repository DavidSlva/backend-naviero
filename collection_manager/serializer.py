
from rest_framework import serializers

from collection_manager.models import Pais, Puerto, TipoOperacion, Aduana, TipoCarga, ViaTransporte, RegimenImportacion, \
    ModalidadVenta, Region, UnidadMedida, TipoMoneda, Clausula, Sector, Muelle
import math

class MuelleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Muelle
        fields = ['nombre', 'extension', 'tipo', 'ubicacion', 'puerto']

class PaisSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pais
        fields = ['codigo', 'nombre', 'continente']

class PuertoSerializer(serializers.ModelSerializer):

    class Meta:
        model = Puerto
        fields = ['codigo', 'nombre', 'tipo', 'pais', 'latitud', 'longitud', 'zona_geografica']

    def to_representation(self, instance):
        representation = super().to_representation(instance)
        if 'latitud' in representation and isinstance(representation['latitud'], float) and math.isnan(representation['latitud']):
            representation['latitud'] = None
        if 'longitud' in representation and isinstance(representation['longitud'], float) and math.isnan(representation['longitud']):
            representation['longitud'] = None
        return representation


class TipoOperacionSerializer(serializers.ModelSerializer):
    class Meta:
        model = TipoOperacion
        fields = ['codigo', 'nombre', 'nomber_a_consignar', 'ind_ingreso', 'ind_salida', 'operacion']

class AduanaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Aduana
        fields = ['codigo', 'nombre', 'longitud', 'latitud']

class TipoCargaSerializer(serializers.ModelSerializer):
    class Meta:
        model = TipoCarga
        fields = ['codigo', 'nombre', 'descripcion']

class ViaTransporteSerializer(serializers.ModelSerializer):
    class Meta:
        model = ViaTransporte
        fields = ['codigo', 'nombre']

class RegimenImportacionSerializer(serializers.ModelSerializer):
    class Meta:
        model = RegimenImportacion
        fields = ['codigo', 'nombre', 'sigla', 'active']

class ModalidadVentaSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModalidadVenta
        fields = ['codigo', 'nombre', 'descripcion']
        
class RegionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Region
        fields = ['codigo', 'nombre']

class UnidadMedidaSerializer(serializers.ModelSerializer):
    class Meta:
        model = UnidadMedida
        fields = ['codigo', 'nombre', 'unidad']

class TipoMonedaSerializer(serializers.ModelSerializer):
    class Meta:
        model = TipoMoneda
        fields = ['codigo', 'nombre', 'pais']

class ClausulaSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clausula
        fields = ['codigo', 'nombre', 'sigla']

class SectorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Sector
        fields = ['id', 'cd_reparticion', 'nombre', 'sitport_valor', 'sitport_nom']