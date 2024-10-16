from django.db import models

from collection_manager.models import Aduana, Pais, Puerto, TipoCarga


class AgenciaTransporte(models.Model):
    nombre = models.CharField(max_length=255, null=True)
    rut = models.CharField(max_length=255, null=True)
    dig_v = models.CharField(max_length=255, null=True)

    def __str__(self):
        return f"{self.nombre} ({self.rut}-{self.dig_v})"

    def __repr__(self):
        return f"{self.nombre} ({self.rut}-{self.dig_v})"

    def get_rut(self):
        return f"{self.rut}-{self.dig_v}"

    class Meta:
        verbose_name = "Agencia de transporte"
        verbose_name_plural = "Agencias de transporte"
        ordering = ['nombre']

class Nave(models.Model):
    IMO = models.IntegerField(null=True)
    MMSI = models.IntegerField(null=True)
    nombre = models.CharField(max_length=255)
    tipo = models.CharField(max_length=255)
    fecha_construccion = models.DateField(null=True)
    bandera = models.CharField(max_length=255, null=True)
    pais = models.ForeignKey(Pais, on_delete=models.CASCADE, null=True)
    callsign = models.CharField(max_length=255, null=True)
    draft = models.IntegerField(null=True)
    length = models.IntegerField(null=True)
    width = models.IntegerField(null=True)
    deadweight = models.IntegerField(null=True)
    historial = models.TextField(null=True)

class Registro(models.Model):
    num_registro = models.IntegerField(null=True)
    puerto_embarque = models.ForeignKey(Puerto, on_delete=models.CASCADE, null=True, related_name='puerto_embarque')
    puerto_desembarque = models.ForeignKey(Puerto, on_delete=models.CASCADE, null=True, related_name='puerto_desembarque')
    fecha_aceptacion = models.DateField(null=True)
    aduana = models.ForeignKey(Aduana, on_delete=models.CASCADE, null=True, related_name='aduana')
    tipo_carga = models.ForeignKey(TipoCarga, on_delete=models.CASCADE, null=True, related_name='tipo_carga', to_field='codigo')
    agencia_transporte = models.ForeignKey(AgenciaTransporte, on_delete=models.CASCADE, null=True, related_name='agencia_transporte')
    nave = models.ForeignKey(Nave, on_delete=models.CASCADE, null=True, related_name='nave')
    nro_manifiesto = models.CharField(max_length=255, null=True)

    def __str__(self):
        return f"{self.num_registro}"
    def __repr__(self):
        return f"{self.num_registro}"
    class Meta:
        verbose_name = "Registro Importación/Exportación"
        verbose_name_plural = "Registros Importación/Exportación"
        ordering = ['num_registro', 'puerto_embarque', 'puerto_desembarque']

