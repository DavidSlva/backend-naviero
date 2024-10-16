from django.db import models
from simple_history.models import HistoricalRecords

class Pais(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    continente = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Pais"
        verbose_name_plural = "Paises"
        ordering = ['nombre']

class Region(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Región"
        verbose_name_plural = "Regiones"
        ordering = ['nombre']

class Sector(models.Model):
    id = models.IntegerField(primary_key=True)
    cd_reparticion = models.IntegerField(null=True)
    nombre = models.CharField(max_length=255, null=True)
    sitport_valor = models.CharField(max_length=255, null=True)
    sitport_nom = models.CharField(max_length=255, null=True)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.id})"
    def __repr__(self):
        return f"{self.nombre} ({self.id})"
    class Meta:
        verbose_name = "Sector"
        verbose_name_plural = "Sectores"
        ordering = ['nombre']

class Puerto(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    tipo = models.CharField(max_length=255)
    pais = models.ForeignKey(Pais, on_delete=models.CASCADE)
    latitud = models.FloatField(null=True)
    longitud = models.FloatField(null=True)
    zona_geografica = models.CharField(max_length=255, null=True)
    history = HistoricalRecords() 
    sector = models.ForeignKey(Sector, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Puerto"
        verbose_name_plural = "Puertos"
        ordering = ['nombre']

class TipoOperacion(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    nomber_a_consignar = models.CharField(max_length=255)
    ind_ingreso = models.BooleanField()
    ind_salida = models.BooleanField()
    operacion = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Tipo de operación"
        verbose_name_plural = "Tipos de operaciones"
        ordering = ['nombre']

class Aduana(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    longitud = models.FloatField(null=True)
    latitud = models.FloatField(null=True)

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Aduana"
        verbose_name_plural = "Aduanas"
        ordering = ['nombre']

class TipoCarga(models.Model):
    codigo = models.CharField(max_length=50, unique=True)
    nombre = models.CharField(max_length=255)
    descripcion = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Tipo de carga"
        verbose_name_plural = "Tipos de carga"
        ordering = ['nombre']

class ViaTransporte(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Via de transporte"
        verbose_name_plural = "Vias de transporte"
        ordering = ['nombre']

class RegimenImportacion(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    sigla = models.CharField(max_length=255)
    active = models.BooleanField()
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Regimen de importación"
        verbose_name_plural = "Regimenes de importación"
        ordering = ['nombre']

class ModalidadVenta(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    descripcion = models.TextField(null=True)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Modalidad de venta"
        verbose_name_plural = "Modalidades de venta"
        ordering = ['nombre']


class UnidadMedida(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    unidad = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Unidad de medida"
        verbose_name_plural = "Unidades de medida"
        ordering = ['nombre']

class TipoMoneda(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    pais = models.ForeignKey(Pais, on_delete=models.CASCADE)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Tipo de moneda"
        verbose_name_plural = "Tipos de moneda"
        ordering = ['nombre']

class Clausula(models.Model):
    codigo = models.IntegerField(primary_key=True)
    nombre = models.CharField(max_length=255)
    sigla = models.CharField(max_length=255)
    history = HistoricalRecords()  

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"
    def __repr__(self):
        return f"{self.nombre} ({self.codigo})"
    class Meta:
        verbose_name = "Clausula"
        verbose_name_plural = "Clausulas"
        ordering = ['nombre']


