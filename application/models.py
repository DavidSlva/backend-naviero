from django.db import models

# Create your models here.
class Test(models.Model):
    numero1 = models.IntegerField()
    numero2 = models.IntegerField()
    resultado = models.IntegerField()

    def __str__(self):
        return f"Resultado: {self.resultado}"

class Nodo(models.Model):
    codigo = models.CharField(max_length=50, unique=True)
    nombre = models.CharField(max_length=255)
    pais = models.CharField(max_length=255)
    latitud = models.FloatField()
    longitud = models.FloatField()

    def __str__(self):
        return f"{self.nombre} ({self.codigo})"

class Arista(models.Model):
    origen = models.ForeignKey(Nodo, related_name='aristas_origen', on_delete=models.CASCADE)
    destino = models.ForeignKey(Nodo, related_name='aristas_destino', on_delete=models.CASCADE)
    distancia = models.FloatField()  # Puedes cambiar esto para representar el costo

    def __str__(self):
        return f"{self.origen} -> {self.destino} ({self.distancia} km)"
