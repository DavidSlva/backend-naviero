from django.db import models

# Create your models here.
class Test(models.Model):
    numero1 = models.IntegerField()
    numero2 = models.IntegerField()
    resultado = models.IntegerField()

    def __str__(self):
        return f"Resultado: {self.resultado}"