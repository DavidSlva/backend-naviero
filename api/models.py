from django.db import models

class BarcosRecalando(models.Model):
    nombre_barco = models.CharField(max_length=255)
    puerto = models.CharField(max_length=50)
    tipo_barco = models.CharField(max_length=255)
    eslora = models.FloatField()
    bandera = models.CharField(max_length=50)
    agente = models.CharField(max_length=255)
    carga = models.CharField(max_length=255)
    detalle_operacion = models.CharField(max_length=255)
    fecha_entrada = models.DateTimeField()
    fecha_salida = models.DateTimeField()

    def __repr__(self):
        return f"{self.nombre_barco} ({self.puerto})"
    class Meta:
        verbose_name = "Barco recalando"
        verbose_name_plural = "Barcos recalando"
        ordering = ['nombre_barco']
        
class Puertos(models.Model):
    codigo = models.CharField(max_length=50)
    pais = models.CharField(max_length=255)
    continente = models.CharField(max_length=255)
    latitud = models.FloatField()
    longitud = models.FloatField()
# -- ===================================
# -- TABLAS PARA METADATAS DE LAS NAVES
# -- ===================================

# -- Tabla para almacenar los datos de los barcos recalando en los puertos Valparaíso y San Antonio
# CREATE TABLE IF NOT EXISTS  barcos_recalando (
#     id SERIAL PRIMARY KEY,
#     nombre_barco VARCHAR(255) NOT NULL,
#     puerto VARCHAR(50) NOT NULL,
#     tipo_barco VARCHAR(255),
#     eslora DOUBLE PRECISION,
#     bandera VARCHAR(50),
#     agente VARCHAR(255),
#     carga VARCHAR(255),
#     detalle_operacion VARCHAR(255),
#     fecha_entrada TIMESTAMP,
#     fecha_salida TIMESTAMP,
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- Tabla para almacenar la información de las naves según su ID
# CREATE TABLE IF NOT EXISTS  ficha_nave (
#     id_nave INT PRIMARY KEY,
#     nombre VARCHAR(255) NOT NULL,
#     tipo_nave VARCHAR(255),
#     bandera VARCHAR(50),
#     puerto_origen VARCHAR(255),
#     puerto_destino VARCHAR(255),
#     fecha_llegada TIMESTAMP,
#     fecha_salida TIMESTAMP,
#     eslora DOUBLE PRECISION,
#     manga DOUBLE PRECISION,
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- Tabla para almacenar las restricciones de las bahías
# CREATE TABLE IF NOT EXISTS  restricciones_bahias (
#     id SERIAL PRIMARY KEY,
#     bahia INT NOT NULL,
#     nombre_bahia VARCHAR(255),
#     restriccion_fecha TIMESTAMP,
#     estado VARCHAR(50),
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- Tabla para almacenar la latitud y longitud de los barcos según el número IMO
# CREATE TABLE IF NOT EXISTS  barcos_lat_long (
#     imo INT PRIMARY KEY,
#     nombre_barco VARCHAR(255),
#     latitud VARCHAR(50),
#     longitud VARCHAR(50),
#     informacion_completa TEXT,
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- ===================================
# -- TABLAS PARA AMENAZAS NATURALES
# -- ===================================

# -- Tabla para almacenar datos de clima
# CREATE TABLE IF NOT EXISTS  clima (
#     id SERIAL PRIMARY KEY,
#     ciudad VARCHAR(50) NOT NULL,
#     temperatura DECIMAL(5,2),
#     humedad INT,
#     descripcion VARCHAR(255),
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- Tabla para almacenar datos de oleaje de San Antonio
# CREATE TABLE IF NOT EXISTS  oleaje_san_antonio (
#     id SERIAL PRIMARY KEY,
#     hora TIMESTAMP,
#     altura DECIMAL(5,2),
#     direccion INT,
#     periodo DECIMAL(5,2),
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- Tabla para almacenar datos de oleaje de Valparaíso
# CREATE TABLE IF NOT EXISTS  oleaje_valparaiso (
#     id SERIAL PRIMARY KEY,
#     hora TIMESTAMP,
#     altura DECIMAL(5,2),
#     direccion INT,
#     periodo DECIMAL(5,2),
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );

# -- Tabla para almacenar datos de sismos en Chile
# CREATE TABLE IF NOT EXISTS  sismos_chile (
#     id SERIAL PRIMARY KEY,
#     fecha_lugar VARCHAR(255) NOT NULL,
#     profundidad VARCHAR(50),
#     magnitud DECIMAL(4,2),
#     fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );



