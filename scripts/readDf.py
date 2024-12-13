import pandas as pd

# 1. Ruta al archivo .parquet
ruta_archivo = 'C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet'

# 2. Leer el archivo Parquet
df = pd.read_parquet(ruta_archivo, engine='pyarrow')  # O 'fastparquet'

# 3. Mostrar las primeras filas
print("Primeras filas del DataFrame:")
print(df.head())

# 4. Información general
print("\nInformación del DataFrame:")
print(df.info())

# 5. Descripción estadística
print("\nDescripción estadística:")
print(df.describe())

# 6. Verificar valores faltantes
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# 7. Nombres de las columnas
print("\nNombres de las columnas:")
print(df.columns.tolist())

# 8. Convertir columnas de fecha
columnas_fecha = ['FECHAACEPT', 'FECHAINFORMEEXP', 'FECHADOCTOCANCELA', 'FECHADOCTRANSP']

for columna in columnas_fecha:
    if columna in df.columns:
        df[columna] = pd.to_datetime(df[columna], format='%d%m%Y', errors='coerce')

# 9. Verificar tipos de datos después de la conversión
print("\nTipos de datos después de la conversión de fechas:")
print(df.dtypes)
