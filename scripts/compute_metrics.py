import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# -------------------------------
# a. Definir la Ruta del Archivo Parquet
# -------------------------------

# Ruta al archivo Parquet generado anteriormente
parquet_path = Path(r"C:\Users\David\Documents\Github\Proyecto Semestral Grafos y Algoritmos\backend\downloads\exportaciones_2024_combinado_clean.parquet")

# Verificar si el archivo existe
if not parquet_path.exists():
    raise FileNotFoundError(f"El archivo {parquet_path} no existe. Verifica la ruta.")

# -------------------------------
# b. Cargar el Archivo Parquet
# -------------------------------

print("Cargando datos desde el archivo Parquet...")
df = pd.read_parquet(parquet_path)
print(f"Datos cargados exitosamente. Número de filas: {df.shape[0]}, Número de columnas: {df.shape[1]}")

# -------------------------------
# c. Limpieza y Preparación de Datos
# -------------------------------

# Asegurarse de que las columnas de fecha sean de tipo datetime
for date_col in ["FECHAACEPT", "FECHAINFORMEEXP", "FECHADOCTOCANCELA", "FECHADOCTRANSP"]:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Manejar valores faltantes en columnas clave
# Por ejemplo, si 'PAISDESTINO' o 'TOTALVALORFOB' son esenciales, puedes decidir cómo manejarlos
# Aquí, eliminamos las filas donde 'PAISDESTINO' o 'TOTALVALORFOB' son NaN
df_clean = df.dropna(subset=['PAISDESTINO', 'TOTALVALORFOB'])

print(f"Datos limpiados. Número de filas después de limpieza: {df_clean.shape[0]}")

# -------------------------------
# d. Cálculo de Métricas
# -------------------------------

# 1. Número Total de Exportaciones
total_exportaciones = len(df_clean)
print(f"Número Total de Exportaciones: {total_exportaciones}")

# 2. Valor Total de Exportaciones (Sumatoria de 'TOTALVALORFOB')
valor_total_exportaciones = df_clean['TOTALVALORFOB'].sum()
print(f"Valor Total de Exportaciones (FOB): {valor_total_exportaciones:,.2f}")

# 3. Exportaciones por País de Destino
exportaciones_por_pais = df_clean.groupby('PAISDESTINO')['TOTALVALORFOB'].sum().sort_values(ascending=False)
print("\nExportaciones por País de Destino:")
print(exportaciones_por_pais)

# 4. Top 10 Países por Valor de Exportación
top10_paises = exportaciones_por_pais.head(10)
print("\nTop 10 Países por Valor de Exportación:")
print(top10_paises)

# 5. Exportaciones por Código Arancelario (Producto)
# Suponiendo que 'CODIGOARANCEL' representa el código del producto
exportaciones_por_producto = df_clean.groupby('CODIGOARANCEL')['TOTALVALORFOB'].sum().sort_values(ascending=False)
print("\nExportaciones por Código Arancelario (Producto):")
print(exportaciones_por_producto)

# 6. Top 10 Productos por Valor de Exportación
top10_productos = exportaciones_por_producto.head(10)
print("\nTop 10 Productos por Valor de Exportación:")
print(top10_productos)

# 7. Tendencia Mensual de Exportaciones
# Utilizamos 'FECHAACEPT' como referencia temporal
if 'FECHAACEPT' in df_clean.columns:
    df_clean['Mes'] = df_clean['FECHAACEPT'].dt.to_period('M')
    exportaciones_mensuales = df_clean.groupby('Mes')['TOTALVALORFOB'].sum().sort_index()
    print("\nTendencia Mensual de Exportaciones:")
    print(exportaciones_mensuales)
else:
    print("\nColumna 'FECHAACEPT' no encontrada para análisis temporal.")

# 8. Exportaciones Promedio por País
exportaciones_promedio_pais = df_clean.groupby('PAISDESTINO')['TOTALVALORFOB'].mean().sort_values(ascending=False)
print("\nExportaciones Promedio por País:")
print(exportaciones_promedio_pais)

# -------------------------------
# e. Visualización de Métricas
# -------------------------------

# Crear una carpeta para guardar las visualizaciones
output_dir = parquet_path.parent / "visualizaciones"  # Usar 'parent' para obtener el directorio 'downloads'
output_dir.mkdir(exist_ok=True)

# 1. Top 10 Países por Valor de Exportación
plt.figure(figsize=(12, 8))
top10_paises.plot(kind='bar')
plt.title('Top 10 Países por Valor de Exportación (FOB)')
plt.xlabel('País de Destino')
plt.ylabel('Valor de Exportación (FOB)')
plt.tight_layout()
plt.savefig(output_dir / "top10_paises_exportacion_fob.png")
plt.close()
print(f"Gráfico de Top 10 Países guardado en {output_dir / 'top10_paises_exportacion_fob.png'}")

# 2. Top 10 Productos por Valor de Exportación
plt.figure(figsize=(12, 8))
top10_productos.plot(kind='bar', color='orange')
plt.title('Top 10 Productos por Valor de Exportación (FOB)')
plt.xlabel('Código Arancelario')
plt.ylabel('Valor de Exportación (FOB)')
plt.tight_layout()
plt.savefig(output_dir / "top10_productos_exportacion_fob.png")
plt.close()
print(f"Gráfico de Top 10 Productos guardado en {output_dir / 'top10_productos_exportacion_fob.png'}")

# 3. Tendencia Mensual de Exportaciones
if 'exportaciones_mensuales' in locals():
    plt.figure(figsize=(14, 7))
    exportaciones_mensuales.plot(kind='line', marker='o')
    plt.title('Tendencia Mensual de Exportaciones (FOB)')
    plt.xlabel('Mes')
    plt.ylabel('Valor de Exportación (FOB)')
    plt.tight_layout()
    plt.savefig(output_dir / "tendencia_mensual_exportaciones_fob.png")
    plt.close()
    print(f"Gráfico de Tendencia Mensual guardado en {output_dir / 'tendencia_mensual_exportaciones_fob.png'}")

# 4. Exportaciones por País (Gráfico de Pastel)
plt.figure(figsize=(10, 10))
exportaciones_por_pais.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Distribución de Exportaciones por País de Destino (FOB)')
plt.ylabel('')
plt.tight_layout()
plt.savefig(output_dir / "distribucion_exportaciones_pais_fob.png")
plt.close()
print(f"Gráfico de Distribución por País guardado en {output_dir / 'distribucion_exportaciones_pais_fob.png'}")

# -------------------------------
# f. Guardar Métricas en Archivos CSV (Opcional)
# -------------------------------

# Crear una carpeta para guardar las métricas
metrics_dir = parquet_path.parent / "metricas"
metrics_dir.mkdir(exist_ok=True)

# Guardar exportaciones por país
exportaciones_por_pais.to_csv(metrics_dir / "exportaciones_por_pais.csv")
print(f"Métricas exportaciones por país guardadas en {metrics_dir / 'exportaciones_por_pais.csv'}")

# Guardar exportaciones por producto
exportaciones_por_producto.to_csv(metrics_dir / "exportaciones_por_producto.csv")
print(f"Métricas exportaciones por producto guardadas en {metrics_dir / 'exportaciones_por_producto.csv'}")

# Guardar exportaciones mensuales
if 'exportaciones_mensuales' in locals():
    exportaciones_mensuales.to_csv(metrics_dir / "exportaciones_mensuales.csv")
    print(f"Métricas exportaciones mensuales guardadas en {metrics_dir / 'exportaciones_mensuales.csv'}")

# Guardar exportaciones promedio por país
exportaciones_promedio_pais.to_csv(metrics_dir / "exportaciones_promedio_pais.csv")
print(f"Métricas exportaciones promedio por país guardadas en {metrics_dir / 'exportaciones_promedio_pais.csv'}")

# -------------------------------
# g. Limpieza Final
# -------------------------------

# Opcional: eliminar la columna temporal 'Mes' si no se necesita
if 'Mes' in df_clean.columns:
    df_clean.drop(columns=['Mes'], inplace=True)

print("\nAnálisis de métricas completado exitosamente.")
