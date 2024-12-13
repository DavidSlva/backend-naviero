import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Configuración de estilo para las visualizaciones
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

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
date_columns = ["FECHAACEPT", "FECHAINFORMEEXP", "FECHADOCTOCANCELA", "FECHADOCTRANSP"]



for date_col in date_columns:
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format=None)  # Considera especificar el formato si es conocido

# Manejar valores faltantes en columnas clave
# Eliminamos las filas donde 'PAISDESTINO' o 'TOTALVALORFOB' son NaN
df_clean = df.dropna(subset=['PAISDESTINO', 'TOTALVALORFOB']).copy()
print(f"Datos limpiados. Número de filas después de limpieza: {df_clean.shape[0]}")

# Identificar columnas numéricas
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
print("Columnas numéricas:", numeric_cols.tolist())
print("¿'TOTALVALORFOB' está en columnas numéricas?", 'TOTALVALORFOB' in numeric_cols)

# Convertir 'TOTALVALORFOB' a numérico si no lo está
if 'TOTALVALORFOB' in df_clean.columns and df_clean['TOTALVALORFOB'].dtype not in ['int64', 'float64']:
    df_clean['TOTALVALORFOB'] = pd.to_numeric(df_clean['TOTALVALORFOB'], errors='coerce')
    print("Convertida 'TOTALVALORFOB' a numérico.")
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns  # Actualizar después de la conversión
    print("Columnas numéricas actualizadas:", numeric_cols.tolist())
    print("¿'TOTALVALORFOB' está en columnas numéricas ahora?", 'TOTALVALORFOB' in numeric_cols)

# Separar columnas enteras y de punto flotante
int_cols = df_clean[numeric_cols].select_dtypes(include=['int64']).columns
float_cols = df_clean[numeric_cols].select_dtypes(include=['float64']).columns

# Convertir columnas enteras a float usando .loc
df_clean.loc[:, int_cols] = df_clean[int_cols].astype(float)
print(f"Columnas convertidas de {int_cols.tolist()} a float.")

# Rellenar valores faltantes en columnas numéricas con la media usando .loc
df_clean.loc[:, numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
print("Valores faltantes rellenados con la media.")

# Verificar nuevamente si 'TOTALVALORFOB' está en numeric_cols
if 'TOTALVALORFOB' not in numeric_cols:
    raise ValueError("La variable objetivo 'TOTALVALORFOB' no está en las columnas numéricas.")

# -------------------------------
# d. Análisis Exploratorio de Datos (EDA)
# -------------------------------

# 1. Matriz de Correlación Pearson
print("\nGenerando Matriz de Correlación Pearson...")
corr_pearson = df_clean[numeric_cols].corr(method='pearson')
print("Columnas en la matriz de correlación Pearson:", corr_pearson.columns.tolist())
if 'TOTALVALORFOB' not in corr_pearson.columns:
    raise KeyError("La columna 'TOTALVALORFOB' no está en la matriz de correlación Pearson.")

plt.figure(figsize=(16, 12))
sns.heatmap(corr_pearson, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación Pearson')
plt.tight_layout()
plt.savefig(parquet_path.parent / "visualizaciones" / "matriz_correlacion_pearson.png")
plt.close()
print("Matriz de Correlación Pearson guardada en 'visualizaciones/matriz_correlacion_pearson.png'.")

# 2. Matriz de Correlación Spearman
print("\nGenerando Matriz de Correlación Spearman...")
corr_spearman = df_clean[numeric_cols].corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación Spearman')
plt.tight_layout()
plt.savefig(parquet_path.parent / "visualizaciones" / "matriz_correlacion_spearman.png")
plt.close()
print("Matriz de Correlación Spearman guardada en 'visualizaciones/matriz_correlacion_spearman.png'.")

# 3. Distribución de Variables Clave
variables_clave = ['TOTALVALORFOB', 'PESOBRUTOTOTAL', 'CANTIDADMERCANCIA', 'PORCENTAJEEXPPPAL']
for var in variables_clave:
    if var in df_clean.columns:
        # Histogramas
        plt.figure(figsize=(10, 6))
        sns.histplot(df_clean[var], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribución de {var}')
        plt.xlabel(var)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.savefig(parquet_path.parent / "visualizaciones" / f"distribucion_{var}.png")
        plt.close()
        print(f"Histograma de {var} guardado en 'visualizaciones/distribucion_{var}.png'.")

        # Boxplots
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df_clean[var], color='lightgreen')
        plt.title(f'Boxplot de {var}')
        plt.xlabel(var)
        plt.tight_layout()
        plt.savefig(parquet_path.parent / "visualizaciones" / f"boxplot_{var}.png")
        plt.close()
        print(f"Boxplot de {var} guardado en 'visualizaciones/boxplot_{var}.png'.")

# 4. Identificación de Variables Altamente Correlacionadas
threshold = 0.7
high_corr_pairs = corr_pearson.abs().unstack().sort_values(ascending=False)
high_corr_pairs = high_corr_pairs[high_corr_pairs > threshold]
# Eliminar duplicados y autocorrelaciones
high_corr_pairs = high_corr_pairs.drop_duplicates()
high_corr_pairs = high_corr_pairs[high_corr_pairs < 1.0]

print(f"\nPares de variables con correlación absoluta > {threshold}:")
print(high_corr_pairs)

# -------------------------------
# e. Preparación para Modelado de IA
# -------------------------------

# Definir la variable objetivo y las variables predictoras
# Suponiendo que queremos predecir 'TOTALVALORFOB'
target = 'TOTALVALORFOB'
predictors = [col for col in numeric_cols if col != target]

# Verificar que la variable objetivo y las predictoras existen
if target not in df_clean.columns:
    raise ValueError(f"La variable objetivo '{target}' no existe en el DataFrame.")

print(f"\nVariable objetivo: {target}")
print(f"Variables predictoras: {predictors}")

# 1. Correlación entre Predictoras y Objetivo
corr_with_target = corr_pearson[target].sort_values(ascending=False)
print(f"\nCorrelación de las variables predictoras con '{target}':")
print(corr_with_target)

# 2. Selección de Características
# Por ejemplo, seleccionar las 10 variables más correlacionadas
top_features = corr_with_target.abs().sort_values(ascending=False).head(11).index.tolist()  # Incluye 'TOTALVALORFOB'
top_features.remove(target)
print(f"\nTop 10 variables predictoras seleccionadas para el modelo:")
print(top_features)

# 3. Escalado de Datos (opcional pero recomendado para algunos modelos)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_clean_scaled = df_clean.copy()
df_clean_scaled[predictors] = scaler.fit_transform(df_clean_scaled[predictors])

print("\nDatos escalados utilizando StandardScaler.")

# -------------------------------
# f. Recomendaciones para Modelado de IA
# -------------------------------

print("\n--- Recomendaciones para Modelado de IA ---")
print("""
1. **Definición del Problema:**
   - Determina si el problema es de **regresión** (predecir un valor continuo) o **clasificación** (predecir categorías).
   - Suponiendo que 'TOTALVALORFOB' es continuo, el problema sería de regresión.

2. **Selección del Modelo:**
   - **Regresión Lineal**
   - **Regresión de Árboles (Random Forest, Gradient Boosting)**
   - **Máquinas de Soporte Vectorial (SVR)**
   - **Redes Neuronales**

3. **División de los Datos:**
   - Divide los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba).

4. **Entrenamiento y Evaluación del Modelo:**
   - Entrena diferentes modelos y evalúa su desempeño utilizando métricas como:
     - **MAE (Error Absoluto Medio)**
     - **MSE (Error Cuadrático Medio)**
     - **R² (Coeficiente de Determinación)**

5. **Ajuste de Hiperparámetros:**
   - Utiliza técnicas como **Grid Search** o **Random Search** para optimizar los hiperparámetros del modelo.

6. **Validación Cruzada:**
   - Implementa validación cruzada para asegurar que el modelo generaliza bien a datos no vistos.

7. **Interpretación del Modelo:**
   - Analiza la importancia de las características para entender qué variables influyen más en la predicción.

8. **Implementación y Monitoreo:**
   - Una vez satisfecho con el modelo, implementarlo y monitorear su desempeño en el tiempo.

**Nota:** Es crucial realizar un análisis más detallado y posiblemente más EDA según los resultados obtenidos para refinar el enfoque de modelado.
""")

# -------------------------------
# g. Guardar Métricas Adicionales (Opcional)
# -------------------------------

# Crear una carpeta para guardar las métricas
metrics_dir = parquet_path.parent / "metricas"
metrics_dir.mkdir(exist_ok=True)

# Guardar pares de variables altamente correlacionadas
high_corr_pairs.to_csv(metrics_dir / "altas_correlaciones.csv")
print(f"\nPares de variables altamente correlacionadas guardados en 'metricas/altas_correlaciones.csv'.")

# Guardar correlación con la variable objetivo
corr_with_target.to_csv(metrics_dir / "correlacion_con_objetivo.csv")
print(f"Correlación de variables con '{target}' guardada en 'metricas/correlacion_con_objetivo.csv'.")

# Guardar top 10 características
pd.Series(top_features).to_csv(metrics_dir / "top10_caracteristicas.csv", index=False, header=['Top_10_Caracteristicas'])
print(f"Top 10 características guardadas en 'metricas/top10_caracteristicas.csv'.")

print("\n--- Script de Análisis Completo ---")
