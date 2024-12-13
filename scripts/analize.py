import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.impute import SimpleImputer

# Configuración de estilo de Seaborn
sns.set(style="whitegrid")

# 1. Cargar el DataFrame
ruta_archivo = 'C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean.parquet'
df = pd.read_parquet(ruta_archivo, engine='pyarrow')

# 2. Resumen General del Dataset
print("Información del DataFrame:")
print(df.info())
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# 3. Eliminar Columnas con Menos del 40% de Datos
# -------------------------------------------------
# Calcular el porcentaje de datos no nulos por columna
non_null_percentage = df.notnull().mean() * 100

# Definir el umbral
threshold = 40

# Identificar columnas con al menos 40% de datos no nulos
columns_to_keep = non_null_percentage[non_null_percentage >= threshold].index.tolist()

# Identificar columnas con menos del 40% de datos no nulos
columns_to_drop = non_null_percentage[non_null_percentage < threshold].index.tolist()

print(f"Se mantendrán {len(columns_to_keep)} columnas.")
print(f"Se eliminarán {len(columns_to_drop)} columnas con menos del {threshold}% de datos.")
print(f"Columnas a eliminar: {columns_to_drop}")

# Crear un nuevo DataFrame eliminando las columnas con menos del 40% de datos
df_cleaned = df[columns_to_keep]

# 4. Guardar el DataFrame Limpio en un Nuevo Archivo
# --------------------------------------------------
# Definir la ruta para el nuevo archivo limpio
ruta_archivo_limpio = 'C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet'

# Guardar el DataFrame limpio en un nuevo archivo Parquet
df_cleaned.to_parquet(ruta_archivo_limpio, engine='pyarrow', index=False)

print(f"DataFrame limpio guardado en: {ruta_archivo_limpio}")

# Opcional: Guardar en formato CSV
ruta_archivo_csv = 'C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.csv'
df_cleaned.to_csv(ruta_archivo_csv, index=False)
print(f"DataFrame limpio guardado en: {ruta_archivo_csv}")

# 5. Continuar con el EDA en el DataFrame Limpio
# ----------------------------------------------
# Reasignar el DataFrame limpio a 'df' para continuar con el EDA
df = df_cleaned

# Estadísticas Descriptivas para Variables Numéricas
print("\nDescripción estadística:")
print(df.describe())

# Estadísticas Descriptivas para Variables Categóricas
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_columns:
    print(f"\nValoraciones de la columna: {col}")
    print(df[col].value_counts().head(10))

# Visualización de Valores Faltantes
plt.figure(figsize=(20, 10))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Mapa de Valores Faltantes')
plt.show()

# Porcentaje de Valores Faltantes por Columna
missing_percentage = df.isnull().mean() * 100
missing_percentage = missing_percentage.sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=missing_percentage[:20], y=missing_percentage.index[:20], palette='viridis')
plt.xlabel('Porcentaje de Valores Faltantes')
plt.ylabel('Columnas')
plt.title('Top 20 Columnas con Más Valores Faltantes')
plt.show()

# Matriz de Correlación
corr_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# Correlación con la Variable Objetivo
target = 'PESOBRUTOTOTAL'  # Ajusta según tu variable objetivo
if target in corr_matrix.columns:
    correlations = corr_matrix[target].sort_values(ascending=False)
    print(f"\nCorrelaciones con {target}:")
    print(correlations)

    # Visualización de Correlaciones Positivas y Negativas
    top_positive_corr = correlations[correlations > 0.5].index.tolist()
    top_negative_corr = correlations[correlations < -0.5].index.tolist()

    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations[top_positive_corr].values, y=correlations[top_positive_corr].index, palette='viridis')
    plt.title(f'Correlaciones Positivas con {target}')
    plt.xlabel('Correlación')
    plt.ylabel('Variables')
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.barplot(x=correlations[top_negative_corr].values, y=correlations[top_negative_corr].index, palette='magma')
    plt.title(f'Correlaciones Negativas con {target}')
    plt.xlabel('Correlación')
    plt.ylabel('Variables')
    plt.show()
else:
    print(f"La variable objetivo '{target}' no está presente en la matriz de correlación.")

# Distribución de Variables Numéricas
numerical_columns = df.select_dtypes(include=[np.number]).columns

for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.histplot(df[col].dropna(), kde=True, bins=50, color='blue')
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

# Distribución de Variables Categóricas
for col in categorical_columns:
    plt.figure(figsize=(12, 6))
    sns.countplot(y=df[col].astype('category'), order=df[col].value_counts().index, palette='viridis')
    plt.title(f'Frecuencia de Categorías en {col}')
    plt.xlabel('Cuenta')
    plt.ylabel(col)
    plt.show()

# Boxplots para Detectar Outliers
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col], color='orange')
    plt.title(f'Boxplot de {col}')
    plt.xlabel(col)
    plt.show()

# Análisis de Series Temporales
if 'FECHAACEPT' in df.columns and 'PESOBRUTOTOTAL' in df.columns:
    df['FECHAACEPT'] = pd.to_datetime(df['FECHAACEPT'])
    volume_over_time = df.groupby('FECHAACEPT')['PESOBRUTOTOTAL'].sum().reset_index()

    plt.figure(figsize=(15, 6))
    sns.lineplot(data=volume_over_time, x='FECHAACEPT', y='PESOBRUTOTOTAL', color='blue')
    plt.title('Volumen Total a lo Largo del Tiempo')
    plt.xlabel('Fecha de Aceptación')
    plt.ylabel('Peso Bruto Total (Kg)')
    plt.show()

    # Descomposición de Series Temporales
    volume_over_time.set_index('FECHAACEPT', inplace=True)
    volume_monthly = volume_over_time['PESOBRUTOTOTAL'].resample('M').sum()

    decomposition = seasonal_decompose(volume_monthly, model='additive')
    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.show()

# Identificación de Multicolinealidad (VIF)
numeric_df = df.select_dtypes(include=[np.number]).dropna()

vif_data = pd.DataFrame()
vif_data["feature"] = numeric_df.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]

print("\nFactores de Inflación de la Varianza (VIF):")
print(vif_data.sort_values(by='VIF', ascending=False))

# Identificación de Outliers con Z-score
variables_para_outliers = ['PESOBRUTOTOTAL', 'TOTALVALORFOB', 'VALORFLETE']  # Ajusta según tus necesidades

for col in variables_para_outliers:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col], color='red')
    plt.title(f'Boxplot de {col}')
    plt.xlabel(col)
    plt.show()

    # Calcular Z-score
    df['Z_score'] = np.abs(stats.zscore(df[col].dropna()))
    outliers = df[df['Z_score'] > 3]
    print(f"Número de outliers en {col}: {outliers.shape[0]}")

# Creación de Variables Temporales Adicionales
df['Mes'] = df['FECHAACEPT'].dt.month
df['Trimestre'] = df['FECHAACEPT'].dt.quarter
df['Año'] = df['FECHAACEPT'].dt.year
df['DiaSemana'] = df['FECHAACEPT'].dt.dayofweek

# Creación de Variables Categóricas a partir de Variables Numéricas
df['CantidadCategoria'] = pd.cut(df['CANTIDADMERCANCIA'],
                                 bins=[0, 100, 500, 1000, np.inf],
                                 labels=['Baja', 'Media', 'Alta', 'Muy Alta'])

# Codificación de Variables Categóricas
categorical_features = df.select_dtypes(include=['object', 'category']).columns
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Escalado de Variables Numéricas
scaler = StandardScaler()
numerical_features = ['PESOBRUTOTOTAL', 'TOTALVALORFOB', 'VALORFLETE']  # Ajusta según tus necesidades
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Visualización de Correlación Después del Feature Engineering
corr_matrix_updated = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix_updated, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación Después del Feature Engineering')
plt.show()
