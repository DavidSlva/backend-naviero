import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, Attention
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from time import sleep
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction import FeatureHasher
from scipy.signal import find_peaks  # Importación para detección de picos

# Configuración de estilo de Seaborn
sns.set(style="whitegrid")

# 1. Cargar el DataFrame limpio
ruta_archivo_limpio = 'C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet'
df = pd.read_parquet(ruta_archivo_limpio, engine='pyarrow')
print(f"Datos cargados exitosamente. Número de filas: {df.shape[0]}, Número de columnas: {df.shape[1]}")

# Asegurarse de que 'FECHAACEPT' es datetime y está ordenado
df['FECHAACEPT'] = pd.to_datetime(df['FECHAACEPT'])
df = df.sort_values('FECHAACEPT')

# Establecer 'FECHAACEPT' como índice
df.set_index('FECHAACEPT', inplace=True)

# Seleccionar la columna objetivo
target = 'PESOBRUTOTOTAL'

# 2. Feature Engineering Avanzado
# Agregar características temporales
df['Mes'] = df.index.month
df['DiaSemana'] = df.index.dayofweek
df['EsFinDeSemana'] = df['DiaSemana'].apply(
    lambda x : 1 if x >= 5 else 0)  # 1 si es sábado o domingo, 0 en caso contrario

# Si hay días festivos o eventos especiales, agregarlos aquí
# Ejemplo:
# holidays = pd.DataFrame({
#     'ds': pd.to_datetime(['2024-12-25', '2025-01-01']),
#     'holiday': 'Navidad',
# })
# df = pd.merge(df, holidays, left_index=True, right_on='ds', how='left')
# df['EsFestivo'] = df['holiday'].apply(lambda x: 1 if pd.notnull(x) else 0)
# df.drop(columns=['holiday'], inplace=True)

# Resamplear la serie temporal a una frecuencia diaria
ts_daily = df[[target, 'Mes', 'DiaSemana', 'EsFinDeSemana']].resample('D').sum()

# Manejar valores faltantes por imputación con la mediana
ts_daily = ts_daily.fillna(ts_daily.median())

# Visualizar la serie temporal diaria
plt.figure(figsize=(15, 6))
plt.plot(ts_daily[target], label='Volumen Total (Kg) - Diario')
plt.title('Volumen Total a lo Largo del Tiempo (Diario)')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# Verificar el número de puntos de datos
print(f"Tamaño total de la serie diaria: {len(ts_daily)}")

# 3. División de Datos en Entrenamiento y Prueba
train_size = int(len(ts_daily) * 0.8)
train, test = ts_daily[:train_size], ts_daily[train_size :]

print(f"Tamaño del entrenamiento: {len(train)}")
print(f"Tamaño de la prueba: {len(test)}")

# 4. Modelo Exponential Smoothing (Holt-Winters)
# Simulación de barra de progreso para Exponential Smoothing
for _ in tqdm(range(100), desc="Entrenando Exponential Smoothing"):
    sleep(0.01)  # Simular tiempo de entrenamiento

# Entrenar el modelo Exponential Smoothing
print("Entrenando el modelo Exponential Smoothing...")
try:
    # Ajustar seasonal_periods según la frecuencia diaria (7 días ≈ 1 semana)
    model_hw = ExponentialSmoothing(train[target], trend='add', seasonal='add', seasonal_periods=7)
    model_hw_fit = model_hw.fit()

    # Predecir
    pred_hw = model_hw_fit.forecast(steps=len(test))

    # Calcular métricas de error
    mae_hw = mean_absolute_error(test[target], pred_hw)
    rmse_hw = np.sqrt(mean_squared_error(test[target], pred_hw))

    print(f"Exponential Smoothing MAE: {mae_hw:.2f}")
    print(f"Exponential Smoothing RMSE: {rmse_hw:.2f}")

    # Visualizar las predicciones de Exponential Smoothing
    plt.figure(figsize=(15, 6))
    plt.plot(train[target], label='Entrenamiento')
    plt.plot(test[target], label='Prueba')
    plt.plot(pred_hw, label='Predicción Exponential Smoothing', color='green')
    plt.title('Modelo Exponential Smoothing (Holt-Winters)')
    plt.xlabel('Fecha')
    plt.ylabel('Peso Bruto Total (Kg)')
    plt.legend()
    plt.show()
except ValueError as e:
    print(f"Error al entrenar Exponential Smoothing: {e}")

# 5. Modelo Complejo: Red Neuronal LSTM
# Normalizar los datos
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[target].values.reshape(-1, 1))
test_scaled = scaler.transform(test[target].values.reshape(-1, 1))

# Función para crear secuencias
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i :(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Definir la longitud de la secuencia
SEQ_LENGTH = 14  # Capturar dos semanas

# Crear secuencias para entrenamiento y prueba
X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# Reshape para LSTM [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}")  # Ejemplo: (N, 14, 1)
print(f"X_test shape: {X_test.shape}")    # Ejemplo: (N, 14, 1)

# Definir una clase de callback para la barra de progreso
class TQDMCallbackCustom(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.pbar = tqdm(total=total_epochs, desc='Entrenando LSTM', leave=False)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()

# Definir el modelo LSTM con mejoras
model_lstm = Sequential()
model_lstm.add(Input(shape=(SEQ_LENGTH, 1)))
model_lstm.add(LSTM(50, activation='relu', return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(25, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mae')  # Cambiado a MAE

# Definir el número de épocas y tamaño de batch
EPOCHS = 100
BATCH_SIZE = 32

# Definir el callback de EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo con la barra de progreso y EarlyStopping
print("Entrenando el modelo LSTM...")
tqdm_callback = TQDMCallbackCustom(total_epochs=EPOCHS)
history = model_lstm.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[tqdm_callback, early_stopping],
    verbose=0  # Desactivar la salida estándar
)

# Predecir
pred_lstm_scaled = model_lstm.predict(X_test)
pred_lstm = scaler.inverse_transform(pred_lstm_scaled)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcular métricas de error
mae_lstm = mean_absolute_error(y_test_inv, pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_lstm))

print(f"LSTM MAE: {mae_lstm:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")

# 6. Nuevo Modelo: XGBoost Regressor
print("Entrenando el modelo XGBoost...")

# Crear features de lag para XGBoost
def create_lag_features_extended(series, lag=14):
    df_lag = pd.DataFrame(series)
    for i in range(1, lag + 1):
        df_lag[f'lag_{i}'] = df_lag[target].shift(i)
    df_lag['Mes'] = df_lag['Mes']
    df_lag['DiaSemana'] = df_lag['DiaSemana']
    df_lag['EsFinDeSemana'] = df_lag['EsFinDeSemana']
    df_lag.dropna(inplace=True)
    return df_lag

# Crear el DataFrame con features de lag
df_xgb = create_lag_features_extended(ts_daily, lag=SEQ_LENGTH)

# Dividir en entrenamiento y prueba
train_xgb, test_xgb = df_xgb.iloc[:train_size], df_xgb.iloc[train_size :]

# Separar features y target
X_train_xgb = train_xgb.drop(columns=[target])
y_train_xgb = train_xgb[target]
X_test_xgb = test_xgb.drop(columns=[target])
y_test_xgb = test_xgb[target]

# Entrenar el modelo XGBoost
model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
model_xgb.fit(X_train_xgb, y_train_xgb)

# Predecir
pred_xgb = model_xgb.predict(X_test_xgb)

# Calcular métricas de error
mae_xgb = mean_absolute_error(y_test_xgb, pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, pred_xgb))

print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"XGBoost RMSE: {rmse_xgb:.2f}")

# 7. Nuevo Modelo: LightGBM Regressor
print("Entrenando el modelo LightGBM...")

# Entrenar el modelo LightGBM
model_lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
model_lgbm.fit(X_train_xgb, y_train_xgb)

# Predecir
pred_lgbm = model_lgbm.predict(X_test_xgb)

# Calcular métricas de error
mae_lgbm = mean_absolute_error(y_test_xgb, pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test_xgb, pred_lgbm))

print(f"LightGBM MAE: {mae_lgbm:.2f}")
print(f"LightGBM RMSE: {rmse_lgbm:.2f}")

# 8. Modelo Alternativo: Facebook Prophet
# Preparar los datos para Prophet
df_prophet = ts_daily.reset_index().rename(columns={'FECHAACEPT' : 'ds', target : 'y'})

# Inicializar el modelo Prophet con estacionalidades semanales y mensuales
model_prophet = Prophet(
    yearly_seasonality=False,  # Desactivar estacionalidad anual
    weekly_seasonality=True,  # Activar estacionalidad semanal
    daily_seasonality=False  # Desactivar estacionalidad diaria
)

# Agregar estacionalidad mensual personalizada
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Entrenar el modelo Prophet
print("Entrenando el modelo Prophet...")
model_prophet.fit(df_prophet[:train_size].rename(columns={'y' : 'y'}))

# Crear un DataFrame para el futuro
future = model_prophet.make_future_dataframe(periods=len(test), freq='D')

# Predecir
forecast = model_prophet.predict(future)

# Seleccionar solo las predicciones del conjunto de prueba
forecast_test = forecast.set_index('ds').loc[test.index]

# Calcular métricas de error
mae_prophet = mean_absolute_error(test[target], forecast_test['yhat'])
rmse_prophet = np.sqrt(mean_squared_error(test[target], forecast_test['yhat']))

print(f"Prophet MAE: {mae_prophet:.2f}")
print(f"Prophet RMSE: {rmse_prophet:.2f}")

# Visualizar las predicciones de Prophet (completo)
plt.figure(figsize=(15, 6))
model_prophet.plot(forecast)
plt.title('Modelo Prophet')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.show()

# 9. Comparación de Predicciones y Alineación de Datos

# Alinear las predicciones de LSTM con el conjunto de prueba
actual_lstm = test[target][SEQ_LENGTH :]
pred_lstm_series = pd.Series(pred_lstm.flatten(), index=actual_lstm.index)

# Alinear Exponential Smoothing, Prophet, XGBoost y LightGBM con LSTM
pred_hw_aligned = pred_hw[-len(actual_lstm) :].values if 'pred_hw' in locals() else [np.nan] * len(actual_lstm)
forecast_test_aligned = forecast_test['yhat'][-len(actual_lstm) :].values
pred_xgb_aligned = pred_xgb[-len(actual_lstm) :] if len(pred_xgb) >= len(actual_lstm) else np.pad(pred_xgb, (
len(actual_lstm) - len(pred_xgb), 0), 'constant')
pred_lgbm_aligned = pred_lgbm[-len(actual_lstm) :] if len(pred_lgbm) >= len(actual_lstm) else np.pad(pred_lgbm, (
len(actual_lstm) - len(pred_lgbm), 0), 'constant')

# Crear el DataFrame con las predicciones alineadas
predictions_df = pd.DataFrame({
    'Actual' : actual_lstm,
    'Exponential Smoothing' : pred_hw_aligned,
    'LSTM' : pred_lstm_series,
    'Prophet' : forecast_test_aligned,
    'XGBoost' : pred_xgb_aligned,
    'LightGBM' : pred_lgbm_aligned
})

# Verificar que todas las columnas tienen la misma longitud
print("\nVerificación de longitudes en predictions_df:")
for col in predictions_df.columns :
    print(f"{col}: {len(predictions_df[col])}")

# Mostrar las primeras filas de las predicciones para revisión
print("\nPrimeras filas de las predicciones:")
print(predictions_df.head())

# Guardar las predicciones en un archivo CSV para facilitar el envío
predictions_df.to_csv('predicciones_modelos.csv', index=True)
print("\nLas predicciones han sido guardadas en 'predicciones_modelos.csv'.")

# Visualizar las predicciones alineadas
plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='black')
plt.plot(predictions_df['Exponential Smoothing'], label='Exponential Smoothing', color='green')
plt.plot(predictions_df['LSTM'], label='LSTM', color='blue')
plt.plot(predictions_df['Prophet'], label='Prophet', color='purple')
plt.plot(predictions_df['XGBoost'], label='XGBoost', color='orange')
plt.plot(predictions_df['LightGBM'], label='LightGBM', color='brown')
plt.title('Comparación de Modelos de Predicción')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# Visualizar la pérdida del entrenamiento y validación del LSTM
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida Durante el Entrenamiento del LSTM')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MAE)')
plt.legend()
plt.show()

# 10. Visualizar Predicciones de LSTM vs Actual

plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='black')
plt.plot(predictions_df['LSTM'], label='LSTM Predicho', color='blue')
plt.title('LSTM: Predicción vs Actual')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# 11. Visualizar Predicciones de XGBoost vs Actual

plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='black')
plt.plot(predictions_df['XGBoost'], label='XGBoost Predicho', color='orange')
plt.title('XGBoost: Predicción vs Actual')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# 12. Visualizar Predicciones de LightGBM vs Actual

plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='black')
plt.plot(predictions_df['LightGBM'], label='LightGBM Predicho', color='brown')
plt.title('LightGBM: Predicción vs Actual')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# 13. Resumen de Métricas de Error
error_metrics = pd.DataFrame({
    'Modelo' : ['Exponential Smoothing', 'LSTM', 'Prophet', 'XGBoost', 'LightGBM'],
    'MAE' : [mae_hw if 'mae_hw' in locals() else np.nan, mae_lstm, mae_prophet, mae_xgb, mae_lgbm],
    'RMSE' : [rmse_hw if 'rmse_hw' in locals() else np.nan, rmse_lstm, rmse_prophet, rmse_xgb, rmse_lgbm]
})

print("\nMétricas de Error de los Modelos:")
print(error_metrics)

# Guardar las métricas de error en un archivo CSV
error_metrics.to_csv('metricas_error_modelos.csv', index=False)
print("\nLas métricas de error han sido guardadas en 'metricas_error_modelos.csv'.")

# 14. Validación Cruzada Temporal
print("\nImplementando Validación Cruzada Temporal...")

tscv = TimeSeriesSplit(n_splits=5)

# Inicializar listas para almacenar métricas de CV
cv_metrics = {
    'Fold': [],
    'Modelo': [],
    'MAE': [],
    'RMSE': []
}

for fold, (train_index, test_index) in enumerate(tscv.split(ts_daily)):
    print(f"\nFold {fold + 1}")
    train_cv, test_cv = ts_daily.iloc[train_index], ts_daily.iloc[test_index]

    # Modelo Exponential Smoothing
    try:
        model_hw_cv = ExponentialSmoothing(train_cv[target], trend='add', seasonal='add', seasonal_periods=7)
        model_hw_fit_cv = model_hw_cv.fit()
        pred_hw_cv = model_hw_fit_cv.forecast(steps=len(test_cv))
        mae_cv_hw = mean_absolute_error(test_cv[target], pred_hw_cv)
        rmse_cv_hw = np.sqrt(mean_squared_error(test_cv[target], pred_hw_cv))
        print(f"CV Exponential Smoothing MAE: {mae_cv_hw:.2f}, CV RMSE: {rmse_cv_hw:.2f}")

        # Almacenar métricas
        cv_metrics['Fold'].append(fold + 1)
        cv_metrics['Modelo'].append('Exponential Smoothing')
        cv_metrics['MAE'].append(mae_cv_hw)
        cv_metrics['RMSE'].append(rmse_cv_hw)
    except ValueError as e:
        print(f"Error en CV Exponential Smoothing: {e}")

    # Modelo XGBoost
    df_cv_xgb = create_lag_features_extended(train_cv, lag=SEQ_LENGTH)
    X_train_cv_xgb = df_cv_xgb.drop(columns=[target])
    y_train_cv_xgb = df_cv_xgb[target]

    df_cv_xgb_test = create_lag_features_extended(test_cv, lag=SEQ_LENGTH)
    X_test_cv_xgb = df_cv_xgb_test.drop(columns=[target])
    y_test_cv_xgb = df_cv_xgb_test[target]

    # Asegurarse de que haya suficientes datos después del lag
    if len(X_test_cv_xgb) > 0:
        model_xgb_cv = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
        model_xgb_cv.fit(X_train_cv_xgb, y_train_cv_xgb)
        pred_xgb_cv = model_xgb_cv.predict(X_test_cv_xgb)
        mae_cv_xgb = mean_absolute_error(y_test_cv_xgb, pred_xgb_cv)
        rmse_cv_xgb = np.sqrt(mean_squared_error(y_test_cv_xgb, pred_xgb_cv))
        print(f"CV XGBoost MAE: {mae_cv_xgb:.2f}, CV RMSE: {rmse_cv_xgb:.2f}")

        # Almacenar métricas
        cv_metrics['Fold'].append(fold + 1)
        cv_metrics['Modelo'].append('XGBoost')
        cv_metrics['MAE'].append(mae_cv_xgb)
        cv_metrics['RMSE'].append(rmse_cv_xgb)
    else:
        print("No hay suficientes datos para XGBoost en este fold.")

    # Modelo LightGBM
    if len(X_test_cv_xgb) > 0:
        model_lgbm_cv = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
        model_lgbm_cv.fit(X_train_cv_xgb, y_train_cv_xgb)
        pred_lgbm_cv = model_lgbm_cv.predict(X_test_cv_xgb)
        mae_cv_lgbm = mean_absolute_error(y_test_cv_xgb, pred_lgbm_cv)
        rmse_cv_lgbm = np.sqrt(mean_squared_error(y_test_cv_xgb, pred_lgbm_cv))
        print(f"CV LightGBM MAE: {mae_cv_lgbm:.2f}, CV RMSE: {rmse_cv_lgbm:.2f}")

        # Almacenar métricas
        cv_metrics['Fold'].append(fold + 1)
        cv_metrics['Modelo'].append('LightGBM')
        cv_metrics['MAE'].append(mae_cv_lgbm)
        cv_metrics['RMSE'].append(rmse_cv_lgbm)
    else:
        print("No hay suficientes datos para LightGBM en este fold.")

    # Puedes agregar más modelos aquí si lo deseas

# Convertir las métricas de CV a un DataFrame
cv_metrics_df = pd.DataFrame(cv_metrics)

print("\nMétricas de Validación Cruzada Temporal:")
print(cv_metrics_df)

# Guardar las métricas de CV en un archivo CSV
cv_metrics_df.to_csv('metricas_cv_modelos.csv', index=False)
print("\nLas métricas de validación cruzada han sido guardadas en 'metricas_cv_modelos.csv'.")

# 15. Feature Importance para XGBoost y LightGBM
print("\nImportancia de las Features para XGBoost:")
xgb_importance = model_xgb.feature_importances_
features = X_train_xgb.columns
importance_xgb_df = pd.DataFrame({
    'Feature': features,
    'Importancia': xgb_importance
}).sort_values(by='Importancia', ascending=False)
print(importance_xgb_df)

# Guardar la importancia de las features de XGBoost
importance_xgb_df.to_csv('importancia_features_xgboost.csv', index=False)
print("\nLa importancia de las features para XGBoost ha sido guardada en 'importancia_features_xgboost.csv'.")

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importance, y=features)
plt.title('Importancia de las Features - XGBoost')
plt.xlabel('Importancia')
plt.ylabel('Features')
plt.show()

print("\nImportancia de las Features para LightGBM:")
lgbm_importance = model_lgbm.feature_importances_
importance_lgbm_df = pd.DataFrame({
    'Feature': features,
    'Importancia': lgbm_importance
}).sort_values(by='Importancia', ascending=False)
print(importance_lgbm_df)

# Guardar la importancia de las features de LightGBM
importance_lgbm_df.to_csv('importancia_features_lightgbm.csv', index=False)
print("\nLa importancia de las features para LightGBM ha sido guardada en 'importancia_features_lightgbm.csv'.")

plt.figure(figsize=(10, 6))
sns.barplot(x=lgbm_importance, y=features)
plt.title('Importancia de las Features - LightGBM')
plt.xlabel('Importancia')
plt.ylabel('Features')
plt.show()

# 16. Detección y Comparación de Picos (Nueva Sección)
# Definir umbral para detectar picos, por ejemplo, 1.5 veces la mediana
umbral_pico = ts_daily[target].median() * 1.5

# Detectar picos en los datos reales
peaks_actual, _ = find_peaks(test[target], height=umbral_pico)
print(f"\nNúmero de picos reales en la prueba: {len(peaks_actual)}")

# Detectar picos en las predicciones de cada modelo
def detectar_picos(predicciones, umbral):
    peaks, _ = find_peaks(predicciones, height=umbral)
    return peaks

peaks_hw = detectar_picos(pred_hw_aligned, umbral_pico)
peaks_lstm = detectar_picos(pred_lstm.flatten(), umbral_pico)
peaks_prophet = detectar_picos(forecast_test_aligned, umbral_pico)
peaks_xgb = detectar_picos(pred_xgb_aligned, umbral_pico)
peaks_lgbm = detectar_picos(pred_lgbm_aligned, umbral_pico)

print(f"Número de picos detectados por Exponential Smoothing: {len(peaks_hw)}")
print(f"Número de picos detectados por LSTM: {len(peaks_lstm)}")
print(f"Número de picos detectados por Prophet: {len(peaks_prophet)}")
print(f"Número de picos detectados por XGBoost: {len(peaks_xgb)}")
print(f"Número de picos detectados por LightGBM: {len(peaks_lgbm)}")

# Comparar picos detectados con los reales
comparacion_picos = pd.DataFrame({
    'Modelo': ['Exponential Smoothing', 'LSTM', 'Prophet', 'XGBoost', 'LightGBM'],
    'Picos Detectados': [len(peaks_hw), len(peaks_lstm), len(peaks_prophet), len(peaks_xgb), len(peaks_lgbm)],
    'Picos Reales': [len(peaks_actual)] * 5
})

print("\nComparación de Detección de Picos:")
print(comparacion_picos)

# Guardar la comparación de picos en un archivo CSV
comparacion_picos.to_csv('comparacion_picos_modelos.csv', index=False)
print("\nLa comparación de detección de picos ha sido guardada en 'comparacion_picos_modelos.csv'.")
