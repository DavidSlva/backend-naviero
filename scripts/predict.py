import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from time import sleep

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

# 2. Resamplear la serie temporal a una frecuencia diaria
ts_daily = df[target].resample('D').sum()

# Manejar valores faltantes por imputación con la mediana
ts_daily = ts_daily.fillna(ts_daily.median())

# Visualizar la serie temporal diaria
plt.figure(figsize=(15, 6))
plt.plot(ts_daily, label='Volumen Total (Kg) - Diario')
plt.title('Volumen Total a lo Largo del Tiempo (Diario)')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# Verificar el número de puntos de datos
print(f"Tamaño total de la serie diaria: {len(ts_daily)}")

# 3. División de Datos en Entrenamiento y Prueba
train_size = int(len(ts_daily) * 0.8)
train, test = ts_daily[:train_size], ts_daily[train_size:]

print(f"Tamaño del entrenamiento: {len(train)}")
print(f"Tamaño de la prueba: {len(test)}")

# 4. Modelo Simple 2: Exponential Smoothing (Holt-Winters)
# Simulación de barra de progreso para Exponential Smoothing
for _ in tqdm(range(100), desc="Entrenando Exponential Smoothing"):
    sleep(0.01)  # Simular tiempo de entrenamiento

# Entrenar el modelo Exponential Smoothing
print("Entrenando el modelo Exponential Smoothing...")
try:
    # Ajustar seasonal_periods según la frecuencia diaria (7 días ≈ 1 semana)
    model_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
    model_hw_fit = model_hw.fit()

    # Predecir
    pred_hw = model_hw_fit.forecast(steps=len(test))

    # Calcular métricas de error
    mae_hw = mean_absolute_error(test, pred_hw)
    rmse_hw = np.sqrt(mean_squared_error(test, pred_hw))

    print(f"Exponential Smoothing MAE: {mae_hw:.2f}")
    print(f"Exponential Smoothing RMSE: {rmse_hw:.2f}")

    # Visualizar las predicciones de Exponential Smoothing
    plt.figure(figsize=(15, 6))
    plt.plot(train, label='Entrenamiento')
    plt.plot(test, label='Prueba')
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
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))


# Función para crear secuencias
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Definir la longitud de la secuencia
SEQ_LENGTH = 14  # Reducido a 14 días para capturar dos semanas

# Crear secuencias para entrenamiento y prueba
X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# Reshape para LSTM [samples, time_steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"X_train shape: {X_train.shape}")  # Esperado: (156, 14, 1)
print(f"X_test shape: {X_test.shape}")    # Esperado: (368, 14, 1)

# Definir una clase de callback para la barra de progreso
class TQDMCallbackCustom(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.pbar = tqdm(total=total_epochs, desc='Entrenando LSTM', leave=False)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()

# Definir el modelo LSTM simplificado con Dropout y EarlyStopping
model_lstm = Sequential()
# Corregir la advertencia de Keras utilizando una capa Input explícita
model_lstm.add(Input(shape=(SEQ_LENGTH, 1)))
model_lstm.add(LSTM(20, activation='relu'))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# Definir el número de épocas y tamaño de batch
EPOCHS = 50
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

# 6. Modelo Alternativo: Facebook Prophet
# Preparar los datos para Prophet
df_prophet = ts_daily.reset_index().rename(columns={'FECHAACEPT': 'ds', target: 'y'})

# Inicializar el modelo Prophet con estacionalidades semanales y mensuales
model_prophet = Prophet(
    yearly_seasonality=False,      # Desactivar estacionalidad anual
    weekly_seasonality=True,       # Activar estacionalidad semanal
    daily_seasonality=False        # Desactivar estacionalidad diaria
)

# Agregar estacionalidad mensual personalizada
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Entrenar el modelo Prophet
model_prophet.fit(df_prophet[:train_size].rename(columns={'FECHAACEPT': 'ds', target: 'y'}))

# Crear un DataFrame para el futuro
future = model_prophet.make_future_dataframe(periods=len(test), freq='D')

# Predecir
forecast = model_prophet.predict(future)

# Seleccionar solo las predicciones del conjunto de prueba
forecast_test = forecast.set_index('ds').loc[test.index]

# Calcular métricas de error
mae_prophet = mean_absolute_error(test, forecast_test['yhat'])
rmse_prophet = np.sqrt(mean_squared_error(test, forecast_test['yhat']))

print(f"Prophet MAE: {mae_prophet:.2f}")
print(f"Prophet RMSE: {rmse_prophet:.2f}")

# Visualizar las predicciones de Prophet (completo)
model_prophet.plot(forecast)
plt.title('Modelo Prophet')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.show()

# 7. Comparación de Predicciones y Alineación de Datos

# Alinear las predicciones de LSTM con el conjunto de prueba
actual_lstm = test[SEQ_LENGTH:]
pred_lstm_series = pd.Series(pred_lstm.flatten(), index=actual_lstm.index)

# Alinear Exponential Smoothing con LSTM
pred_hw_aligned = pred_hw[-len(actual_lstm):].values if 'pred_hw' in locals() else [np.nan]*len(actual_lstm)
forecast_test_aligned = forecast_test['yhat'][-len(actual_lstm):].values

# Crear el DataFrame con las predicciones alineadas
predictions_df = pd.DataFrame({
    'Actual': actual_lstm,
    'Exponential Smoothing': pred_hw_aligned,
    'LSTM': pred_lstm_series,
    'Prophet': forecast_test_aligned
})

# Verificar que todas las columnas tienen la misma longitud
print("\nVerificación de longitudes en predictions_df:")
for col in predictions_df.columns:
    print(f"{col}: {len(predictions_df[col])}")

# Visualizar las predicciones alineadas
plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='black')
plt.plot(predictions_df['Exponential Smoothing'], label='Exponential Smoothing', color='green')
plt.plot(predictions_df['LSTM'], label='LSTM', color='blue')
plt.plot(predictions_df['Prophet'], label='Prophet', color='purple')
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
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.show()

# 8. Visualizar Predicciones de LSTM vs Actual

plt.figure(figsize=(15, 6))
plt.plot(predictions_df['Actual'], label='Actual', color='black')
plt.plot(predictions_df['LSTM'], label='LSTM Predicho', color='blue')
plt.title('LSTM: Predicción vs Actual')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# 9. Resumen de Métricas de Error
error_metrics = pd.DataFrame({
    'Modelo': ['Exponential Smoothing', 'LSTM', 'Prophet'],
    'MAE': [mae_hw if 'mae_hw' in locals() else np.nan, mae_lstm, mae_prophet],
    'RMSE': [rmse_hw if 'rmse_hw' in locals() else np.nan, rmse_lstm, rmse_prophet]
})

print("\nMétricas de Error de los Modelos:")
print(error_metrics)
