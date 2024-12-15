import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks

# Configuración de estilo de Seaborn
sns.set(style="whitegrid")


def cargar_modelos():
    print("\nCargando modelos guardados...")
    try:
        model_hw_fit = joblib.load('exponential_smoothing_model.pkl')
        print("Modelo Exponential Smoothing cargado.")
    except FileNotFoundError:
        print("Error: 'exponential_smoothing_model.pkl' no encontrado.")
        model_hw_fit = None

    try:
        model_lstm = load_model('lstm_model.h5')
        scaler_lstm = joblib.load('scaler_lstm.pkl')
        print("Modelo LSTM y scaler cargados.")
    except FileNotFoundError:
        print("Error: 'lstm_model.h5' o 'scaler_lstm.pkl' no encontrado.")
        model_lstm = None
        scaler_lstm = None

    try:
        model_xgb = joblib.load('xgboost_model.pkl')
        print("Modelo XGBoost cargado.")
    except FileNotFoundError:
        print("Error: 'xgboost_model.pkl' no encontrado.")
        model_xgb = None

    try:
        model_lgbm = joblib.load('lightgbm_model.pkl')
        print("Modelo LightGBM cargado.")
    except FileNotFoundError:
        print("Error: 'lightgbm_model.pkl' no encontrado.")
        model_lgbm = None

    try:
        model_prophet = joblib.load('prophet_model.pkl')
        print("Modelo Prophet cargado.")
    except FileNotFoundError:
        print("Error: 'prophet_model.pkl' no encontrado.")
        model_prophet = None

    return model_hw_fit, model_lstm, scaler_lstm, model_xgb, model_lgbm, model_prophet


def crear_features_lag(series, target, lag=14):
    df_lag = pd.DataFrame(series)
    for i in range(1, lag + 1):
        df_lag[f'lag_{i}'] = df_lag[target].shift(i)
    df_lag['Mes'] = df_lag['Mes']
    df_lag['DiaSemana'] = df_lag['DiaSemana']
    df_lag['EsFinDeSemana'] = df_lag['EsFinDeSemana']
    df_lag.dropna(inplace=True)
    return df_lag


def crear_secuencias(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def entrenar_exponential_smoothing(train, test, target):
    print("Entrenando el modelo Exponential Smoothing...")
    try:
        model_hw = ExponentialSmoothing(train[target], trend='add', seasonal='add', seasonal_periods=7)
        model_hw_fit = model_hw.fit()
        pred_hw = model_hw_fit.forecast(steps=len(test))
        mae_hw = mean_absolute_error(test[target], pred_hw)
        rmse_hw = np.sqrt(mean_squared_error(test[target], pred_hw))
        print(f"Exponential Smoothing MAE: {mae_hw:.2f}")
        print(f"Exponential Smoothing RMSE: {rmse_hw:.2f}")

        # Guardar el modelo
        joblib.dump(model_hw_fit, 'exponential_smoothing_model_prueba.pkl')
        print("Modelo Exponential Smoothing de prueba guardado en 'exponential_smoothing_model_prueba.pkl'.")

        return pred_hw, mae_hw, rmse_hw, model_hw_fit
    except ValueError as e:
        print(f"Error al entrenar Exponential Smoothing: {e}")
        return None, None, None, None


def entrenar_lstm(train, test, target, seq_length=14, epochs=100, batch_size=32):
    print("Entrenando el modelo LSTM...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[target].values.reshape(-1, 1))
    test_scaled = scaler.transform(test[target].values.reshape(-1, 1))
    print("Datos normalizados con StandardScaler.")

    X_train, y_train = crear_secuencias(train_scaled, seq_length)
    X_test, y_test = crear_secuencias(test_scaled, seq_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    print(f"Secuencias creadas. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Definir el modelo LSTM
    model_lstm = Sequential([
        Input(shape=(seq_length, 1)),
        LSTM(50, activation='relu', return_sequences=True),
        Dropout(0.2),
        LSTM(25, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_absolute_error')  # Cambiado a 'mean_absolute_error'
    print("Modelo LSTM compilado.")

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tqdm_callback = TQDMCallbackCustom(total_epochs=epochs)

    # Entrenamiento
    history = model_lstm.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[tqdm_callback, early_stopping],
        verbose=0
    )
    print("Entrenamiento LSTM completado.")

    # Predicción
    pred_lstm_scaled = model_lstm.predict(X_test)
    pred_lstm = scaler.inverse_transform(pred_lstm_scaled)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae_lstm = mean_absolute_error(y_test_inv, pred_lstm)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_lstm))
    print(f"LSTM MAE: {mae_lstm:.2f}")
    print(f"LSTM RMSE: {rmse_lstm:.2f}")

    # Guardar el modelo y el scaler
    model_lstm.save('lstm_model_prueba.h5')
    print("Modelo LSTM de prueba guardado en 'lstm_model_prueba.h5'.")
    joblib.dump(scaler, 'scaler_lstm_prueba.pkl')
    print("Scaler LSTM de prueba guardado en 'scaler_lstm_prueba.pkl'.")

    return pred_lstm, mae_lstm, rmse_lstm, history, model_lstm, scaler


def entrenar_xgboost(train, test, target, seq_length=14):
    print("Entrenando el modelo XGBoost...")

    def crear_features_lag(series, lag=14):
        df_lag = pd.DataFrame(series)
        for i in range(1, lag + 1):
            df_lag[f'lag_{i}'] = df_lag[target].shift(i)
        df_lag['Mes'] = df_lag['Mes']
        df_lag['DiaSemana'] = df_lag['DiaSemana']
        df_lag['EsFinDeSemana'] = df_lag['EsFinDeSemana']
        df_lag.dropna(inplace=True)
        return df_lag

    df_xgb = crear_features_lag(train, lag=seq_length)
    X_train_xgb = df_xgb.drop(columns=[target])
    y_train_xgb = df_xgb[target]

    df_test_xgb = crear_features_lag(test, lag=seq_length)
    X_test_xgb = df_test_xgb.drop(columns=[target])
    y_test_xgb = df_test_xgb[target]

    model_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
    model_xgb.fit(X_train_xgb, y_train_xgb)
    print("Modelo XGBoost entrenado.")

    # Guardar el modelo
    joblib.dump(model_xgb, 'xgboost_model_prueba.pkl')
    print("Modelo XGBoost de prueba guardado en 'xgboost_model_prueba.pkl'.")

    pred_xgb = model_xgb.predict(X_test_xgb)
    mae_xgb = mean_absolute_error(y_test_xgb, pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test_xgb, pred_xgb))
    print(f"XGBoost MAE: {mae_xgb:.2f}")
    print(f"XGBoost RMSE: {rmse_xgb:.2f}")

    return pred_xgb, mae_xgb, rmse_xgb, model_xgb, X_train_xgb.columns


def entrenar_lightgbm(train, test, target, seq_length=14):
    print("Entrenando el modelo LightGBM...")

    def crear_features_lag(series, lag=14):
        df_lag = pd.DataFrame(series)
        for i in range(1, lag + 1):
            df_lag[f'lag_{i}'] = df_lag[target].shift(i)
        df_lag['Mes'] = df_lag['Mes']
        df_lag['DiaSemana'] = df_lag['DiaSemana']
        df_lag['EsFinDeSemana'] = df_lag['EsFinDeSemana']
        df_lag.dropna(inplace=True)
        return df_lag

    df_lgbm = crear_features_lag(train, lag=seq_length)
    X_train_lgbm = df_lgbm.drop(columns=[target])
    y_train_lgbm = df_lgbm[target]

    df_test_lgbm = crear_features_lag(test, lag=seq_length)
    X_test_lgbm = df_test_lgbm.drop(columns=[target])
    y_test_lgbm = df_test_lgbm[target]

    model_lgbm = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
    model_lgbm.fit(X_train_lgbm, y_train_lgbm)
    print("Modelo LightGBM entrenado.")

    # Guardar el modelo
    joblib.dump(model_lgbm, 'lightgbm_model_prueba.pkl')
    print("Modelo LightGBM de prueba guardado en 'lightgbm_model_prueba.pkl'.")

    pred_lgbm = model_lgbm.predict(X_test_lgbm)
    mae_lgbm = mean_absolute_error(y_test_lgbm, pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test_lgbm, pred_lgbm))
    print(f"LightGBM MAE: {mae_lgbm:.2f}")
    print(f"LightGBM RMSE: {rmse_lgbm:.2f}")

    return pred_lgbm, mae_lgbm, rmse_lgbm, model_lgbm


def entrenar_prophet(train, test, target):
    print("Entrenando el modelo Prophet...")
    df_prophet = train.reset_index().rename(columns={'FECHAACEPT': 'ds', target: 'y'})

    model_prophet = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model_prophet.fit(df_prophet)
    print("Modelo Prophet entrenado.")

    # Guardar el modelo
    joblib.dump(model_prophet, 'prophet_model_prueba.pkl')
    print("Modelo Prophet de prueba guardado en 'prophet_model_prueba.pkl'.")

    future = model_prophet.make_future_dataframe(periods=len(test), freq='D')
    forecast = model_prophet.predict(future)
    forecast_test = forecast.set_index('ds').loc[test.index]

    mae_prophet = mean_absolute_error(test[target], forecast_test['yhat'])
    rmse_prophet = np.sqrt(mean_squared_error(test[target], forecast_test['yhat']))
    print(f"Prophet MAE: {mae_prophet:.2f}")
    print(f"Prophet RMSE: {rmse_prophet:.2f}")

    # Visualización
    plt.figure(figsize=(15, 6))
    model_prophet.plot(forecast)
    plt.title('Modelo Prophet')
    plt.xlabel('Fecha')
    plt.ylabel('Peso Bruto Total (Kg)')
    plt.show()

    return forecast_test['yhat'], mae_prophet, rmse_prophet, model_prophet


def comparar_modelos(test, pred_hw, pred_lstm, pred_xgb, pred_lgbm, pred_prophet, seq_length=14):
    # Alinear predicciones
    actual = test[target]
    # Exponential Smoothing: Predicción completa
    pred_hw_aligned = pred_hw.values if isinstance(pred_hw, pd.Series) else pred_hw
    # LSTM: secuencia desplazada
    pred_lstm_series = pd.Series(pred_lstm.flatten(), index=actual.index[seq_length:])
    # XGBoost y LightGBM: menos predicciones debido a lag
    pred_xgb_series = pd.Series(pred_xgb, index=actual.index[14:])
    pred_lgbm_series = pd.Series(pred_lgbm, index=actual.index[14:])
    # Prophet: Predicción completa
    pred_prophet_series = pd.Series(pred_prophet, index=actual.index)

    # Crear DataFrame de Predicciones
    predictions_df = pd.DataFrame({
        'Actual': actual,
        'Exponential Smoothing': pred_hw_aligned,
        'LSTM': pred_lstm_series,
        'Prophet': pred_prophet_series,
        'XGBoost': pred_xgb_series,
        'LightGBM': pred_lgbm_series
    })

    # Manejar NaN introducidos por la alineación
    predictions_df = predictions_df.dropna()

    # Guardar predicciones
    predictions_df.to_csv('predicciones_modelos_prueba.csv', index=True)
    print("Las predicciones han sido guardadas en 'predicciones_modelos_prueba.csv'.")

    # Visualización de las predicciones
    plt.figure(figsize=(15, 8))
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

    # Calcular Métricas de Error
    error_metrics = pd.DataFrame({
        'Modelo': ['Exponential Smoothing', 'LSTM', 'Prophet', 'XGBoost', 'LightGBM'],
        'MAE': [
            mean_absolute_error(predictions_df['Actual'], predictions_df['Exponential Smoothing']),
            mean_absolute_error(predictions_df['Actual'], predictions_df['LSTM']),
            mean_absolute_error(predictions_df['Actual'], predictions_df['Prophet']),
            mean_absolute_error(predictions_df['Actual'], predictions_df['XGBoost']),
            mean_absolute_error(predictions_df['Actual'], predictions_df['LightGBM'])
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['Exponential Smoothing'])),
            np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['LSTM'])),
            np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['Prophet'])),
            np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['XGBoost'])),
            np.sqrt(mean_squared_error(predictions_df['Actual'], predictions_df['LightGBM']))
        ]
    })

    print("\nMétricas de Error de los Modelos:")
    print(error_metrics)

    # Guardar métricas
    error_metrics.to_csv('metricas_error_modelos_prueba.csv', index=False)
    print("Las métricas de error han sido guardadas en 'metricas_error_modelos_prueba.csv'.")

    return predictions_df, error_metrics


def visualizar_importancia(model_xgb, model_lgbm, features):
    print("\nVisualizando la importancia de las features para XGBoost y LightGBM...")

    # XGBoost
    if model_xgb is not None:
        importances_xgb = model_xgb.feature_importances_
        importance_xgb_df = pd.DataFrame({
            'Feature': features,
            'Importancia': importances_xgb
        }).sort_values(by='Importancia', ascending=False)
        print("\nImportancia de las Features para XGBoost:")
        print(importance_xgb_df)
        importance_xgb_df.to_csv('importancia_features_xgboost_prueba.csv', index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importancia', y='Feature', data=importance_xgb_df)
        plt.title('Importancia de las Features - XGBoost')
        plt.xlabel('Importancia')
        plt.ylabel('Features')
        plt.show()
    else:
        print("Modelo XGBoost no disponible para visualizar importancia de features.")

    # LightGBM
    if model_lgbm is not None:
        importances_lgbm = model_lgbm.feature_importances_
        importance_lgbm_df = pd.DataFrame({
            'Feature': features,
            'Importancia': importances_lgbm
        }).sort_values(by='Importancia', ascending=False)
        print("\nImportancia de las Features para LightGBM:")
        print(importance_lgbm_df)
        importance_lgbm_df.to_csv('importancia_features_lightgbm_prueba.csv', index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importancia', y='Feature', data=importance_lgbm_df)
        plt.title('Importancia de las Features - LightGBM')
        plt.xlabel('Importancia')
        plt.ylabel('Features')
        plt.show()
    else:
        print("Modelo LightGBM no disponible para visualizar importancia de features.")


def detectar_picos_test(ts_test, preds, umbral_pico):
    print("\nDetección de picos en las predicciones...")

    picos_actual, _ = find_peaks(ts_test, height=umbral_pico)
    print(f"Número de picos reales en la prueba: {len(picos_actual)}")

    comparacion_picos = {'Modelo': [], 'Picos Detectados': [], 'Picos Reales': [len(picos_actual)] * 5}

    modelos = ['Exponential Smoothing', 'LSTM', 'Prophet', 'XGBoost', 'LightGBM']
    for modelo in modelos:
        pred = preds.get(modelo, [])
        if len(pred) > 0:
            peaks, _ = find_peaks(pred, height=umbral_pico)
            comparacion_picos['Modelo'].append(modelo)
            comparacion_picos['Picos Detectados'].append(len(peaks))
            print(f"Número de picos detectados por {modelo}: {len(peaks)}")
        else:
            comparacion_picos['Modelo'].append(modelo)
            comparacion_picos['Picos Detectados'].append(0)
            print(f"Número de picos detectados por {modelo}: 0")

    comparacion_df = pd.DataFrame(comparacion_picos)
    print("\nComparación de Detección de Picos:")
    print(comparacion_df)

    comparacion_df.to_csv('comparacion_picos_modelos_prueba.csv', index=False)
    print("La comparación de detección de picos ha sido guardada en 'comparacion_picos_modelos_prueba.csv'.")

    return comparacion_df


# Flujo principal
if __name__ == "__main__":
    # Ruta del archivo de datos sintéticos
    ruta_archivo_sintetico = 'datos_sinteticos.parquet'
    target = 'PESOBRUTOTOTAL'

    # 1. Crear Datos Sintéticos
    def crear_datos_sinteticos():
        np.random.seed(42)  # Para reproducibilidad
        fechas = pd.date_range(start='2023-01-01', periods=200, freq='D')
        tendencia = np.linspace(50, 100, 200)  # Tendencia lineal de 50 a 100
        estacionalidad_semanal = 10 * np.sin(2 * np.pi * fechas.dayofweek / 7)
        ruido = np.random.normal(0, 5, 200)  # Ruido aleatorio
        peso_bruto_total = tendencia + estacionalidad_semanal + ruido
        df_sintetico = pd.DataFrame({
            'FECHAACEPT': fechas,
            'PESOBRUTOTOTAL': peso_bruto_total
        })
        print("Conjunto de datos sintético creado.")
        return df_sintetico

    df_sintetico = crear_datos_sinteticos()

    # Guardar los datos sintéticos (opcional)
    df_sintetico.to_parquet(ruta_archivo_sintetico, index=False)
    print(f"Datos sintéticos guardados en '{ruta_archivo_sintetico}'.")

    # 2. Preprocesar Datos
    ts_daily = preprocesar_datos(df_sintetico, target)

    # 3. Dividir Datos
    train, test = dividir_datos(ts_daily, target)

    # 4. Cargar Modelos Guardados
    model_hw_fit, model_lstm, scaler_lstm, model_xgb, model_lgbm, model_prophet = cargar_modelos()

    # 5. Realizar Predicciones
    pred_hw, pred_lstm, pred_xgb, pred_lgbm, pred_prophet = entrenar_exponential_smoothing(
        train, test, target
    )  # Si deseas reentrenar aquí

    # Si no deseas reentrenar, usa la función de predicción
    # pred_hw, pred_lstm, pred_xgb, pred_lgbm, pred_prophet = realizar_predicciones(
    #     train, test, target, model_hw_fit, model_lstm, scaler_lstm, model_xgb, model_lgbm, model_prophet
    # )

    # 6. Comparar Predicciones con Valores Reales
    predictions_df, error_metrics = comparar_modelos(
        test, pred_hw, pred_lstm, pred_xgb, pred_lgbm, pred_prophet, seq_length=14
    )

    # 7. Visualizar Importancia de las Features
    if model_xgb is not None and model_lgbm is not None:
        importancia_features(model_xgb, model_xgb.feature_names_in_, 'XGBoost')
        importancia_features(model_lgbm, model_lgbm.feature_name_, 'LightGBM')
    else:
        print("No se pueden visualizar las importancias de las features porque algunos modelos no están disponibles.")

    # 8. Detección de Picos
    umbral_pico = ts_daily[target].median() * 1.5
    preds = {
        'Exponential Smoothing': pred_hw_aligned if pred_hw is not None else [],
        'LSTM': pred_lstm_series.values,
        'Prophet': pred_prophet_series.values,
        'XGBoost': pred_xgb_series.values,
        'LightGBM': pred_lgbm_series.values
    }
    comparacion_picos = detectar_picos_test(predictions_df['Actual'], preds, umbral_pico)
