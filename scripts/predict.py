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
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from scipy.signal import find_peaks
import joblib

# Configuración de estilo de Seaborn
sns.set(style="whitegrid")


def cargar_datos(ruta):
    df = pd.read_parquet(ruta, engine='pyarrow')
    print(f"Datos cargados exitosamente. Número de filas: {df.shape[0]}, Número de columnas: {df.shape[1]}")
    return df


def preprocesar_datos(df, target):
    # Convertir y ordenar fechas
    df['FECHAACEPT'] = pd.to_datetime(df['FECHAACEPT'])
    df = df.sort_values('FECHAACEPT')
    df.set_index('FECHAACEPT', inplace=True)
    print("Datos ordenados por 'FECHAACEPT' y establecidos como índice.")

    # Feature Engineering
    df['Mes'] = df.index.month
    df['DiaSemana'] = df.index.dayofweek
    df['EsFinDeSemana'] = df['DiaSemana'].apply(lambda x: 1 if x >= 5 else 0)
    print("Características temporales agregadas: Mes, DiaSemana, EsFinDeSemana.")

    # Resamplear a diario y manejar valores faltantes
    ts_daily = df[[target, 'Mes', 'DiaSemana', 'EsFinDeSemana']].resample('D').sum()
    ts_daily = ts_daily.fillna(ts_daily.median())
    print(f"Serie temporal resampleada a frecuencia diaria. Tamaño: {ts_daily.shape}")

    # Visualización inicial
    plt.figure(figsize=(15, 6))
    plt.plot(ts_daily[target], label='Volumen Total (Kg) - Diario')
    plt.title('Volumen Total a lo Largo del Tiempo (Diario)')
    plt.xlabel('Fecha')
    plt.ylabel('Peso Bruto Total (Kg)')
    plt.legend()
    plt.show()

    return ts_daily


def dividir_datos(ts, target, train_ratio=0.8):
    train_size = int(len(ts) * train_ratio)
    train, test = ts[:train_size], ts[train_size:]
    print(f"Datos divididos en entrenamiento ({len(train)}) y prueba ({len(test)}).")
    return train, test


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

        # Visualización
        plt.figure(figsize=(15, 6))
        plt.plot(train[target], label='Entrenamiento')
        plt.plot(test[target], label='Prueba')
        plt.plot(pred_hw, label='Predicción Exponential Smoothing', color='green')
        plt.title('Modelo Exponential Smoothing (Holt-Winters)')
        plt.xlabel('Fecha')
        plt.ylabel('Peso Bruto Total (Kg)')
        plt.legend()
        plt.show()

        # Guardar el modelo
        joblib.dump(model_hw_fit, 'exponential_smoothing_model.pkl')
        print("Modelo Exponential Smoothing guardado en 'exponential_smoothing_model.pkl'.")

        return pred_hw, mae_hw, rmse_hw, model_hw_fit
    except ValueError as e:
        print(f"Error al entrenar Exponential Smoothing: {e}")
        return None, None, None, None


def crear_secuencias(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class TQDMCallbackCustom(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.pbar = tqdm(total=total_epochs, desc='Entrenando LSTM', leave=False)

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()


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
    model_lstm.compile(optimizer='adam', loss='mae')
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

    # Visualización de la pérdida
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida Durante el Entrenamiento del LSTM')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MAE)')
    plt.legend()
    plt.show()

    # Guardar el modelo y el scaler
    model_lstm.save('lstm_model.h5')
    print("Modelo LSTM guardado en 'lstm_model.h5'.")
    joblib.dump(scaler, 'scaler_lstm.pkl')
    print("Scaler LSTM guardado en 'scaler_lstm.pkl'.")

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
    joblib.dump(model_xgb, 'xgboost_model.pkl')
    print("Modelo XGBoost guardado en 'xgboost_model.pkl'.")

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
    joblib.dump(model_lgbm, 'lightgbm_model.pkl')
    print("Modelo LightGBM guardado en 'lightgbm_model.pkl'.")

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
    joblib.dump(model_prophet, 'prophet_model.pkl')
    print("Modelo Prophet guardado en 'prophet_model.pkl'.")

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


def comparar_modelos(test, pred_hw, pred_lstm, pred_xgb, pred_lgbm, forecast_prophet, seq_length=14):
    # Alinear predicciones
    actual = test[target][seq_length:]
    pred_lstm_series = pd.Series(pred_lstm.flatten(), index=actual.index)

    pred_hw_aligned = pred_hw[-len(actual):].values if pred_hw is not None else [np.nan] * len(actual)
    forecast_aligned = forecast_prophet[-len(actual):].values
    pred_xgb_aligned = pred_xgb[-len(actual):] if len(pred_xgb) >= len(actual) else np.pad(pred_xgb, (len(actual) - len(pred_xgb), 0), 'constant')
    pred_lgbm_aligned = pred_lgbm[-len(actual):] if len(pred_lgbm) >= len(actual) else np.pad(pred_lgbm, (len(actual) - len(pred_lgbm), 0), 'constant')

    predictions_df = pd.DataFrame({
        'Actual': actual,
        'Exponential Smoothing': pred_hw_aligned,
        'LSTM': pred_lstm_series,
        'Prophet': forecast_aligned,
        'XGBoost': pred_xgb_aligned,
        'LightGBM': pred_lgbm_aligned
    })

    # Verificación de longitudes
    print("\nVerificación de longitudes en predictions_df:")
    for col in predictions_df.columns:
        print(f"{col}: {len(predictions_df[col])}")

    # Mostrar primeras filas
    print("\nPrimeras filas de las predicciones:")
    print(predictions_df.head())

    # Guardar predicciones
    predictions_df.to_csv('predicciones_modelos.csv', index=True)
    print("Las predicciones han sido guardadas en 'predicciones_modelos.csv'.")

    # Visualización de las predicciones
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

    # Resumen de métricas
    error_metrics = pd.DataFrame({
        'Modelo': ['Exponential Smoothing', 'LSTM', 'Prophet', 'XGBoost', 'LightGBM'],
        'MAE': [mae_hw if mae_hw is not None else np.nan, mae_lstm, mae_prophet, mae_xgb, mae_lgbm],
        'RMSE': [rmse_hw if rmse_hw is not None else np.nan, rmse_lstm, rmse_prophet, rmse_xgb, rmse_lgbm]
    })

    print("\nMétricas de Error de los Modelos:")
    print(error_metrics)

    # Guardar métricas
    error_metrics.to_csv('metricas_error_modelos.csv', index=False)
    print("Las métricas de error han sido guardadas en 'metricas_error_modelos.csv'.")

    return predictions_df, error_metrics


def validacion_cruzada(ts, target, n_splits=5, seq_length=14):
    print("\nImplementando Validación Cruzada Temporal...")
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_metrics = {
        'Fold': [],
        'Modelo': [],
        'MAE': [],
        'RMSE': []
    }

    for fold, (train_index, test_index) in enumerate(tscv.split(ts)):
        print(f"\nFold {fold + 1}")
        train_cv, test_cv = ts.iloc[train_index], ts.iloc[test_index]

        # Exponential Smoothing
        try:
            model_hw_cv = ExponentialSmoothing(train_cv[target], trend='add', seasonal='add', seasonal_periods=7)
            model_hw_fit_cv = model_hw_cv.fit()
            pred_hw_cv = model_hw_fit_cv.forecast(steps=len(test_cv))
            mae_cv_hw = mean_absolute_error(test_cv[target], pred_hw_cv)
            rmse_cv_hw = np.sqrt(mean_squared_error(test_cv[target], pred_hw_cv))
            print(f"CV Exponential Smoothing MAE: {mae_cv_hw:.2f}, CV RMSE: {rmse_cv_hw:.2f}")

            cv_metrics['Fold'].append(fold + 1)
            cv_metrics['Modelo'].append('Exponential Smoothing')
            cv_metrics['MAE'].append(mae_cv_hw)
            cv_metrics['RMSE'].append(rmse_cv_hw)
        except ValueError as e:
            print(f"Error en CV Exponential Smoothing: {e}")

        # XGBoost
        try:
            df_cv_xgb = crear_features_lag(train_cv, target, seq_length)
            X_train_cv_xgb = df_cv_xgb.drop(columns=[target])
            y_train_cv_xgb = df_cv_xgb[target]

            df_test_cv_xgb = crear_features_lag(test_cv, target, seq_length)
            X_test_cv_xgb = df_test_cv_xgb.drop(columns=[target])
            y_test_cv_xgb = df_test_cv_xgb[target]

            if not X_test_cv_xgb.empty:
                model_xgb_cv = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
                model_xgb_cv.fit(X_train_cv_xgb, y_train_cv_xgb)
                pred_xgb_cv = model_xgb_cv.predict(X_test_cv_xgb)
                mae_cv_xgb = mean_absolute_error(y_test_cv_xgb, pred_xgb_cv)
                rmse_cv_xgb = np.sqrt(mean_squared_error(y_test_cv_xgb, pred_xgb_cv))
                print(f"CV XGBoost MAE: {mae_cv_xgb:.2f}, CV RMSE: {rmse_cv_xgb:.2f}")

                # Guardar el modelo de CV
                joblib.dump(model_xgb_cv, f'xgboost_model_fold_{fold + 1}.pkl')
                print(f"Modelo XGBoost del Fold {fold + 1} guardado en 'xgboost_model_fold_{fold + 1}.pkl'.")

                cv_metrics['Fold'].append(fold + 1)
                cv_metrics['Modelo'].append('XGBoost')
                cv_metrics['MAE'].append(mae_cv_xgb)
                cv_metrics['RMSE'].append(rmse_cv_xgb)
            else:
                print("No hay suficientes datos para XGBoost en este fold.")
        except Exception as e:
            print(f"Error en CV XGBoost: {e}")

        # LightGBM
        try:
            if not X_test_cv_xgb.empty:
                model_lgbm_cv = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
                model_lgbm_cv.fit(X_train_cv_xgb, y_train_cv_xgb)
                pred_lgbm_cv = model_lgbm_cv.predict(X_test_cv_xgb)
                mae_cv_lgbm = mean_absolute_error(y_test_cv_xgb, pred_lgbm_cv)
                rmse_cv_lgbm = np.sqrt(mean_squared_error(y_test_cv_xgb, pred_lgbm_cv))
                print(f"CV LightGBM MAE: {mae_cv_lgbm:.2f}, CV RMSE: {rmse_cv_lgbm:.2f}")

                # Guardar el modelo de CV
                joblib.dump(model_lgbm_cv, f'lightgbm_model_fold_{fold + 1}.pkl')
                print(f"Modelo LightGBM del Fold {fold + 1} guardado en 'lightgbm_model_fold_{fold + 1}.pkl'.")

                cv_metrics['Fold'].append(fold + 1)
                cv_metrics['Modelo'].append('LightGBM')
                cv_metrics['MAE'].append(mae_cv_lgbm)
                cv_metrics['RMSE'].append(rmse_cv_lgbm)
            else:
                print("No hay suficientes datos para LightGBM en este fold.")
        except Exception as e:
            print(f"Error en CV LightGBM: {e}")

    cv_metrics_df = pd.DataFrame(cv_metrics)
    print("\nMétricas de Validación Cruzada Temporal:")
    print(cv_metrics_df)

    # Guardar métricas
    cv_metrics_df.to_csv('metricas_cv_modelos.csv', index=False)
    print("Las métricas de validación cruzada han sido guardadas en 'metricas_cv_modelos.csv'.")

    return cv_metrics_df


def importancia_features(model, features, modelo_nombre):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=False)
    print(f"\nImportancia de las Features para {modelo_nombre}:")
    print(importance_df)

    # Guardar en CSV
    filename = f'importancia_features_{modelo_nombre.lower()}.csv'
    importance_df.to_csv(filename, index=False)
    print(f"La importancia de las features para {modelo_nombre} ha sido guardada en '{filename}'.")

    # Visualización
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Feature', data=importance_df)
    plt.title(f'Importancia de las Features - {modelo_nombre}')
    plt.xlabel('Importancia')
    plt.ylabel('Features')
    plt.show()


def detectar_picos(ts_test, preds, umbral, modelos):
    print("\nDetección de picos en las predicciones:")
    picos_actual, _ = find_peaks(ts_test, height=umbral)
    print(f"Número de picos reales en la prueba: {len(picos_actual)}")

    comparacion_picos = {'Modelo': [], 'Picos Detectados': [], 'Picos Reales': [len(picos_actual)] * len(modelos)}

    for modelo, pred in preds.items():
        peaks, _ = find_peaks(pred, height=umbral)
        comparacion_picos['Modelo'].append(modelo)
        comparacion_picos['Picos Detectados'].append(len(peaks))
        print(f"Número de picos detectados por {modelo}: {len(peaks)}")

    comparacion_df = pd.DataFrame(comparacion_picos)
    print("\nComparación de Detección de Picos:")
    print(comparacion_df)

    comparacion_df.to_csv('comparacion_picos_modelos.csv', index=False)
    print("La comparación de detección de picos ha sido guardada en 'comparacion_picos_modelos.csv'.")

    return comparacion_df


def crear_features_lag(train_cv, target, lag=14):
    df_lag = pd.DataFrame(train_cv)
    for i in range(1, lag + 1):
        df_lag[f'lag_{i}'] = df_lag[target].shift(i)
    df_lag['Mes'] = df_lag['Mes']
    df_lag['DiaSemana'] = df_lag['DiaSemana']
    df_lag['EsFinDeSemana'] = df_lag['EsFinDeSemana']
    df_lag.dropna(inplace=True)
    return df_lag


# Flujo principal
if __name__ == "__main__":
    ruta_archivo_limpio = 'C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet'
    target = 'PESOBRUTOTOTAL'

    # 1. Cargar Datos
    df = cargar_datos(ruta_archivo_limpio)

    # 2. Preprocesar Datos
    ts_daily = preprocesar_datos(df, target)

    # 3. Dividir Datos
    train, test = dividir_datos(ts_daily, target)

    # 4. Entrenar Modelos
    pred_hw, mae_hw, rmse_hw, model_hw_fit = entrenar_exponential_smoothing(train, test, target)
    pred_lstm, mae_lstm, rmse_lstm, history, model_lstm, scaler_lstm = entrenar_lstm(train, test, target)
    pred_xgb, mae_xgb, rmse_xgb, model_xgb, X_train_xgb_columns = entrenar_xgboost(train, test, target)
    pred_lgbm, mae_lgbm, rmse_lgbm, model_lgbm = entrenar_lightgbm(train, test, target)
    forecast_prophet, mae_prophet, rmse_prophet, model_prophet = entrenar_prophet(train, test, target)

    # 5. Comparar Modelos
    predictions_df, error_metrics = comparar_modelos(test, pred_hw, pred_lstm, pred_xgb, pred_lgbm, forecast_prophet)

    # 6. Validación Cruzada Temporal
    cv_metrics_df = validacion_cruzada(ts_daily, target)

    # 7. Importancia de Features
    entrenando_xgb_lgbm = True
    if entrenando_xgb_lgbm:
        # Importancia para XGBoost
        importancia_features(model_xgb, X_train_xgb_columns, 'XGBoost')

        # Importancia para LightGBM
        importancia_features(model_lgbm, X_train_xgb_columns, 'LightGBM')

    # 8. Detección de Picos
    # Alinear predicciones
    actual_lstm = test[target][14:]  # seq_length=14
    pred_lstm_series = pd.Series(pred_lstm.flatten(), index=actual_lstm.index)

    pred_hw_aligned = pred_hw[-len(actual_lstm):].values if pred_hw is not None else [np.nan] * len(actual_lstm)
    forecast_prophet_aligned = forecast_prophet[-len(actual_lstm):].values
    pred_xgb_aligned = pred_xgb[-len(actual_lstm):] if len(pred_xgb) >= len(actual_lstm) else np.pad(pred_xgb, (len(actual_lstm) - len(pred_xgb), 0), 'constant')
    pred_lgbm_aligned = pred_lgbm[-len(actual_lstm):] if len(pred_lgbm) >= len(actual_lstm) else np.pad(pred_lgbm, (len(actual_lstm) - len(pred_lgbm), 0), 'constant')

    # Crear el DataFrame con las predicciones alineadas
    predictions_df = pd.DataFrame({
        'Actual': actual_lstm,
        'Exponential Smoothing': pred_hw_aligned,
        'LSTM': pred_lstm_series,
        'Prophet': forecast_prophet_aligned,
        'XGBoost': pred_xgb_aligned,
        'LightGBM': pred_lgbm_aligned
    })

    # Guardar predicciones
    predictions_df.to_csv('predicciones_modelos.csv', index=True)
    print("Las predicciones han sido guardadas en 'predicciones_modelos.csv'.")

    # Visualización de las predicciones
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

    # Resumen de métricas
    error_metrics = pd.DataFrame({
        'Modelo': ['Exponential Smoothing', 'LSTM', 'Prophet', 'XGBoost', 'LightGBM'],
        'MAE': [mae_hw if mae_hw is not None else np.nan, mae_lstm, mae_prophet, mae_xgb, mae_lgbm],
        'RMSE': [rmse_hw if rmse_hw is not None else np.nan, rmse_lstm, rmse_prophet, rmse_xgb, rmse_lgbm]
    })

    print("\nMétricas de Error de los Modelos:")
    print(error_metrics)

    # Guardar métricas
    error_metrics.to_csv('metricas_error_modelos.csv', index=False)
    print("Las métricas de error han sido guardadas en 'metricas_error_modelos.csv'.")

    # 9. Detección de Picos
    umbral_pico = ts_daily[target].median() * 1.5
    preds = {
        'Exponential Smoothing': pred_hw_aligned if pred_hw is not None else [],
        'LSTM': pred_lstm_series.values,
        'Prophet': forecast_prophet_aligned,
        'XGBoost': pred_xgb_aligned,
        'LightGBM': pred_lgbm_aligned
    }
    comparacion_picos = detectar_picos(actual_lstm, preds, umbral_pico, preds.keys())
