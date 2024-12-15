# predict.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import random
from datetime import timedelta
import os

def load_scalers(scalers_path):
    """
    Carga los scalers desde una carpeta.
    """
    with open(os.path.join(scalers_path, "scaler_vol.pkl"), 'rb') as f:
        scaler_vol = pickle.load(f)
    with open(os.path.join(scalers_path, "scaler_port.pkl"), 'rb') as f:
        scaler_port = pickle.load(f)
    return scaler_vol, scaler_port

def predict_n_future_weeks(
    model,
    scaler_vol,
    scaler_port,
    past_weeks,
    n_future,
    df_input,
    puertoemb,
    fecha_semana_objetivo
):
    """
    Dado un puerto y una fecha "semana objetivo", retorna la predicción de
    n_future semanas.
    """
    fecha_semana_objetivo = pd.to_datetime(fecha_semana_objetivo)
    semana_obj = fecha_semana_objetivo  # Timestamp

    df_port = df_input[df_input['PUERTOEMB'] == puertoemb].copy()
    df_port = df_port.dropna(subset=['FECHAACEPT', 'PESOBRUTOTOTAL'])

    # Crear columna SEMANA como datetime (inicio de la semana)
    df_port['SEMANA'] = df_port['FECHAACEPT'].dt.to_period('W').apply(lambda r: r.start_time)
    df_port_weekly = (
        df_port.groupby(['PUERTOEMB','SEMANA'], as_index=False)['PESOBRUTOTOTAL']
        .sum()
        .sort_values('SEMANA')
    )

    # Filtrar semanas anteriores a semana_obj
    df_port_weekly = df_port_weekly[df_port_weekly['SEMANA'] < semana_obj]

    if len(df_port_weekly) < past_weeks:
        print(f"No hay suficiente historial (past_weeks={past_weeks}) para el puerto {puertoemb} antes de {semana_obj.date()}.")
        return None

    # Escalar
    df_port_weekly['PESOBRUTOTOTAL_SCALED'] = scaler_vol.transform(df_port_weekly[['PESOBRUTOTOTAL']])
    df_port_weekly['PUERTOEMB_SCALED'] = scaler_port.transform(df_port_weekly[['PUERTOEMB']])

    # Tomar las últimas past_weeks filas
    df_port_weekly_recent = df_port_weekly.tail(past_weeks).copy()

    vol_scaled = df_port_weekly_recent['PESOBRUTOTOTAL_SCALED'].values
    port_scaled = df_port_weekly_recent['PUERTOEMB_SCALED'].values
    seq_features = np.column_stack([vol_scaled, port_scaled])  # (past_weeks, 2)
    X_input = seq_features.reshape((1, past_weeks, 2))

    # Predicción multi-step
    pred_scaled = model.predict(X_input)  # (1, n_future)
    pred_scaled = pred_scaled.reshape(-1, 1)  # (n_future, 1)
    pred = scaler_vol.inverse_transform(pred_scaled).flatten()  # (n_future,)

    print(f"Predicción multi-step para {n_future} semanas a partir de {semana_obj.date()}, puerto={puertoemb}")
    for i, val in enumerate(pred, start=1):
        print(f"  Semana +{i}: {val:.2f}")

    return pred

if __name__ == "__main__":
    # ----------------------------------------------------------
    # Predicción con el Modelo Guardado
    # ----------------------------------------------------------
    # Rutas a modelo y scalers
    MODEL_PATH = "./model_multistep.h5"
    SCALERS_PATH = "./scalers"    # Carpeta donde guardamos los pkl

    # 1) Cargar modelo
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en la ruta: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"Modelo cargado desde: {MODEL_PATH}")

    # 2) Cargar scalers
    if not os.path.exists(SCALERS_PATH):
        raise FileNotFoundError(f"No se encontró la carpeta de scalers en la ruta: {SCALERS_PATH}")
    scaler_vol, scaler_port = load_scalers(SCALERS_PATH)
    print(f"Scalers cargados desde: {SCALERS_PATH}")

    # 3) Cargar DataFrame
    parquet_path = r"C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el archivo parquet en la ruta: {parquet_path}")
    df_input = pd.read_parquet(parquet_path)
    df_input['FECHAACEPT'] = pd.to_datetime(df_input['FECHAACEPT'], errors='coerce')
    df_input = df_input.dropna(subset=['FECHAACEPT', 'PESOBRUTOTOTAL','PUERTOEMB'])
    print(f"DataFrame cargado desde: {parquet_path}")

    # Parámetros
    past_weeks = 4    # Debe coincidir con lo que entrenaste
    n_future  = 4     # Número de semanas a predecir

    # 4) Elegir un puerto aleatorio (o fijo) y hallar su última semana
    puertos_unicos = df_input['PUERTOEMB'].unique()
    puertoemb = random.choice(puertos_unicos)  # Elige puerto aleatorio
    # puertoemb = 12  # Si prefieres un puerto fijo, descomenta esta línea y comenta la anterior
    print(f"Puerto seleccionado: {puertoemb}")

    # Agrupar para saber la última semana disponible de ese puerto
    df_port = df_input[df_input['PUERTOEMB'] == puertoemb].copy()
    df_port['SEMANA'] = df_port['FECHAACEPT'].dt.to_period('W').apply(lambda r: r.start_time)
    df_port_weekly = df_port.groupby(['PUERTOEMB','SEMANA'], as_index=False)['PESOBRUTOTOTAL'].sum().sort_values('SEMANA')

    # Identificar la última semana en el dataset para ese puerto
    ultima_semana = df_port_weekly['SEMANA'].max()
    print(f"Última semana en dataset para el puerto {puertoemb}: {ultima_semana.date()}")

    # 5) Usar esa última semana + 7 días como "semana objetivo"
    #    Para predecir las 4 semanas siguientes
    fecha_semana_objetivo = ultima_semana + timedelta(days=7)
    print(f"Semana objetivo para predicción: {fecha_semana_objetivo.date()}")

    # 6) Llamar la función predict_n_future_weeks
    preds = predict_n_future_weeks(
        model,
        scaler_vol,
        scaler_port,
        past_weeks,
        n_future,
        df_input,
        puertoemb=puertoemb,
        fecha_semana_objetivo=fecha_semana_objetivo
    )
    print(preds)
    # 'preds' contendrá un array con las predicciones de 4 semanas a futuro.
