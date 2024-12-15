import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm  # Para mostrar barras de progreso
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import timedelta
import random

# Configuración de los logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def train_lstm_all_ports_multistep(
    data_path_parquet=None,
    data_path_csv=None,
    past_weeks=4,            # Número de semanas en la ventana de entrada
    n_future=4,              # Número de semanas a predecir en una sola pasada
    epochs=10,
    batch_size=32,
    save_model_path="./model_multistep.h5",    # Ruta para guardar el modelo
    save_scalers_path="./scalers"             # Carpeta para guardar los scalers
):
    """
    Entrena un modelo LSTM "multi-step" para predecir n_future semanas a partir de
    las últimas past_weeks semanas de historia, utilizando TODOS los PUERTOEMB como feature adicional.
    Luego, guarda el modelo y los scalers en disco.

    Devuelve:
      - El modelo entrenado (Keras)
      - scaler_vol (escala/desescala PESOBRUTOTOTAL)
      - scaler_port (escala/desescala PUERTOEMB)
      - past_weeks  (para reconstruir secuencias)
      - n_future    (semanas que pronosticamos)
    """

    logging.info("=== Inicio del entrenamiento del modelo LSTM multi-step ===")

    # --------------------------------------------------------
    # 1. Carga de datos
    # --------------------------------------------------------
    logging.info("Cargando datos...")
    if data_path_parquet:
        if not os.path.exists(data_path_parquet):
            logging.error(f"No se encontró el archivo Parquet en la ruta: {data_path_parquet}")
            raise FileNotFoundError(f"No se encontró el archivo Parquet en la ruta: {data_path_parquet}")
        df = pd.read_parquet(data_path_parquet)
        logging.info(f"Datos cargados desde Parquet: {data_path_parquet}")
    elif data_path_csv:
        if not os.path.exists(data_path_csv):
            logging.error(f"No se encontró el archivo CSV en la ruta: {data_path_csv}")
            raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {data_path_csv}")
        df = pd.read_csv(data_path_csv, parse_dates=['FECHAACEPT'])
        logging.info(f"Datos cargados desde CSV: {data_path_csv}")
    else:
        logging.error("Debes proporcionar al menos una ruta de archivo (Parquet o CSV).")
        raise ValueError("Proporciona al menos una ruta de archivo (Parquet o CSV).")

    # Aseguramos que FECHAACEPT sea datetime
    if not pd.api.types.is_datetime64_any_dtype(df['FECHAACEPT']):
        df['FECHAACEPT'] = pd.to_datetime(df['FECHAACEPT'], errors='coerce')
        logging.info("Columna 'FECHAACEPT' convertida a datetime.")

    # --------------------------------------------------------
    # 2. Manejo de Valores Nulos
    # --------------------------------------------------------
    logging.info("Analizando valores nulos antes de la imputación...")

    # Contar valores nulos antes
    null_counts_before = df[['FECHAACEPT', 'PESOBRUTOTOTAL', 'PUERTOEMB']].isnull().sum()
    logging.info("Valores nulos por columna antes de la imputación:")
    for col, count in null_counts_before.items():
        logging.info(f"  {col}: {count} ({(count/len(df))*100:.2f}%)")

    # Imputar valores nulos en 'PESOBRUTOTOTAL' con el promedio por 'PUERTOEMB'
    logging.info("Imputando valores nulos en 'PESOBRUTOTOTAL' con el promedio por 'PUERTOEMB'...")
    df['PESOBRUTOTOTAL'] = df.groupby('PUERTOEMB')['PESOBRUTOTOTAL'].transform(lambda x: x.fillna(x.mean()))
    logging.info("Imputación completada.")

    # Verificar valores nulos después de la imputación
    null_counts_after = df[['FECHAACEPT', 'PESOBRUTOTOTAL', 'PUERTOEMB']].isnull().sum()
    logging.info("Valores nulos por columna después de la imputación:")
    for col, count in null_counts_after.items():
        logging.info(f"  {col}: {count} ({(count/len(df))*100:.2f}%)")

    # Ahora, eliminar filas con valores nulos en 'FECHAACEPT' y 'PUERTOEMB'
    logging.info("Eliminando filas con valores nulos en 'FECHAACEPT' y 'PUERTOEMB'...")
    initial_shape = df.shape
    df = df.dropna(subset=['FECHAACEPT', 'PUERTOEMB'])
    logging.info(f"Filas eliminadas por nulos en 'FECHAACEPT' y 'PUERTOEMB': {initial_shape[0] - df.shape[0]}")

    # --------------------------------------------------------
    # 3. Agrupar datos por (PUERTOEMB, SEMANA)
    # --------------------------------------------------------
    logging.info("Agrupando datos por PUERTOEMB y SEMANA...")
    df['SEMANA'] = df['FECHAACEPT'].dt.to_period('W').apply(lambda r: r.start_time)

    # Sumar PESOBRUTOTOTAL por (PUERTOEMB, SEMANA)
    df_weekly = df.groupby(['PUERTOEMB', 'SEMANA'], as_index=False)['PESOBRUTOTOTAL'].sum()

    # Logs adicionales: contar semanas por puerto
    semanas_por_puerto = df_weekly.groupby('PUERTOEMB').size()
    logging.info("Cantidad de semanas por PUERTOEMB:")
    for puerto, count in semanas_por_puerto.items():
        logging.info(f"  PUERTOEMB {puerto}: {count} semanas")

    # --------------------------------------------------------
    # 4. Preparar escaladores
    # --------------------------------------------------------
    logging.info("Preparando escaladores...")
    scaler_vol = MinMaxScaler(feature_range=(0, 1))
    scaler_port = MinMaxScaler(feature_range=(0, 1))

    # Ajustar y transformar PESOBRUTOTOTAL
    df_weekly['PESOBRUTOTOTAL_SCALED'] = scaler_vol.fit_transform(df_weekly[['PESOBRUTOTOTAL']])
    logging.info("Escalador para 'PESOBRUTOTOTAL' ajustado y transformado.")

    # Ajustar y transformar PUERTOEMB
    df_weekly['PUERTOEMB_SCALED'] = scaler_port.fit_transform(df_weekly[['PUERTOEMB']])
    logging.info("Escalador para 'PUERTOEMB' ajustado y transformado.")

    # --------------------------------------------------------
    # 5. Construir secuencias multi-step
    # --------------------------------------------------------
    logging.info("Construyendo secuencias multi-step...")
    df_weekly = df_weekly.sort_values(by=['PUERTOEMB', 'SEMANA'])

    X_list, y_list = [], []
    unique_ports = df_weekly['PUERTOEMB'].unique()
    logging.info(f"Cantidad de PUERTOEMB únicos: {len(unique_ports)}")

    for p in tqdm(unique_ports, desc="Creando secuencias multi-step"):
        df_port = df_weekly[df_weekly['PUERTOEMB'] == p].copy()
        vol_scaled = df_port['PESOBRUTOTOTAL_SCALED'].values
        port_scaled = df_port['PUERTOEMB_SCALED'].values

        for i in range(len(df_port) - past_weeks - n_future + 1):
            seq_vol  = vol_scaled[i : i + past_weeks]
            seq_port = port_scaled[i : i + past_weeks]
            seq_features = np.column_stack([seq_vol, seq_port])  # (past_weeks, 2)
            X_list.append(seq_features)

            # Las próximas n_future semanas:
            future_vol = vol_scaled[i + past_weeks : i + past_weeks + n_future]
            y_list.append(future_vol)  # (n_future,)

    X = np.array(X_list)  # (#samples, past_weeks, 2)
    y = np.array(y_list)  # (#samples, n_future)
    logging.info(f"Total de muestras creadas: {X.shape[0]}")

    # --------------------------------------------------------
    # 6. División train / test
    # --------------------------------------------------------
    logging.info("Dividiendo datos en train y test...")
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"Train size: {X_train.shape[0]} muestras")
    logging.info(f"Test size: {X_test.shape[0]} muestras")

    # --------------------------------------------------------
    # 7. Definir modelo LSTM multi-step
    # --------------------------------------------------------
    logging.info("Definiendo el modelo LSTM multi-step...")
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=False, input_shape=(past_weeks, 2)))
    model.add(Dropout(0.2))
    model.add(Dense(n_future))  # Salida multi-step

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    logging.info("Modelo compilado:")
    model.summary(print_fn=lambda x: logging.info(x))

    # --------------------------------------------------------
    # 8. Entrenar
    # --------------------------------------------------------
    logging.info("Iniciando entrenamiento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    logging.info("Entrenamiento completado.")

    # --------------------------------------------------------
    # 9. Evaluar
    # --------------------------------------------------------
    loss = model.evaluate(X_test, y_test, verbose=0)
    logging.info(f"Test Loss (MSE): {loss:.6f}")

    # --------------------------------------------------------
    # 10. Guardar modelo y scalers
    # --------------------------------------------------------
    if save_model_path:
        model.save(save_model_path)
        logging.info(f"Modelo guardado en: {save_model_path}")

    if save_scalers_path:
        os.makedirs(save_scalers_path, exist_ok=True)
        with open(os.path.join(save_scalers_path, "scaler_vol.pkl"), 'wb') as f:
            pickle.dump(scaler_vol, f)
        with open(os.path.join(save_scalers_path, "scaler_port.pkl"), 'wb') as f:
            pickle.dump(scaler_port, f)
        logging.info(f"Scalers guardados en la carpeta: {save_scalers_path}")

    # --------------------------------------------------------
    # 11. Visualizar Predicciones vs Reales en el Conjunto de Test
    # --------------------------------------------------------
    logging.info("Generando predicciones para el conjunto de test...")
    y_pred_scaled = model.predict(X_test)  # (test_size, n_future)
    y_pred = scaler_vol.inverse_transform(y_pred_scaled)  # (test_size, n_future)
    y_test_inv = scaler_vol.inverse_transform(y_test)    # (test_size, n_future)

    # Aplanar las matrices para comparar todas las predicciones y reales
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test_inv.flatten()

    # Plotting Predicciones vs Reales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_flat, y=y_pred_flat, alpha=0.5)
    plt.plot([y_test_flat.min(), y_test_flat.max()], [y_test_flat.min(), y_test_flat.max()], 'r--')  # Línea y=x
    plt.title("Predicciones vs Reales para Todo el Conjunto de Test")
    plt.xlabel("Real PESOBRUTOTOTAL")
    plt.ylabel("Predicho PESOBRUTOTOTAL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predicciones_vs_reales_general.png")
    plt.show()
    logging.info("Gráfico general de predicciones vs reales generado y guardado como 'predicciones_vs_reales_general.png'.")

    # También puedes visualizar el historial de entrenamiento
    # Plot loss over epochs
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, epochs + 1), y=history.history['loss'], label='Train Loss')
    sns.lineplot(x=range(1, epochs + 1), y=history.history['val_loss'], label='Validation Loss')
    plt.title("Historial de Entrenamiento")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("historial_entrenamiento.png")
    plt.show()
    logging.info("Gráfico del historial de entrenamiento generado y guardado como 'historial_entrenamiento.png'.")

    logging.info("=== Fin del entrenamiento ===")

    return model, scaler_vol, scaler_port, past_weeks, n_future

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
    df_port = df_port.dropna(subset=['FECHAACEPT'])  # Ya no es necesario eliminar 'PESOBRUTOTOTAL' NAs

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
        logging.warning(f"No hay suficiente historial (past_weeks={past_weeks}) para el puerto {puertoemb} antes de {semana_obj.date()}.")
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

    logging.info(f"Predicción multi-step para {n_future} semanas a partir de {semana_obj.date()}, puerto={puertoemb}")
    for i, val in enumerate(pred, start=1):
        logging.info(f"  Semana +{i}: {val:.2f}")

    return pred

def load_scalers(scalers_path):
    """
    Carga los scalers desde una carpeta.
    """
    with open(os.path.join(scalers_path, "scaler_vol.pkl"), 'rb') as f:
        scaler_vol = pickle.load(f)
    with open(os.path.join(scalers_path, "scaler_port.pkl"), 'rb') as f:
        scaler_port = pickle.load(f)
    return scaler_vol, scaler_port

if __name__ == "__main__":
    # ----------------------------------------------------------
    # Entrenamiento y Guardado del Modelo
    # ----------------------------------------------------------
    parquet_path = r"C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet"

    # Entrenar y guardar el modelo y los scalers
    model, scaler_vol, scaler_port, past_weeks, n_future = train_lstm_all_ports_multistep(
        data_path_parquet=parquet_path,
        past_weeks=4,
        n_future=4,      # 4 semanas de horizonte de pronóstico
        epochs=10,
        batch_size=32,
        save_model_path="./model_multistep.h5",       # Guarda el modelo en la ruta especificada
        save_scalers_path="./scalers"                 # Carpeta para pkl
    )

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
    logging.info(f"Modelo cargado desde: {MODEL_PATH}")

    # 2) Cargar scalers
    if not os.path.exists(SCALERS_PATH):
        raise FileNotFoundError(f"No se encontró la carpeta de scalers en la ruta: {SCALERS_PATH}")
    scaler_vol, scaler_port = load_scalers(SCALERS_PATH)
    logging.info(f"Scalers cargados desde: {SCALERS_PATH}")

    # 3) Cargar DataFrame
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el archivo parquet en la ruta: {parquet_path}")
    df_input = pd.read_parquet(parquet_path)
    df_input['FECHAACEPT'] = pd.to_datetime(df_input['FECHAACEPT'], errors='coerce')
    # No eliminar 'PESOBRUTOTOTAL' NAs aquí, ya fueron manejados en el entrenamiento
    df_input = df_input.dropna(subset=['FECHAACEPT', 'PUERTOEMB'])
    logging.info(f"DataFrame cargado desde: {parquet_path}")

    # Parámetros
    past_weeks = 4    # Debe coincidir con lo que entrenaste
    n_future  = 4     # Número de semanas a predecir

    # 4) Elegir un puerto aleatorio (o fijo) y hallar su última semana
    puertos_unicos = df_input['PUERTOEMB'].unique()
    puertoemb = random.choice(puertos_unicos)  # Elige puerto aleatorio
    # puertoemb = 12  # Si prefieres un puerto fijo, descomenta esta línea y comenta la anterior
    logging.info(f"Puerto seleccionado: {puertoemb}")

    # Agrupar para saber la última semana disponible de ese puerto
    df_port = df_input[df_input['PUERTOEMB'] == puertoemb].copy()
    df_port['SEMANA'] = df_port['FECHAACEPT'].dt.to_period('W').apply(lambda r: r.start_time)
    df_port_weekly = df_port.groupby(['PUERTOEMB','SEMANA'], as_index=False)['PESOBRUTOTOTAL'].sum().sort_values('SEMANA')

    # Logs adicionales: cantidad de semanas para el puerto seleccionado
    total_semanas = df_port_weekly.shape[0]
    logging.info(f"Cantidad total de semanas para PUERTOEMB {puertoemb}: {total_semanas}")

    # Identificar la última semana en el dataset para ese puerto
    ultima_semana = df_port_weekly['SEMANA'].max()
    logging.info(f"Última semana en dataset para el puerto {puertoemb}: {ultima_semana.date()}")

    # 5) Usar esa última semana + 7 días como "semana objetivo"
    #    Para predecir las 4 semanas siguientes
    fecha_semana_objetivo = ultima_semana + timedelta(days=7)
    logging.info(f"Semana objetivo para predicción: {fecha_semana_objetivo.date()}")

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

    # 'preds' contendrá un array con las predicciones de 4 semanas a futuro.
