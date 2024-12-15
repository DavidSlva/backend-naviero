# interpreter/management/commands/predict_volumen.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
from datetime import timedelta, datetime
from django.core.management.base import BaseCommand
from django.conf import settings
from interpreter.models import VolumenPredicho
from collection_manager.models import Puerto  # Asegúrate de que este es el nombre correcto de tu modelo de Puerto
from django.db import transaction


class Command(BaseCommand):
    help = 'Realiza predicciones de volumen para los próximos 4 semanas a partir de la fecha actual para todos los puertos con data histórica suficiente.'

    def handle(self, *args, **options):
        # Obtener el directorio actual del script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 1) Definir rutas relativas al directorio del script
        model_path = os.path.join(script_dir, "model_multistep.h5")
        scalers_path = os.path.join(script_dir, "scalers")
        parquet_path = r"C://Users//David//Documents//Github//Proyecto Semestral Grafos y Algoritmos//backend//downloads//exportaciones_2024_combinado_clean_40perc.parquet"
        past_weeks = 4    # Debe coincidir con lo que entrenaste
        n_future  = 4     # Número de semanas a predecir

        self.stdout.write(self.style.NOTICE('Iniciando el proceso de predicción...'))

        # 2) Cargar modelo
        if not os.path.exists(model_path):
            self.stderr.write(self.style.ERROR(f"No se encontró el modelo en la ruta: {model_path}"))
            return
        model = load_model(model_path)
        self.stdout.write(self.style.SUCCESS(f"Modelo cargado desde: {model_path}"))

        # 3) Cargar scalers
        try:
            with open(os.path.join(scalers_path, "scaler_vol.pkl"), 'rb') as f:
                scaler_vol = pickle.load(f)
            with open(os.path.join(scalers_path, "scaler_port.pkl"), 'rb') as f:
                scaler_port = pickle.load(f)
            self.stdout.write(self.style.SUCCESS(f"Scalers cargados desde: {scalers_path}"))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error al cargar los scalers: {e}"))
            return

        # 4) Cargar DataFrame
        if not os.path.exists(parquet_path):
            self.stderr.write(self.style.ERROR(f"No se encontró el archivo parquet en la ruta: {parquet_path}"))
            return
        df_input = pd.read_parquet(parquet_path)
        df_input['FECHAACEPT'] = pd.to_datetime(df_input['FECHAACEPT'], errors='coerce')
        df_input = df_input.dropna(subset=['FECHAACEPT', 'PESOBRUTOTOTAL', 'PUERTOEMB'])
        self.stdout.write(self.style.SUCCESS(f"DataFrame cargado desde: {parquet_path}"))

        # 5) Obtener todos los puertos
        puertos = Puerto.objects.all()
        self.stdout.write(self.style.SUCCESS(f"Total de puertos a procesar: {puertos.count()}"))

        # 6) Determinar la semana actual (inicio de la semana)
        today = datetime.today().date()
        current_week_start = today - timedelta(days=today.weekday())  # Lunes de la semana actual
        self.stdout.write(f"Fecha actual: {today}")
        self.stdout.write(f"Inicio de la semana actual: {current_week_start}")

        # 7) Iterar sobre cada puerto
        for puerto in puertos:
            puertoemb = puerto.codigo  # Asumiendo que 'codigo' es el campo correcto
            self.stdout.write(f"\nProcesando puerto: {puerto.nombre} (Código: {puertoemb})")

            # Filtrar datos para el puerto actual
            df_port = df_input[df_input['PUERTOEMB'] == puertoemb].copy()
            df_port = df_port.dropna(subset=['FECHAACEPT', 'PESOBRUTOTOTAL'])

            # Crear columna SEMANA como datetime (inicio de la semana)
            df_port['SEMANA'] = df_port['FECHAACEPT'].dt.to_period('W').apply(lambda r: r.start_time.date())
            df_port_weekly = (
                df_port.groupby(['PUERTOEMB', 'SEMANA'], as_index=False)['PESOBRUTOTOTAL']
                .sum()
                .sort_values('SEMANA')
            )

            # Identificar la última semana en el dataset para ese puerto
            ultima_semana = df_port_weekly['SEMANA'].max()
            if pd.isna(ultima_semana):
                self.stderr.write(f"No se encontró la última semana para el puerto {puerto.nombre}. Saltando.")
                continue
            self.stdout.write(f"Última semana en dataset para el puerto {puerto.nombre}: {ultima_semana}")

            # Definir la semana objetivo para la predicción
            # Queremos que las predicciones lleguen hasta la fecha actual y 4 semanas más
            # Por lo tanto, la semana objetivo es la semana actual
            # Y las predicciones son las próximas 4 semanas
            if ultima_semana < current_week_start:
                # Si la última semana en datos es antes de la semana actual, establecemos la semana objetivo como la semana actual
                fecha_semana_objetivo = current_week_start
            else:
                # Si la última semana en datos es la semana actual o más reciente, establecemos la semana objetivo como la siguiente semana
                fecha_semana_objetivo = ultima_semana + timedelta(weeks=1)

            self.stdout.write(f"Semana objetivo para predicción: {fecha_semana_objetivo}")

            # Calcular las semanas a predecir para cubrir hasta 4 semanas más
            # Si la semana objetivo ya cubre la actualidad, simplemente predecir 4 semanas más
            # Si no, predecir hasta la semana actual y luego 4 semanas más
            # Para simplificar, predecimos siempre 4 semanas a partir de la semana objetivo

            # Predecir las próximas n_future semanas
            predicciones = self.predict_n_future_weeks(
                model=model,
                scaler_vol=scaler_vol,
                scaler_port=scaler_port,
                past_weeks=past_weeks,
                n_future=n_future,
                df_port_weekly=df_port_weekly,
                puertoemb=puertoemb,
                fecha_semana_objetivo=fecha_semana_objetivo
            )

            if predicciones is None:
                self.stderr.write(f"No se pudieron realizar predicciones para el puerto {puerto.nombre}.")
                continue

            # Guardar las predicciones en el nuevo modelo
            self.save_predicciones(puerto, fecha_semana_objetivo, predicciones, past_weeks)

        self.stdout.write(self.style.SUCCESS('\nProceso de predicción completado exitosamente.'))

    def predict_n_future_weeks(
        self,
        model,
        scaler_vol,
        scaler_port,
        past_weeks,
        n_future,
        df_port_weekly,
        puertoemb,
        fecha_semana_objetivo
    ):
        """
        Dado un puerto y una fecha "semana objetivo", retorna la predicción de
        n_future semanas.
        """
        try:
            # Filtrar semanas anteriores a la semana objetivo
            df_port_weekly_prev = df_port_weekly[df_port_weekly['SEMANA'] < fecha_semana_objetivo]

            if len(df_port_weekly_prev) < past_weeks:
                self.stderr.write(f"No hay suficientes datos para el puerto {puertoemb} antes de {fecha_semana_objetivo}.")
                return None

            # Tomar las últimas past_weeks filas
            df_recent = df_port_weekly_prev.tail(past_weeks).copy()

            # Escalar los datos
            df_recent['PESOBRUTOTOTAL_SCALED'] = scaler_vol.transform(df_recent[['PESOBRUTOTOTAL']])
            df_recent['PUERTOEMB_SCALED'] = scaler_port.transform(df_recent[['PUERTOEMB']])

            # Preparar las secuencias para el modelo
            vol_scaled = df_recent['PESOBRUTOTOTAL_SCALED'].values
            port_scaled = df_recent['PUERTOEMB_SCALED'].values
            seq_features = np.column_stack([vol_scaled, port_scaled])  # (past_weeks, 2)
            X_input = seq_features.reshape((1, past_weeks, 2))

            # Realizar la predicción multi-step
            pred_scaled = model.predict(X_input)  # (1, n_future)
            pred_scaled = pred_scaled.reshape(-1, 1)  # (n_future, 1)
            pred = scaler_vol.inverse_transform(pred_scaled).flatten()  # (n_future,)

            self.stdout.write(f"Predicción para {n_future} semanas a partir de {fecha_semana_objetivo}: {pred}")

            return pred
        except Exception as e:
            self.stderr.write(f"Error al predecir para el puerto {puertoemb}: {e}")
            return None

    @transaction.atomic
    def save_predicciones(self, puerto, fecha_semana_objetivo, predicciones, past_weeks):
        """
        Guarda las predicciones en el modelo VolumenPredicho.
        """
        try:
            pred_objs = []
            for i, valor in enumerate(predicciones, start=1):
                semana_predicha = fecha_semana_objetivo + timedelta(weeks=i-1)
                pred_objs.append(VolumenPredicho(
                    puerto=puerto,
                    semana=semana_predicha,
                    volumen_predicho=valor,
                    past_weeks=past_weeks  # Guardar cuántas semanas pasadas se usaron
                ))
            VolumenPredicho.objects.bulk_create(pred_objs, ignore_conflicts=True)
            self.stdout.write(self.style.SUCCESS(f"Predicciones guardadas para el puerto {puerto.nombre}."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error al guardar predicciones para el puerto {puerto.nombre}: {e}"))
