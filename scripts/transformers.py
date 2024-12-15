import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
from tqdm import tqdm
from time import sleep
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks

# Para Transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
# En versiones nuevas de PyTorch Lightning, usamos el callback de progreso
from pytorch_lightning.callbacks import TQDMProgressBar

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
df['Mes'] = df.index.month
df['DiaSemana'] = df.index.dayofweek
df['EsFinDeSemana'] = df['DiaSemana'].apply(lambda x: 1 if x >= 5 else 0)

# Resamplear la serie temporal a frecuencia diaria
ts_daily = df[[target, 'Mes', 'DiaSemana', 'EsFinDeSemana']].resample('D').sum()
ts_daily = ts_daily.fillna(ts_daily.median())

# Visualizar la serie temporal diaria
plt.figure(figsize=(15, 6))
plt.plot(ts_daily[target], label='Volumen Total (Kg) - Diario')
plt.title('Volumen Total a lo Largo del Tiempo (Diario)')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

print(f"Tamaño total de la serie diaria: {len(ts_daily)}")

# 3. División de Datos en Entrenamiento y Prueba
train_size = int(len(ts_daily) * 0.8)
train, test = ts_daily[:train_size], ts_daily[train_size:]
print(f"Tamaño del entrenamiento: {len(train)}")
print(f"Tamaño de la prueba: {len(test)}")

# ============================
# 4. Modelos de Transformers
# ============================

class TimeSeriesDataset(Dataset):
    def __init__(self, series, window_size):
        self.series = series
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.series_scaled = self.scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.series_scaled) - self.window_size

    def __getitem__(self, idx):
        x = self.series_scaled[idx:idx+self.window_size]
        y = self.series_scaled[idx+self.window_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TransformerTimeSeries(pl.LightningModule):
    def __init__(self, input_dim=1, model_dim=64, num_heads=4, num_encoder_layers=3, dim_feedforward=128, dropout=0.1, window_size=14):
        super(TransformerTimeSeries, self).__init__()
        self.save_hyperparameters()
        self.model_dim = model_dim
        self.window_size = window_size

        self.input_fc = nn.Linear(input_dim, model_dim)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.output_fc = nn.Linear(model_dim, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, src, tgt):
        src = self.input_fc(src)  # [batch, window, model_dim]
        tgt = self.input_fc(tgt)  # [batch, window, model_dim]
        output = self.transformer(src, tgt)
        output = self.output_fc(output[:, -1, :])
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x shape: [batch_size, window_size]
        # y shape: [batch_size]

        # Convierte x e y a 3D
        x_3d = x.unsqueeze(-1)                # [batch_size, window_size, 1]
        y_3d = y.unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]

        # Concatenar en la dimensión 1
        tgt_input = torch.cat([x_3d, y_3d], dim=1)  # [batch_size, window_size + 1, 1]
        tgt_input = tgt_input[:, :-1, :]           # [batch_size, window_size, 1] (teacher forcing)

        # Forward del Transformer
        y_pred = self.forward(x_3d, tgt_input)     # [batch_size, 1]

        # Calcular pérdida
        loss = self.loss_fn(y_pred.squeeze(), y)

        # Registrar pérdida
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_3d = x.unsqueeze(-1)
        y_3d = y.unsqueeze(-1).unsqueeze(-1)

        tgt_input = torch.cat([x_3d, y_3d], dim=1)
        tgt_input = tgt_input[:, :-1, :]

        y_pred = self.forward(x_3d, tgt_input)
        loss = self.loss_fn(y_pred.squeeze(), y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

WINDOW_SIZE = 14
train_dataset = TimeSeriesDataset(train[[target]], WINDOW_SIZE)
test_dataset = TimeSeriesDataset(test[[target]], WINDOW_SIZE)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model_transformer = TransformerTimeSeries(window_size=WINDOW_SIZE)

# Callback de barra de progreso
tqdm_bar = TQDMProgressBar(refresh_rate=20)

# PyTorch Lightning Trainer (nuevo API: accelerator/devices)
trainer = pl.Trainer(
    max_epochs=50,
    callbacks=[tqdm_bar],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

print("Entrenando el modelo Transformer...")
trainer.fit(model_transformer, train_loader, val_loader)

# Predicciones
model_transformer.eval()
predictions = []
actuals = []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(model_transformer.device)
        y = y.to(model_transformer.device)
        tgt_input = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1).unsqueeze(-1)], dim=1)[:, :-1, :]
        y_pred = model_transformer(x.unsqueeze(-1), tgt_input)
        predictions.extend(y_pred.squeeze().cpu().numpy())
        actuals.extend(y.cpu().numpy())

scaler = train_dataset.scaler
predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals_inv = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

mae_transformer = mean_absolute_error(actuals_inv, predictions_inv)
rmse_transformer = np.sqrt(mean_squared_error(actuals_inv, predictions_inv))
print(f"Transformer MAE: {mae_transformer:.2f}")
print(f"Transformer RMSE: {rmse_transformer:.2f}")

plt.figure(figsize=(15, 6))
plt.plot(test.index[WINDOW_SIZE:], actuals_inv, label='Actual', color='black')
plt.plot(test.index[WINDOW_SIZE:], predictions_inv, label='Predicción Transformer', color='magenta')
plt.title('Modelo Transformer para Series Temporales')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# ============================
# 5. Modelos Basados en Detección de Anomalías
# ============================

print("Entrenando el modelo Isolation Forest para detección de anomalías...")
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(ts_daily[[target]])
anomalies_iso = iso_forest.predict(ts_daily[[target]])
ts_daily['Anomaly_ISO'] = anomalies_iso

print("Entrenando el Autoencoder para detección de anomalías...")

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 4),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder_scaler = StandardScaler()
data_scaled = autoencoder_scaler.fit_transform(ts_daily[[target]])
data_tensor = torch.FloatTensor(data_scaled)

autoencoder = Autoencoder().to(model_transformer.device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 100
batch_size = 32
dataset_ae = torch.utils.data.TensorDataset(data_tensor)
dataloader_ae = DataLoader(dataset_ae, batch_size=batch_size, shuffle=True, num_workers=4)

print("Entrenando el Autoencoder...")
autoencoder.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data in dataloader_ae:
        img = data[0].to(model_transformer.device)
        output = autoencoder(img)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * img.size(0)
    epoch_loss /= len(dataloader_ae.dataset)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

autoencoder.eval()
with torch.no_grad():
    reconstructed = autoencoder(data_tensor.to(model_transformer.device))
    loss_values = torch.mean((reconstructed - data_tensor.to(model_transformer.device))**2, dim=1).cpu().numpy()

threshold = np.percentile(loss_values, 95)
print(f"Umbral para anomalías (Autoencoder): {threshold:.4f}")

anomalies_autoencoder = loss_values > threshold
ts_daily['Anomaly_AE'] = 0
ts_daily.loc[anomalies_autoencoder, 'Anomaly_AE'] = -1  # -1 indica anomalía

# ============================
# 6. Visualización de Anomalías
# ============================

plt.figure(figsize=(15, 6))
plt.plot(ts_daily.index, ts_daily[target], label='Volumen Total (Kg)', color='blue')
# Anomalías Isolation Forest
plt.scatter(ts_daily.index[ts_daily['Anomaly_ISO'] == -1],
            ts_daily[target][ts_daily['Anomaly_ISO'] == -1],
            color='red', label='Anomalías Isolation Forest')
# Anomalías Autoencoder
plt.scatter(ts_daily.index[ts_daily['Anomaly_AE'] == -1],
            ts_daily[target][ts_daily['Anomaly_AE'] == -1],
            color='magenta', label='Anomalías Autoencoder')
plt.title('Detección de Anomalías en la Serie Temporal')
plt.xlabel('Fecha')
plt.ylabel('Peso Bruto Total (Kg)')
plt.legend()
plt.show()

# ============================
# 7. Resumen de Anomalías
# ============================

anomalies_count = pd.DataFrame({
    'Modelo': ['Isolation Forest', 'Autoencoder'],
    'Anomalías Detectadas': [
        ts_daily['Anomaly_ISO'].value_counts().get(-1, 0),
        ts_daily['Anomaly_AE'].value_counts().get(-1, 0)
    ]
})

print("\nResumen de Detección de Anomalías:")
print(anomalies_count)

ts_daily.to_csv('deteccion_anomalias.csv')
print("\nLas anomalías detectadas han sido guardadas en 'deteccion_anomalias.csv'.")

# ============================
# 8. Comparación de Resultados
# ============================

plt.figure(figsize=(10,6))
sns.histplot(actuals_inv - predictions_inv, kde=True, bins=50)
plt.title('Distribución de Errores del Transformer')
plt.xlabel('Error')
plt.ylabel('Frecuencia')
plt.show()

print("\nMétricas de Error del Transformer:")
print(f"MAE: {mae_transformer:.2f}")
print(f"RMSE: {rmse_transformer:.2f}")
