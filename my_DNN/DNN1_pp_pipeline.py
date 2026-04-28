import awkward as ak
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. Daten laden
# ==========================================
print("Lade Parquet-Dateien...")

parquet_directory = os.environ["PARQUET_DIR"]
events_dy = ak.from_parquet(f"{parquet_directory}/dy_22pre_v14.parquet")
events_tt = ak.from_parquet(f"{parquet_directory}/tt_22pre_v14.parquet")
events_hh = ak.from_parquet(f"{parquet_directory}/hh_22pre_v14.parquet")

# Zu verwendende Variablen
features = [
    'bb_pt', 'bb_eta', 'bb_phi', 'bb_mass',
    'll_pt', 'll_eta', 'll_phi', 'll_mass',
    'met_pt', 'met_phi'
]

# ==========================================
# 2. Datenvorbereitung (Awkward -> NumPy)
# ==========================================
# Hilfsfunktion, um Features zu extrahieren und Labels zuzuweisen
def extract_features(events, label):
    # Wandle jedes Feature in ein 1D NumPy-Array um und staple sie als Spalten
    # Hinweis: Wir gehen davon aus, dass dies flache Arrays pro Event sind (keine Jagged-Arrays).
    data = np.column_stack([ak.to_numpy(events[feat]) for feat in features])
    labels = np.full(data.shape[0], label, dtype=np.int64)
    return data, labels

print("Extrahiere Features...")
# Klassenzuweisung: DY = 0, TT = 1, HH = 2
X_dy, y_dy = extract_features(events_dy, 0)      # np.shape(X_dy):(1441251,10), np.shape(y_dy): (1441251,)
X_tt, y_tt = extract_features(events_tt, 1)
X_hh, y_hh = extract_features(events_hh, 2)

# Kombiniere alle Datensätze
X = np.vstack([X_dy, X_tt, X_hh])           # np.shape(X):(3453729,10)
y = np.concatenate([y_dy, y_tt, y_hh])      # np.shape(y): (3453729,)

total_events = 10000 # len(X)
X = X[:total_events]
y = y[:total_events]

# Train-Test-Split (80% Training, 20% Validierung)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Skalierung (Standardisierung) der Input-Variablen - sehr wichtig für Neuronale Netze!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ==========================================
# 3. PyTorch DataLoaders erstellen
# ==========================================
# Konvertiere NumPy-Arrays in PyTorch-Tensoren


X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)


batch_size = 512
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
