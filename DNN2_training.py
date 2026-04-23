import awkward as ak
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

dict_info={}
#to do: features und andere Daten in das Dictionary einfügen
# ==========================================
# 1. Daten laden
# ==========================================
print("Lade Parquet-Dateien...")

parquet_directory = os.environ["PARQUET_DIR"]
events_dy = ak.from_parquet(f"{parquet_directory}/dy_22pre_v14.parquet")
events_tt = ak.from_parquet(f"{parquet_directory}/tt_22pre_v14.parquet")
events_hh = ak.from_parquet(f"{parquet_directory}/hh_22pre_v14.parquet")

# Zu verwendende Variablen, die ersten 10 werden fürs Training verwendet
features = [
    'bb_pt', 'bb_eta', 'bb_phi', 'bb_mass',
    'll_pt', 'll_eta', 'll_phi', 'll_mass',
    'met_pt', 'met_phi', 
    'channel_id', 'event_weight' #evtl hier noch mehr Infos hinzufügen   ('gen_dy_tau_decayproducts', 'gen_ll_pdgid')
]
dict_info["features"]=features

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
X = np.vstack([X_dy, X_tt, X_hh])           # np.shape(X):(3453729,10+2)
y = np.concatenate([y_dy, y_tt, y_hh])      # np.shape(y): (3453729,)


# Train-Test-Split (70% Training, 15% Validierung, 15% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

#Trainingsdaten von weiteren Infos trennen
X_train = X_train[:,:10]
X_val_info = X_val[:,-2:]
X_val = X_val[:,:10]
X_test_info = X_test[:,-2:]
X_test = X_test[:,:10]

dict_info["X_test_info"] = X_test_info


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
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

#kürzere Trainingszeit:
#==========================
nr_events = len(X_train)
#==========================

batch_size = 512
train_loader = DataLoader(TensorDataset(X_train_t[:nr_events], y_train_t[:nr_events]), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)
dict_info["val_loader"] = val_loader
dict_info["test_loader"] = test_loader

# ==========================================
# 4. Definition des Neuronalen Netzes
# input_dim - 128 - 64 - 32 - num_classes
# ==========================================
class MultiClassNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiClassNN, self).__init__()
        # Einfaches Feed-Forward Netzwerk
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),     # Hilft beim schnellen und stabilen Konvergieren
            nn.Dropout(0.3),         # Verhindert Overfitting
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, num_classes) 
            # Wichtig: Kein Softmax hier! CrossEntropyLoss wendet das intern an.
        )

    def forward(self, x):
        return self.network(x)

# Modell initialisieren
input_dim = 10 ############### Achtung! adaptiert nicht automatisch
dict_info["input_dim"]= input_dim
num_classes = 3 # DY, TT, HH
dict_info["num_classes"]=num_classes
model = MultiClassNN(input_dim, num_classes)

# Falls eine GPU vorhanden ist, nutze sie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dict_info["device"]=device
model.to(device)
print(f"Trainiere auf Gerät: {device}")

# ==========================================
# 5. Loss und Optimizer
# ==========================================
# CrossEntropyLoss erwartet die rohen Logits (ohne Softmax) und Integer-Labels
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 6. Trainings-Schleife
# ==========================================
epochs = 20
dict_info["epochs"]=epochs

train_loss_list=[]   
val_loss_list=[]  #liste für loss definieren
train_auc_history = []
val_auc_history = []

print("Starte Training...")


for epoch in range(epochs):
    # -- Training --
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()           # Gradienten zurücksetzen
        outputs = model(batch_X)        # Forward Pass
        loss = criterion(outputs, batch_y) # Loss berechnen
        
        loss.backward()                 # Backward Pass
        optimizer.step()                # Gewichte aktualisieren
        
        train_loss += loss.item() * batch_X.size(0)
        
        # Genauigkeit (Accuracy) berechnen
        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()

    train_loss /= total_train
    train_acc = correct_train / total_train

    train_loss_list.append(train_loss)

    # -- Validierung --
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad(): # Keine Gradientenberechnung für die Evaluierung
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            val_loss += loss.item() * batch_X.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += batch_y.size(0)
            correct_val += (predicted == batch_y).sum().item()
            
    val_loss /= total_val
    val_acc = correct_val / total_val
    val_loss_list.append(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


#listen zum dict hinzufügen
dict_info["train_loss_list"] = train_loss_list
dict_info["val_loss_list"] = val_loss_list
dict_info["train_auc_history"] = train_auc_history
dict_info["val_auc_history"] = val_auc_history



# Optional: Speichern des trainierten Modells
torch.save(model.state_dict(), "hh2bbtautau_multiclass_model.pth") # heißt: save the state_dictionary of the model under this pathwe
#Ansatz: weiteres Dictionary saven, in dem weitere Informationen sind
torch.save(dict_info, "model_info.pth")
