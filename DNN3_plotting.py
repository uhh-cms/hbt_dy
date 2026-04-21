#to do: die 2 dictionarys zusammenfügen
#evtl auf die naf mit grafikkarte wechseln

#import awkward as ak
import numpy as np
import torch
import torch.nn as nn
#import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#import os
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==============================================
# model einrichten, Variablen definieren
# ==============================================

#Variablen aus pth datei laden und definieren
dict_info=torch.load("model_info.pth", weights_only=False)

epochs = dict_info["epochs"]
val_loss_list = dict_info["val_loss_list"]
input_dim = dict_info["input_dim"]
num_classes = dict_info["num_classes"]
train_loss_list = dict_info["train_loss_list"]
val_loader=dict_info["val_loader"]
device= dict_info["device"]

#Networkclass definieren
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

#model mit weihts füllen
model=MultiClassNN(input_dim, num_classes)
model.load_state_dict(torch.load("hh2bbtautau_multiclass_model.pth"))


model.eval()
# ==========================================
# 7. ROC-Kurve und AUC evaluieren & plotten
# ==========================================

#loss über die epochen plotten
plt.plot(np.arange(epochs),val_loss_list,label="val_loss")
plt.plot(np.arange(epochs),train_loss_list,label="train_loss")
plt.title("Training and validation loss")
plt.xlabel("epochs")
plt.ylabel("value of loss function")
plt.legend()
plt.savefig("loss_epochs.png")
plt.figure()


#ROC-Kurve berechnen
print("Berechne Wahrscheinlichkeiten für ROC-Kurve...")

all_preds = []
all_labels = []

# Vorhersagen für das Validierungsset sammeln
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        
        # Wichtig: Da das Netzwerk unskalierte Logits ausgibt, 
        # müssen wir Softmax anwenden, um Wahrscheinlichkeiten zwischen 0 und 1 zu erhalten.
        probs = F.softmax(outputs, dim=1)
        
        # Tensoren vom VRAM in den normalen RAM (CPU) verschieben und in NumPy konvertieren
        all_preds.append(probs.cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())

# Listen zu vollständigen NumPy-Arrays zusammenfügen
all_preds = np.vstack(all_preds)
all_labels = np.concatenate(all_labels)

# Labels binarisieren für One-vs-Rest (0: DY, 1: TT, 2: HH)
classes = [0, 1, 2]
class_names = ['DY (0)', 'TT (1)', 'HH (Signal) (2)']
y_val_bin = label_binarize(all_labels, classes=classes)

# Plot initialisieren
plt.figure(figsize=(10, 8))

# ROC und AUC für jede einzelne Klasse berechnen und plotten
for i in range(len(classes)):
    # fpr = False Positive Rate, tpr = True Positive Rate
    fpr, tpr, _ = roc_curve(y_val_bin[:, i], all_preds[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

# Diagonale Linie (Zufallsraten-Baseline)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.500)')

# Plot formatieren
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Background Efficiency)', fontsize=12)
plt.ylabel('True Positive Rate (Signal Efficiency)', fontsize=12)
plt.title('Multiclass ROC Curves (One-vs-Rest)', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)

# Plot speichern und anzeigen
plt.savefig("roc_auc_multiclass.png", dpi=300, bbox_inches='tight')
print("Plot wurde als 'roc_auc_multiclass.png' gespeichert.")
plt.show()
