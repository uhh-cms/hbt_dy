import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, BatchNormalization, LeakyReLU
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc  später noch installieren
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import AUC
from numpy import loadtxt
import time
import awkward as ak

# Datei einlesen
events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data


#df = pd.read_csv("HIGGS.csv.gz")  alte daten

# Die ersten Zeilen anzeigen
#print(events_dy.typestr) 
print(ak.fields(events_dy))
print(events_dy[1].show())
print(events_dy.type)



# 1. Wähle die Features aus, die das Modell lernen soll
features = ["bb_eta", "bb_mass", "bb_phi", "bb_pt", "category_ids", "channel_id", "event", "event_weight", "gen_dy_tau_decayproducts", "gen_ll_pdgid", "gen_ll_pt", "jet1_eta", "jet1_phi", "jet1_pt", "keep_in_union", "lep1_eta", "lep1_phi", "lep1_pt", "leptons_os", "ll_eta", "ll_mass", "ll_phi", "ll_pt", "llbb_eta", "llbb_mass", "llbb_phi", "llbb_pt", "luminosityBlock", "met_phi", "met_pt", "n_btag_pnet", "n_btag_pnet_hhb", "n_jet", "process_id", "reg_dnn_moe_nu1_px", "reg_dnn_moe_nu1_py", "reg_dnn_moe_nu1_pz", "reg_dnn_moe_nu2_px", "reg_dnn_moe_nu2_py", "reg_dnn_moe_nu2_pz", "run", "run3_dnn_moe_dy", "run3_dnn_moe_hh", "run3_dnn_moe_tt", "tau_decayMode", "tau_decayModePNet", "tau_decayModeUParT", "tau_eta", "tau_genPartFlav", "tau_phi", "tau_pt"]

# 2. Extrahiere diese aus den Objekten und stacke sie zu einer 2D-Matrix
# Wir nutzen ak.unflatten oder einfaches np.stack, falls sie nicht jagged sind
input_matrix = ak.unflatten([events_dy[f] for f in features])

print(input_matrix.shape) 
# Output: (1441251, 4) -> Das ist jetzt eine echte 2D-Matrix!


#dataset= events_dy.to_numpy()
#print(np.shape(dataset))

##dataset32= df.to_numpy(dtype=np.float32) # Dateigröße halbiert, hiermit arbeiten. (nach tests: doch nicht nötig)

##  df.describe()  alles ok, aber negative Werte -> leaky relu
##  df.types        alles float64  -> sehr groß, evtl nur float32 oder float16 nötig

#df.describe()
#plt.hist(events_dy.bb_mass)    # -> float32 nötig weil viele punkte um minimum bei 10^-4. Evtl am ende trotzdem float16 ausprobieren. Hat aber nichts an der zeit geändert-> tensorflow arbeitet ohnehin mit flota32
#print(min(events_dy.bb_mass))



# stds = df.std(axis=0, ddof=1)   # ddof=1 → Stichproben-Standardabweichung

# Minimum und Maximum der Standardabweichungen
# std_min = stds.min()
# std_max = stds.max()

# print(stds)
# print("Minimale Standardabweichung:", std_min)
# print("Maximale Standardabweichung:", std_max)


x=dataset[:,1:29]
y=dataset[:,0]

#normierung nicht nötig

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=10)



model=Sequential()
model.add(Dense(300,input_dim=28))      # ,activation='relu'))                  #input layer
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(200))                  # ,activation='relu'))                 #300 nodes a 5 layers sind empfohlen.
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(100))                  # ,activation='relu'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(50))                  # ,activation='relu'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(25))                  # ,activation='relu'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

# evtl noch model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) #(evtl sigmoid noch ändern?) und relu evtl auch noch ändern


# Lernrate reduzieren, wenn Validierungsverlust stagniert
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # beobachte den Validierungsverlust
    factor=0.5,           # LR wird halbiert
    patience=5,           # nach 5 Epochen ohne Verbesserung
    min_lr=1e-6,          # minimale Lernrate
    verbose=1
)

# Frühes Stoppen, um Overfitting zu vermeiden
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,          # stoppt nach 10 Epochen ohne Verbesserung
    restore_best_weights=True,
    verbose=1
)



epochs=2

start = time.time()
history = model.fit(
    x_train, y_train,
    validation_split=0.2,  # 20% der Daten für Validierung
    epochs=epochs,            # maximale Epochen
    batch_size=1024,
    verbose=1,
    callbacks=[reduce_lr, early_stop]  # <-- hier einfügen
)
end = time.time()
#callbacks, reduce lr on plateaus, auc im model.compile einfügen
print(f"Dauer: {end - start:.2f} Sekunden")



arr = np.arange(1, len(train_loss)+1)
plt.plot(arr,val_loss,label="Validation loss")
plt.plot(arr,train_loss, label="Train loss")
#plt.ylim(0, 1)
plt.title("Training and validation loss")
plt.xlabel("epochs")
plt.ylabel("value of loss function")
plt.legend()
# x=[58,65,72,77,82,87]
# for i in x:
#     plt.axvline(i, linestyle='--',color="black")
# plt.axvline(x=67, linestyle="--", color="red")



plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Validation AUC')
#plt.ylim(0,1)
plt.title("Area under the curve")
plt.xlabel('Epoche')
plt.ylabel('AUC')
plt.legend()
# x=[58,65,72,77,82,87]
# for i in x:
#     plt.axvline(i, linestyle='--',color="black")
# plt.axvline(x=67, linestyle="--", color="red")



#ROC curve
# Vorhersagen auf Testdaten (Wahrscheinlichkeiten für Signal)
y_pred = model.predict(x_test).ravel()  # 1D Array

# Echte Labels
y_true = y_test

# ROC-Kurve berechnen
fpr, tpr, thresholds = roc_curve(y_true, y_pred)

# AUC berechnen
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')  # Zufallslinie
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()