import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
#events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
#events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data

#maske definieren
dy_mask = events_dy.run3_dnn_moe_hh > 0.0   #auf hh node cutten??? oder run3_dnn_moe_dy ? probably hh

#1 grouped bar chart for seeing assignment of hadronic tau jets
flat_data = ak.flatten(events_dy.tau_genPartFlav[dy_mask])
np_data = ak.to_numpy(flat_data)
counts = np.bincount(np_data) #minlength=6 ?
x_labels = np.arange(6)

plt.figure()
plt.bar(x_labels, counts, color='#4C72B0', edgecolor='black')

plt.xlabel('Werte', fontsize=12)
plt.ylabel('Häufigkeit', fontsize=12)
plt.title('Häufigkeitsverteilung der Werte 0 bis 5', fontsize=14)

# Sicherstellen, dass nur ganze Zahlen auf der X-Achse stehen
plt.xticks(x_labels)

# Raster im Hintergrund für bessere Lesbarkeit
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("plots/bar-charts/test.png", dpi=300, bbox_inches='tight')
plt.figure()