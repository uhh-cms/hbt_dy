import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
#events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
#events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data

#labels 
labels = [
    'Unknown',# 0
    r'Prompt $e$',         # 1
    r'Prompt $\mu$',        # 2
    r'$\tau_e$',        # 3
    r'$\tau_\mu$',      # 4
    r'$\tau_h$'         # 5
]

#====================================================
#1 bar chart for seeing origin of reconstructed tau_h
# with weights and cut
#====================================================

#maske definieren
hh_cut = 0.9
dy_mask = events_dy.run3_dnn_moe_hh > hh_cut

#weights definieren
weights_ak = ak.zeros_like(events_dy.tau_genPartFlav) + events_dy.event_weight
weights = ak.flatten(weights_ak[dy_mask])

flat_data = ak.flatten(events_dy.tau_genPartFlav[dy_mask])
np_data = ak.to_numpy(flat_data)
counts = np.bincount(np_data, weights = weights) #minlength=6 ?
x_labels = np.arange(6)

plt.figure()
plt.bar(x_labels, counts, color='#4C72B0', edgecolor='black')

plt.xlabel('Origins', fontsize=12)
plt.ylabel('Number of events', fontsize=12)
plt.title(fr'Origins of reconstructed $\tau_h$ with weights and ${hh_cut}$ cut', fontsize=14)

# Sicherstellen, dass nur ganze Zahlen auf der X-Achse stehen
plt.xticks(x_labels,labels)

# Raster im Hintergrund für bessere Lesbarkeit
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("plots/bar-charts/origin-reco-tau_with_cut_and_weights.png", dpi=300, bbox_inches='tight')
plt.figure()

#====================================================
#2 bar chart for seeing origin of reconstructed tau_h
# with weights, no cut
#====================================================

#maske definieren
hh_cut = 0.0
dy_mask = events_dy.run3_dnn_moe_hh > hh_cut

#weights definieren
weights_ak = ak.zeros_like(events_dy.tau_genPartFlav) + events_dy.event_weight
weights = ak.flatten(weights_ak[dy_mask])

flat_data = ak.flatten(events_dy.tau_genPartFlav[dy_mask])
np_data = ak.to_numpy(flat_data)
counts = np.bincount(np_data, weights = weights) #minlength=6 ?
x_labels = np.arange(6)

plt.figure()
plt.bar(x_labels, counts, color='#4C72B0', edgecolor='black')

plt.xlabel('Origins', fontsize=12)
plt.ylabel('Number of events', fontsize=12)
plt.title(r'Origins of reconstructed $\tau_h$ with weights, no cut', fontsize=14)

# Sicherstellen, dass nur ganze Zahlen auf der X-Achse stehen
plt.xticks(x_labels,labels)

# Raster im Hintergrund für bessere Lesbarkeit
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("plots/bar-charts/origin-reco-tau_with_weights_no_cut.png", dpi=300, bbox_inches='tight')
plt.figure()

#====================================================
#3 bar chart for seeing origin of reconstructed tau_h
# without weights, with cut
#====================================================

#maske definieren
hh_cut = 0.9
dy_mask = events_dy.run3_dnn_moe_hh > hh_cut

# #weights definieren
# weights_ak = ak.zeros_like(events_dy.tau_genPartFlav) + events_dy.event_weight
# weights = ak.flatten(weights_ak[dy_mask])

flat_data = ak.flatten(events_dy.tau_genPartFlav[dy_mask])
np_data = ak.to_numpy(flat_data)
counts = np.bincount(np_data) #minlength=6 ?
x_labels = np.arange(6)

plt.figure()
plt.bar(x_labels, counts, color='#4C72B0', edgecolor='black')

plt.xlabel('Origins', fontsize=12)
plt.ylabel('Number of events', fontsize=12)
plt.title(fr'Origins of reconstructed $\tau_h$, no weights, ${hh_cut}$ cut', fontsize=14)

# Sicherstellen, dass nur ganze Zahlen auf der X-Achse stehen
plt.xticks(x_labels,labels)

# Raster im Hintergrund für bessere Lesbarkeit
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("plots/bar-charts/origin-reco-tau_with_cut_no_weights.png", dpi=300, bbox_inches='tight')
plt.figure()

#====================================================
#4 bar chart for seeing origin of reconstructed tau_h
# no weights, no cut
#====================================================

#maske definieren
hh_cut = 0.0
dy_mask = events_dy.run3_dnn_moe_hh > hh_cut

# #weights definieren
# weights_ak = ak.zeros_like(events_dy.tau_genPartFlav) + events_dy.event_weight
# weights = ak.flatten(weights_ak[dy_mask])

flat_data = ak.flatten(events_dy.tau_genPartFlav[dy_mask])
np_data = ak.to_numpy(flat_data)
counts = np.bincount(np_data) #minlength=6 ?
x_labels = np.arange(6)

plt.figure()
plt.bar(x_labels, counts, color='#4C72B0', edgecolor='black')

plt.xlabel('Origins', fontsize=12)
plt.ylabel('Number of events', fontsize=12)
plt.title(r'Origins of reconstructed $\tau_h$, no weights, no cut', fontsize=14)

# Sicherstellen, dass nur ganze Zahlen auf der X-Achse stehen
plt.xticks(x_labels,labels)

# Raster im Hintergrund für bessere Lesbarkeit
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("plots/bar-charts/origin-reco-tau_no_cut_no_weights.png", dpi=300, bbox_inches='tight')
plt.figure()