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

plt.savefig("plots/fake-taus/origin-reco-tau_with_cut_and_weights.png", dpi=300, bbox_inches='tight')
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

plt.savefig("plots/fake-taus/origin-reco-tau_with_weights_no_cut.png", dpi=300, bbox_inches='tight')
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

plt.savefig("plots/fake-taus/origin-reco-tau_with_cut_no_weights.png", dpi=300, bbox_inches='tight')
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

plt.savefig("plots/fake-taus/origin-reco-tau_no_cut_no_weights.png", dpi=300, bbox_inches='tight')
plt.figure()


#====================================================
#2.1 bar chart for seeing origin of reconstructed tau_h with category id
#====================================================

#maske definieren
hh_cut = 0.9
dy_mask = events_dy.run3_dnn_moe_hh > hh_cut

IDs=["etau__res1b__os__iso","etau__res2b__os__iso","mutau__res1b__os__iso","mutau__res2b__os__iso","tautau__res1b__os__iso","tautau__res2b__os__iso"]

for i,id in enumerate([147,151,175,179,203,207],start=0): #id steht für category ids, i ist index
    id_mask = ak.any(events_dy.category_ids == id,axis=1)

    #weights definieren
    weights_ak = ak.zeros_like(events_dy.tau_genPartFlav) + events_dy.event_weight
    weights = ak.flatten(weights_ak[(dy_mask) & (id_mask)])

    #alle category_ids zusammen nicht alle events glaub ich...

    flat_data = ak.flatten(events_dy.tau_genPartFlav[(dy_mask) & (id_mask)])
    np_data = ak.to_numpy(flat_data)
    counts = np.bincount(np_data, weights = weights) #minlength=6 ?
    x_labels = np.arange(6)

    plt.figure()
    plt.bar(x_labels, counts, color='#4C72B0', edgecolor='black')

    plt.xlabel('Origins', fontsize=12)
    plt.ylabel('Number of events', fontsize=12)
    plt.title(fr'Origins of reconstructed $\tau_h$ ({IDs[i]}, ${hh_cut}$ cut)', fontsize=14)

    # Sicherstellen, dass nur ganze Zahlen auf der X-Achse stehen
    plt.xticks(x_labels,labels)

    # Raster im Hintergrund für bessere Lesbarkeit
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig(f"plots/fake-taus/cat_id/origin-reco-tau_{IDs[i]}.png", dpi=300, bbox_inches='tight')
    plt.figure()


#=====================================================
#3.1 DNN output für verschiedene Anzahlen an tau fakes
#=====================================================
#für etau und mutau nur die erste Zahl des ak arrays nehmen! 
weights=events_dy.event_weight

dy = Hist(
    hist.axis.StrCategory([], name="fakes_nr", growth=True),  #diese Achse wird später gestacked
    hist.axis.Regular(bins=100, start=0, stop=1, name="x")
)
#für die richtigen Farben:
dy.fill(x=[1],fakes_nr="2_fake_taus")
dy.fill(x=[1],fakes_nr="1_fake_tau")
dy.fill(x=[1],fakes_nr="0_fake_taus")
dy.reset()

IDs=["etau__res1b__os__iso","etau__res2b__os__iso","mutau__res1b__os__iso","mutau__res2b__os__iso","tautau__res1b__os__iso","tautau__res2b__os__iso"]

for i,id in enumerate([147,151,175,179,203,207],start=0): #id steht für category ids, i ist index
    id_mask = ak.any(events_dy.category_ids == id,axis=1)
    fakes_temp = ak.where(events_dy.tau_genPartFlav < 5, 1,events_dy.tau_genPartFlav)
    fakes = ak.where(fakes_temp == 5, 0,fakes_temp)

    #weights und fake anzahl definieren
    if id in [147,151,175,179]:
        #weights_ak = ak.zeros_like(events_dy.tau_genPartFlav[:,0]) + events_dy.event_weight
        fakes_sum = fakes[:,0] #nur ersen jet evaluieren
    elif id in [203,207]:
        #weights_ak = ak.zeros_like(events_dy.tau_genPartFlav) + events_dy.event_weight
        fakes_sum = ak.sum(fakes, axis=1)
        #Hist neu definieren, um Reihenfolge der Einträge zu bewahren
        dy = Hist(
        hist.axis.StrCategory([], name="fakes_nr", growth=True),  #diese Achse wird später gestacked
        hist.axis.Regular(bins=100, start=0, stop=1, name="x")
        )


    dy.fill(x=events_dy.run3_dnn_moe_hh[(id_mask) & (fakes_sum==4)],fakes_nr="4_fake_taus", weight=weights[(id_mask) & (fakes_sum==4)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(id_mask) & (fakes_sum==3)],fakes_nr="3_fake_taus", weight=weights[(id_mask) & (fakes_sum==3)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(id_mask) & (fakes_sum==2)],fakes_nr="2_fake_taus", weight=weights[(id_mask) & (fakes_sum==2)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(id_mask) & (fakes_sum==1)],fakes_nr="1_fake_tau", weight=weights[(id_mask) & (fakes_sum==1)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(id_mask) & (fakes_sum==0)],fakes_nr="0_fake_taus", weight=weights[(id_mask) & (fakes_sum==0)])
    
    # Stack-Plot erstellen
    stack = dy.stack("fakes_nr")
    stack.plot(stack=True, histtype="fill",color=["tab:green","tab:orange","tab:blue"])

    plt.yscale('log')
    plt.legend()
    plt.xlabel("Di-Higgs-outputnode of the DNN")
    plt.ylabel("Number of events")
    plt.title(f"Hist of DNN-outputnode $HH$ for DY-simulations ({IDs[i]})")
    plt.savefig(f"plots/fake-taus/cat_id/hist_nr-of-fakes_{IDs[i]}.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()