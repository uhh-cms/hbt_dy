# plotting the results of my different DNNs (trained with different sampling weights for the process ids).
#trained on 2022pre; planned: 2022 pre/post & 2023 pre/post MC data
#DNN output scores come in logit form

import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Literal

import os
from pathlib import Path

from utils.asimov import asimov_no_background
from utils.flat_s_binning import def_equbin

hist_edge_l= -14
hist_edge_r= 14
bin_num = 20
b_min = 1 #minimal allowed background in last bin
dataset_key: Literal["training","validation","test"] = "test" 


#Array der Dateinamen definieren
model_names=np.array([
    "unchanged"
    # "2x_tau",
    # "2x_e",
    # "5x_tau",
    # "5x_e",
    # "10x_mu",
    # "100x_mu",
    # "1.5x_tau"
])

# Holt den Pfad aus $EVALUATION_DIR und wandelt ihn in ein Path-Objekt um
for i,model in enumerate(model_names):
    if 'EVALUATION_DIR' in os.environ:
        basis_pfad = Path(os.environ['EVALUATION_DIR'])
        
        # Direkt auf die Datei zugreifen
        datei_pfad = basis_pfad / f'{model}_{dataset_key}.pt'
        print(f"{model}.pt")

        # Datei direkt lesen
        events = torch.load(datei_pfad)
        dataset = events[0]#[dataset_key]
    else:
        print("Die Umgebungsvariable $EVALUATION_DIR existiert nicht.")
    
    #DY,tt und HH keys sortieren
    alle_keys = sorted(list(dataset.keys()))
    #dataset_keys = sorted(dataset.keys())
    dy_keys = [key for key in dataset.keys() if key[0] == 'dy']
    tt_keys = [key for key in dataset.keys() if key[0] == 'tt']
    hh_key  = [key for key in dataset.keys() if key[0] == 'hh']
    #keys der DY-subprozessen
    dy2e_keys= (
        ('dy', 51667), ('dy', 51664), ('dy', 51674), ('dy', 51665), ('dy', 51661), ('dy', 51670), 
        ('dy', 51671), ('dy', 51672), ('dy', 51673), ('dy', 51675), ('dy', 51666), ('dy', 51668), ('dy', 51663))
    dy2mu_keys= (
        ('dy', 51683), ('dy', 51680), ('dy', 51690), ('dy', 51681), ('dy', 51677), ('dy', 51686), 
        ('dy', 51687), ('dy', 51688), ('dy', 51689), ('dy', 51691), ('dy', 51682), ('dy', 51684), 
        ('dy', 51679))
    dy2tau_keys= (
        ('dy', 51720), ('dy', 51723), ('dy', 51726), ('dy', 51729), ('dy', 51732), ('dy', 51735), 
        ('dy', 51699), ('dy', 51702), ('dy', 51705), ('dy', 51708), ('dy', 51711), ('dy', 51714), ('dy', 51693))

    #falls key in dy_keys nicht vertreten ist
    dy2e_keys = [key for key in dy_keys if key in dy2e_keys]
    dy2mu_keys = [key for key in dy_keys if key in dy2mu_keys]
    dy2tau_keys = [key for key in dy_keys if key in dy2tau_keys]

    #clip funktion definieren
    def clip(x, limit_l= hist_edge_l, limit_r= hist_edge_r):
        return np.clip(x,limit_l,limit_r)

    #Histogramme definieren, 2-D für dy wegen Unterteilung
    for l,key in enumerate(hh_key):
        bin_edges = def_equbin(dataset[key]["scores"][:,0], bin_num=bin_num-1, hist_edge_l=hist_edge_l)
        if bin_edges[-1] == bin_edges[-2]:
            print("two identical bin edges!")
            print("old bin edges:", bin_edges)
            bin_edges = bin_edges[:-1]
            print("new bin edges:", bin_edges)
        if l > 0:
            print("incorrect code since there are multiple keys in dy_key")
    

    channel_mapping = {
        r"gen: DY $\to e^+e^-$": dy2e_keys,
        r"gen: DY $\to \mu^+\mu^-$": dy2mu_keys,
        r"gen: DY $\to \tau^+\tau^-$": dy2tau_keys,
        r"tt": tt_keys
    }




    # 1. Alle Background-Scores und Weights für eine schnelle Evaluierung sammeln
    bg_scores_list = []
    bg_weights_list = []
    for label, key_list in channel_mapping.items():
        for key in key_list:
            bg_scores_list.append(clip(dataset[key]["scores"][:,0]))
            bg_weights_list.append(dataset[key]["eventweights"])
            
    # Zu 1D-Arrays zusammenfassen
    bg_scores_arr = np.concatenate(bg_scores_list)
    bg_weights_arr = np.concatenate(bg_weights_list)

    # 2. Iterativ Bins zusammenfügen, bis ALLE Bins >= b_min Background haben
    while True:
        # Schnelles Zählen der Background-Events pro Bin mit Numpy
        b_counts, _ = np.histogram(bg_scores_arr, bins=bin_edges, weights=bg_weights_arr)
        
        # Finde die Indizes aller Bins, die weniger als b_min Events haben
        failed_bins = np.where(b_counts < b_min)[0]
        
        # Abbruchbedingung: Alle Bins sind in Ordnung oder das Histogramm ist komplett zusammengeschmolzen
        if len(failed_bins) == 0 or len(bin_edges) <= 2:
            break
            
        # Wir beheben das Problem von rechts nach links (aus der Signalregion heraus)
        idx = failed_bins[-1]
        
        if idx == 0:
            # Ist das allererste Bin fehlerhaft, verschmelzen wir es mit dem rechten Nachbarn
            edge_to_remove = 1
        else:
            # Ansonsten verschmelzen wir es in Richtung Background-Region (mit dem linken Nachbarn)
            edge_to_remove = idx
            
        # Entferne die entsprechende Bin-Kante
        bin_edges = np.delete(bin_edges, edge_to_remove)

    # 3. Nachdem die sicheren bin_edges feststehen, wird das finale Histogramm befüllt
    dy_tt = Hist(
        hist.axis.StrCategory([], name="Zerfallskanal", growth=True), 
        hist.axis.Variable(bin_edges, name="x")
    )

    hh = Hist(hist.axis.Variable(bin_edges, name="x"))

    for label, key_list in channel_mapping.items():
        for key in key_list:
            dy_tt.fill(
                x=clip(dataset[key]["scores"][:,0]),
                Zerfallskanal=label,
                weight=dataset[key]["eventweights"]
            )

    for key in hh_key:
        hh.fill(clip(dataset[key]["scores"][:,0]),weight=dataset[key]["eventweights"])

    fig, ax1 = plt.subplots()   

    # Stack-Plot erstellen
    stack = dy_tt.stack("Zerfallskanal")
    stack.plot(stack=True, histtype="fill",ax=ax1) # 'stack=True' ist entscheidend!

    hh.plot(label=r"$HH$",ax=ax1)

    ax1.set_ylabel("number of events")
    plt.yscale('log')    #linke Achse logarithmisch skalieren 
    plt.grid()

    ax2 = ax1.twinx()  # Erstellt die rechte Achse
    binned_asimov = asimov_no_background(hh.values(),np.sum(dy_tt.values(),axis=0))
    total_asimov = np.sum(binned_asimov)
    ax2.stairs(binned_asimov, bin_edges, label=f"significance (total = {total_asimov:.3f})", color="black") #\u00B1 {significance_total_std}
    ax2.set_ylabel('Significance')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.yscale('log')    #rechte Achse logarithmisch skalieren 

    #legende:
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Gemeinsam plotten
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0, 1.3), ncol=2, frameon=True)

    plt.xlabel("Di-Higgs-outputnode of DNN")
    plt.title(f"$HH$-outputnode - {model}")
    plt.savefig(f"plots/reweighted_process_id/{model}_flat_s.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    #histogramme für nächste iteration clearen
    dy_tt.reset()
    hh.reset()