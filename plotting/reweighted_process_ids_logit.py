# plotting the results of my different DNNs (trained with different sampling weights for the process ids).
#trained on 2022pre; planned: 2022 pre/post & 2023 pre/post MC data

import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import torch

import os
from pathlib import Path

bin_num = 20

#Array der Dateinamen definieren
model_names=np.array([
    "unchanged",
    "2x_tau",
    "2x_e",
    "5x_tau",
    "5x_e",
    "10x_mu",
    "100x_mu"
])

# Holt den Pfad aus $EVALUATION_DIR und wandelt ihn in ein Path-Objekt um
for i,k in enumerate(model_names):
    if 'EVALUATION_DIR' in os.environ:
        basis_pfad = Path(os.environ['EVALUATION_DIR'])
        
        # Direkt auf die Datei zugreifen
        datei_pfad = basis_pfad / f'{k}.pt'
        print(f"{k}.pt")

        # Datei direkt lesen
        events = torch.load(datei_pfad)

        training = events[0]["training"]
        validation = events[0]["validation"]
        test = events[0]["test"]
    else:
        print("Die Umgebungsvariable $EVALUATION_DIR existiert nicht.")

    #DY,tt und HH keys sortieren
    alle_keys = list(test.keys())
    dy_keys = [key for key in test.keys() if key[0] == 'dy']
    tt_keys = [key for key in test.keys() if key[0] == 'tt']
    hh_keys = [key for key in test.keys() if key[0] == 'hh']
    #keys der DY-subprozessen
    dy2e_keys=  (('dy', 51667), ('dy', 51664), ('dy', 51674), ('dy', 51665), ('dy', 51661), ('dy', 51670), 
    ('dy', 51671), ('dy', 51672), ('dy', 51673), ('dy', 51675), ('dy', 51666), ('dy', 51668), ('dy', 51663))
    dy2mu_keys= (('dy', 51683), ('dy', 51680), ('dy', 51690), ('dy', 51681), ('dy', 51677), ('dy', 51686), 
    ('dy', 51687), ('dy', 51688), ('dy', 51689), ('dy', 51691), ('dy', 51682), ('dy', 51684), 
    ('dy', 51679))
    dy2tau_keys=(('dy', 51720), ('dy', 51723), ('dy', 51726), ('dy', 51729), ('dy', 51732), ('dy', 51735), 
    ('dy', 51699), ('dy', 51702), ('dy', 51705), ('dy', 51708), ('dy', 51711), ('dy', 51714), ('dy', 51693))

    #falls key in dy_keys nicht vertreten ist
    dy2e_keys = [key for key in dy_keys if key in dy2e_keys]
    dy2mu_keys = [key for key in dy_keys if key in dy2mu_keys]
    dy2tau_keys = [key for key in dy_keys if key in dy2tau_keys]


    #logit funktion definieren
    def stable_logit(x, eps=1e-6, limit=14.0):
    #    x = np.clip(x, eps, 1 - eps) # Begrenzt x auf [0.000001, 0.999999]
    #    y = np.log(x / (1 - x))
    #    return np.clip(y,-14, limit-1e-5)
        return x

    #Histogramme definieren, 2-D für dy wegen Unterteilung
    dy_tt = Hist(
        hist.axis.StrCategory([], name="Zerfallskanal", growth=True),  #diese Achse wird später gestacked
        hist.axis.Regular(bins= bin_num, start=-14, stop=14, name="x")
    )
    hh = Hist(hist.axis.Regular(bins= bin_num, start=-14, stop=14, name="x"))
    s =  Hist(hist.axis.Regular(bins= bin_num, start=-14, stop=14, name="x"))

    #1. Histogramme nach Zerfallskanal aufteilen und befüllen (ohne weights erstmal)
    for key in dy2e_keys:
        dy_tt.fill(x=stable_logit(test[key]["scores"][:,0]),Zerfallskanal=r"gen: DY $\to e^+e^-$",weight=test[key]["eventweights"])
    for key in dy2mu_keys:
        #e = test.get(key, None)
        dy_tt.fill(x=stable_logit(test[key]["scores"][:,0]),Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$",weight=test[key]["eventweights"])
    for key in dy2tau_keys:
        dy_tt.fill(x=stable_logit(test[key]["scores"][:,0]),Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$",weight=test[key]["eventweights"])
    for key in tt_keys:
        dy_tt.fill(x=stable_logit(test[key]["scores"][:,0]),Zerfallskanal=r"tt",weight=test[key]["eventweights"])

    for key in hh_keys:
        hh.fill(stable_logit(test[key]["scores"][:,0]),weight=test[key]["eventweights"])

    fig, ax1 = plt.subplots()

    # Stack-Plot erstellen
    stack = dy_tt.stack("Zerfallskanal")
    stack.plot(stack=True, histtype="fill",ax=ax1) # 'stack=True' ist entscheidend!

    hh.plot(label=r"$HH$",ax=ax1)

    ax1.set_ylabel("number of events")
    plt.yscale('log')    #linke Achse logarithmisch skalieren 
    plt.grid()

    #zweite Achse
    background_bins = np.sum(dy_tt.values(),axis=0)
    signal_bins = hh.values()
    significance = signal_bins/np.sqrt(background_bins)
    significance = np.nan_to_num(significance, nan=0.0,posinf=0.0) #ist hier möglich, da signal bei den Problemstellen = 0 ist?
    significance_total = round(np.sqrt(np.sum(significance**2)),3)
    ax2 = ax1.twinx()  # Erstellt die rechte Achse
    ax2.step(np.linspace(-14+28/bin_num, 14, bin_num),significance, label=f"significance (total = {significance_total})", color="black")
    ax2.set_ylabel('Significance')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.yscale('log')    #rechte Achse logarithmisch skalieren 

    #legende:
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Gemeinsam plotten
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=True)

    plt.xlabel("Di-Higgs-outputnode of DNN")
    plt.title(f"$HH$-outputnode - {k}")
    plt.savefig(f"plots/reweighted_process_id/{k}.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy_tt.reset()
    hh.reset()