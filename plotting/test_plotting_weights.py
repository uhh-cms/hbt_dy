import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
import torch

import os
from pathlib import Path

from utils.asimov import asimov_no_background

hist_edge_l= -14
hist_edge_r= 14
bin_num = 20

#Array der Dateinamen definieren
model_names=np.array([
    "unchanged",
    # "2x_tau",
    # "2x_e",
    # "5x_tau",
    # "5x_e",
    # "10x_mu",
    # "100x_mu"
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
    
    dataset_list_names = ["training","validation","test"]
    dataset_list = [training,validation,test]
    for i,dataset in enumerate(dataset_list):
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


        dy_tt = Hist(
            hist.axis.StrCategory([], name="Zerfallskanal", growth=True),  #diese Achse wird später gestacked
            hist.axis.Regular(50,-10,10, name="x")
        )
        hh = Hist(hist.axis.Regular(50,0,8, name="x"))
        s  = Hist(hist.axis.Regular(50,0,8, name="x"))

        #1. Histogramme nach Zerfallskanal aufteilen und befüllen (ohne weights erstmal)
        lengths = np.array([])
        length=0
        for key in dy2e_keys:
            dy_tt.fill(x=dataset[key]["eventweights"],Zerfallskanal=r"gen: DY $\to e^+e^-$")
            print(len(dataset[key]["eventweights"]))
            print(";")
            length +=len(dataset[key]["eventweights"])
            print(length)
        lengths = np.append(lengths,length)
        length=0
        for key in dy2mu_keys:
            #e = dataset.get(key, None)
            dy_tt.fill(x=dataset[key]["eventweights"],Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$")
            print(len(dataset[key]["eventweights"]))
            print(";")
            length +=len(dataset[key]["eventweights"])
            print(length)
        lengths = np.append(lengths,length)
        length=0
        for key in dy2tau_keys:
            dy_tt.fill(x=dataset[key]["eventweights"],Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$")
            print(len(dataset[key]["eventweights"]))
            print(";")
            length +=len(dataset[key]["eventweights"])
            print(length)
        lengths = np.append(lengths,length)
        length=0
        for key in tt_keys:
            dy_tt.fill(x=dataset[key]["eventweights"],Zerfallskanal=r"tt")
            print(len(dataset[key]["eventweights"]))
            print(";")
            length +=len(dataset[key]["eventweights"])
            print(length)
        lengths = np.append(lengths,length)
        length=0

        for key in hh_key:
            hh.fill(dataset[key]["eventweights"])
            print(len(dataset[key]["eventweights"]))
            print(";")
            length +=len(dataset[key]["eventweights"])
            print(length)

        from IPython import embed; embed(header="MESSAGE Line 93 | File: test_plotting_weights.py")
        means = np.concatenate([np.sum(dy_tt.values(),axis=1)/lengths,[np.sum(hh.values())/length]])
        means_names = ["dy2e:","dy2mu:","dy2tau:","tt:","hh:"]
        text_content = "\n".join([f"mean = {means_names[i]}{means[i]:.10f}" for i in range(len(means))])
        plt.figtext(0.15, 1,text_content)
        plt.xscale('log')

        hh.plot(label=r"$HH$")
        dy_tt.plot(label=r"$dy_tt$")

        plt.xlabel("eventweights")
        plt.title(f"hist of {dataset_list_names[i]} eventweights")
        plt.savefig(f"{dataset_list_names[i]}.png", dpi=100, bbox_inches='tight')
        plt.figure()

        #histogramme für nächste iteration clearen
        dy_tt.reset()
        hh.reset()