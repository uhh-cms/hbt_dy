import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data

print(events_dy.process_id)

#logit funktion definieren
def stable_logit(x, eps=1e-6, limit=5.0):
    x = np.clip(x, eps, 1 - eps) # Begrenzt x auf [0.000001, 0.999999]
    y = np.log(x / (1 - x))
    return np.clip(y,-14, limit-1e-5)

#Histogramme definieren, 2-D für dy wegen Unterteilung
dy = Hist(
    hist.axis.StrCategory([], name="Zerfallskanal", growth=True),  #diese Achse wird später gestacked
    hist.axis.Regular(bins=50, start=-14, stop=5, name="x")
)
process = Hist(hist.axis.Regular(bins=50, start=-14, stop=5, name="x"))
hh = Hist(hist.axis.Regular(bins=50, start=-14, stop=5, name="x"))
s = Hist(hist.axis.Regular(bins=50, start=-14, stop=5, name="x"))



#Plots für hintergrund, signal und bestimmten DY-process für output >= 0.9
Process_ids = [51667, 51683, 51664, 51680, 51720, 51723, 51726,
        51729, 51732, 51735, 51674, 51690, 51665, 51681,
        51661, 51677, 51670, 51671, 51672, 51673, 51675,
        51686, 51687, 51688, 51689, 51691, 51666, 51682,
        51699, 51702, 51705, 51708, 51711, 51714, 51693,
        51668, 51684, 51663, 51679]
Process_ids_name = [dy_ee_m50toinf_1j_pt400to600
, dy_mumu_m50toinf_1j_pt400to600
, dy_ee_m50toinf_1j_pt40to100
, dy_mumu_m50toinf_1j_pt40to100
, dy_tautau_m50toinf_2j_pt0to40
, dy_tautau_m50toinf_2j_pt40to100
, dy_tautau_m50toinf_2j_pt100to200
, dy_tautau_m50toinf_2j_pt200to400
, dy_tautau_m50toinf_2j_pt400to600
, dy_tautau_m50toinf_2j_pt600toinf
, dy_ee_m50toinf_2j_pt400to600
, dy_mumu_m50toinf_2j_pt400to600
, dy_ee_m50toinf_1j_pt100to200
, dy_mumu_m50toinf_1j_pt100to200
, dy_ee_m50toinf_0j
, dy_mumu_m50toinf_0j
, dy_ee_m50toinf_2j_pt0to40
, dy_ee_m50toinf_2j_pt40to100
, dy_ee_m50toinf_2j_pt100to200
, dy_ee_m50toinf_2j_pt200to400
, dy_ee_m50toinf_2j_pt600toinf
, dy_mumu_m50toinf_2j_pt0to40
, dy_mumu_m50toinf_2j_pt40to100
, dy_mumu_m50toinf_2j_pt100to200
, dy_mumu_m50toinf_2j_pt200to400
, dy_mumu_m50toinf_2j_pt600toinf
, dy_ee_m50toinf_1j_pt200to400
, dy_mumu_m50toinf_1j_pt200to400
, dy_tautau_m50toinf_1j_pt0to40
, dy_tautau_m50toinf_1j_pt40to100
, dy_tautau_m50toinf_1j_pt100to200
, dy_tautau_m50toinf_1j_pt200to400
, dy_tautau_m50toinf_1j_pt400to600
, dy_tautau_m50toinf_1j_pt600toinf
, dy_tautau_m50toinf_0j
, dy_ee_m50toinf_1j_pt600toinf
, dy_mumu_m50toinf_1j_pt600toinf
, dy_ee_m50toinf_1j_pt0to40
, dy_mumu_m50toinf_1j_pt0to40]

IDs=["etau__res1b__os__iso","etau__res2b__os__iso","mutau__res1b__os__iso","mutau__res2b__os__iso","tautau__res1b__os__iso","tautau__res2b__os__iso"]
#Neudefinition von dy wichtig um histogrammachsen neu zu definieren (anscheinend werden sie bei dy.reset gespeichert)
dy = Hist(
    hist.axis.StrCategory([], name="Zerfallskanal", growth=True),  #diese Achse wird später gestacked
    hist.axis.Regular(bins=100, start=0, stop=1, name="x")
)

for i,id in enumerate([147,151,175,179,203,207],start=0): #id steht für category ids, i ist index
    dy.fill(x=events_dy.run3_dnn_moe_hh[ak.any(events_dy.category_ids == id,axis=1) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal=r"$e^+e^-$", weight=events_dy.event_weight[ak.any(events_dy.category_ids == id,axis=1) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[ak.any(events_dy.category_ids == id,axis=1) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal=r"$\mu^+\mu^-$", weight=events_dy.event_weight[ak.any(events_dy.category_ids == id,axis=1) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[ak.any(events_dy.category_ids == id,axis=1) & (events_dy.gen_ll_pdgid == 15)],Zerfallskanal=r"$\tau^+\tau^-$", weight=events_dy.event_weight[ak.any(events_dy.category_ids == id,axis=1) & (events_dy.gen_ll_pdgid == 15)])
    dy.fill(x=events_tt.run3_dnn_moe_hh[ak.any(events_tt.category_ids == id,axis=1)],Zerfallskanal=r"$t\bar{t}$", weight=events_tt.event_weight[ak.any(events_tt.category_ids == id,axis=1)])

    hh.fill(events_hh.run3_dnn_moe_hh[ak.any(events_hh.category_ids == id,axis=1)],weight=events_hh.event_weight[ak.any(events_hh.category_ids == id,axis=1)])

    fig, ax1 = plt.subplots()

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal") #technically zerfallskanal+tt als korrekter name
    stack.plot(stack=True, histtype="fill", ax=ax1) # 'stack=True' ist entscheidend!

    hh.plot(label=r"$HH$", ax=ax1)

    ax1.set_ylabel("number of events (weighted)")
    plt.yscale('log')    #linke Achse logarithmisch skalieren 

    #zweite Achse
    background_bins = np.sum(dy.values(),axis=0)
    signal_bins = hh.values()
    significance = signal_bins/np.sqrt(background_bins)
    significance_total = round(np.sqrt(np.sum(significance**2)),3)
    ax2 = ax1.twinx()  # Erstellt die rechte Achse
    ax2.step(np.linspace(0, 1, 100),significance, label=f"significance (total = {significance_total})", color="black")
    ax2.set_ylabel('Significance')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.yscale('log')    #rechte Achse logarithmisch skalieren 

    #legende:
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Gemeinsam plotten
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=True)

    plt.xlabel("Di-Higgs-outputnode of the DNN")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins -{IDs[i]}-")
    plt.savefig(f"plots/hists_HH-outputnode/segmenting_in_pdgid_and_tau/linear_binning/stacked_tt/category_id/{IDs[i]}.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    hh.reset()