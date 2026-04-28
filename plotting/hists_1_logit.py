import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logit

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data

#logit funktion definieren
def stable_logit(x, eps=1e-6, limit=5.0):
    x = np.clip(x, eps, 1 - eps) # Begrenzt x auf [0.000001, 0.999999]
    y = np.log(x / (1 - x))
    return np.clip(y,-14, limit - 1e-5)

#Histogramme definieren, 2-D für dy wegen Unterteilung
dy = Hist(
    hist.axis.StrCategory([], name="Zerfallskanal", growth=True),  #diese Achse wird später gestacked
    hist.axis.Regular(bins=100, start=-14, stop=5, name="x")
)
tt = Hist(hist.axis.Regular(bins=100, start=-14, stop=5, name="x"))
hh = Hist(hist.axis.Regular(bins=100, start=-14, stop=5, name="x"))
s = Hist(hist.axis.Regular(bins=100, start=-14, stop=5, name="x"))

#Namen der decay channel definieren:
channelname=["e-tau", "mu-tau", "tau-tau"]
channelname_r=[r"$\tau_e\tau_h$",r"$\tau_\mu\tau_h$",r"$\tau_h\tau_h$"]


#1. Histogramme nach channel aufteilen, fillen (für dy nach Zerfallskanal aufteilen + stacken), plotten.
for i in [1,2,3]:
    dy.fill(x=stable_logit(events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)]),Zerfallskanal=r"gen: DY $\to e^+e^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=stable_logit(events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)]),Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=stable_logit(events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)]),Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)])

    tt.fill(stable_logit(events_tt.run3_dnn_moe_hh[events_tt.channel_id == i]),weight=events_tt.event_weight[events_tt.channel_id == i])
    hh.fill(stable_logit(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i]),weight=events_hh.event_weight[events_hh.channel_id == i])


    fig, ax1 = plt.subplots()

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal")
    stack.plot(stack=True, histtype="fill",ax=ax1) # 'stack=True' ist entscheidend!

    tt.plot(label=r"$t\bar{t}$",ax=ax1)
    hh.plot(label=r"$HH$",ax=ax1)

    ax1.set_ylabel("number of events (weighted)")
    plt.yscale('log')    #linke Achse logarithmisch skalieren 

    #zweite Achse
    background_bins = np.sum(dy.values(),axis=0)+ tt.values()
    signal_bins = hh.values()
    significance = signal_bins**2/background_bins
    significance_total = round(np.sqrt(np.sum(significance**2)),3)
    ax2 = ax1.twinx()  # Erstellt die rechte Achse
    ax2.step(np.linspace(-14, 14, 100),significance, label=f"significance (total = {significance_total})", color="black")
    ax2.set_ylabel('Significance')
    ax2.tick_params(axis='y', labelcolor='black')

    plt.yscale('log')    #rechte Achse logarithmisch skalieren 

    #legende:
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # Gemeinsam plotten
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, frameon=True)


    plt.xlabel("Di-Higgs-outputnode of the DNN - corrected with logit function")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins - {channelname_r[i-1]}-channel")
    plt.savefig(f"plots/hist_hhnode/logit_{channelname[i-1]}-channel.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    tt.reset()
    hh.reset()




#2. wie 1, aber tt wird zum stack hinzugefügt
for i in [1,2,3]: #i steht für den channel
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal=r"gen: DY $\to e^+e^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)],Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)])
    dy.fill(x=events_tt.run3_dnn_moe_hh[events_tt.channel_id == i],Zerfallskanal=r"$t\bar{t}$", weight=events_tt.event_weight[events_tt.channel_id == i])

    hh.fill(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i],weight=events_hh.event_weight[events_hh.channel_id == i])

    fig, ax1 = plt.subplots()

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal") #technically zerfallskanal+tt als korrekter name
    stack.plot(stack=True, histtype="fill", ax=ax1) # 'stack=True' ist entscheidend!

    hh.plot(label=r"$HH$",ax=ax1)

    ax1.set_ylabel("number of events (weighted)")
    plt.yscale('log')    #linke Achse logarithmisch skalieren 

    #zweite Achse
    background_bins = np.sum(dy.values(),axis=0)
    signal_bins = hh.values()
    significance = signal_bins**2/background_bins
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

    plt.xlabel("Di-Higgs-outputnode of the DNN - corrected with logit function")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins - {channelname_r[i-1]}-channel")
    plt.savefig(f"plots/hist_hhnode_stacked-tt/logit_{channelname[i-1]}-channel.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    tt.reset()
    hh.reset()
