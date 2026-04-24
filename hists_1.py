import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data

#Histogramme definieren, 2-D für dy wegen Unterteilung
dy = Hist(
    hist.axis.StrCategory([], name="Zerfallskanal", growth=True),  #diese Achse wird später gestacked
    hist.axis.Regular(bins=100, start=0, stop=1, name="x")
)
tt = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))
hh = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))
s = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))

#Namen der decay channel definieren:
channelname=["e-tau", "mu-tau", "tau-tau"]
channelname_r=[r"$\tau_e\tau_h$",r"$\tau_\mu\tau_h$",r"$\tau_h\tau_h$"]


#1. Histogramme nach channel aufteilen, fillen (für dy nach Zerfallskanal aufteilen + stacken), plotten.
for i in [1,2,3]:
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal=r"gen: DY $\to e^+e^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)],Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)])

    tt.fill(events_tt.run3_dnn_moe_hh[events_tt.channel_id == i],weight=events_tt.event_weight[events_tt.channel_id == i])
    hh.fill(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i],weight=events_hh.event_weight[events_hh.channel_id == i])

    #from IPython import embed; embed(header="MESSAGE Line 34 | File: hists_1.py")
    fig, ax1 = plt.subplots()
    background_bins = np.sum(dy.values(),axis=0)+ tt.values()
    signal_bins = hh.values()
    significance = signal_bins**2/background_bins
    ax2 = ax1.twinx()  # Erstellt die rechte Achse
    ax2.plot(np.linspace(0, 1, 100),significance, label="significance")
    ax2.set_ylabel('Exponentielle Werte')
    ax2.tick_params(axis='y', labelcolor='b')
    #plt.plot(significance)

    plt.yscale('log')    #Achse logarithmisch skalieren 

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal")
    stack.plot(stack=True, histtype="fill") # 'stack=True' ist entscheidend!

    tt.plot(label=r"$t\bar{t}$")
    hh.plot(label=r"$HH$")

    plt.legend()
    plt.ylabel("number of events (weighted)")
    plt.xlabel("Di-Higgs-outputnode of the DNN")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins - {channelname_r[i-1]}-channel")
    plt.savefig(f"plots/hist_hhnode/{channelname[i-1]}-channel.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    tt.reset()
    hh.reset()



#2. Normierte, lineare histogramme plotten (recht unwichtig)
for i in [1,2,3]:
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal=r"gen: DY $\to e^+e^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)],Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)])

    tt.fill(events_tt.run3_dnn_moe_hh[events_tt.channel_id == i],weight=events_tt.event_weight[events_tt.channel_id == i])
    hh.fill(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i],weight=events_hh.event_weight[events_hh.channel_id == i])


    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal")
    stack.plot(stack=True, histtype="fill", density=True) # 'stack=True' ist entscheidend!

    tt.plot(label=r"$t\bar{t}$", density=True)
    hh.plot(label=r"$HH$", density=True)

    plt.legend()
    plt.ylabel("number of events (normalized)")
    plt.xlabel("Di-Higgs-outputnode of the DNN")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins - {channelname_r[i-1]}-channel")
    plt.savefig(f"plots/hist_hhnode_lin/{channelname[i-1]}-channel_linear.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    tt.reset()
    hh.reset()



#3. wie 1, aber tt wird zum stack hinzugefügt
for i in [1,2,3]: #i steht für den channel
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal=r"gen: DY $\to e^+e^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal=r"gen: DY $\to \mu^+\mu^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)],Zerfallskanal=r"gen: DY $\to \tau^+\tau^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)])
    dy.fill(x=events_tt.run3_dnn_moe_hh[events_tt.channel_id == i],Zerfallskanal=r"$t\bar{t}$", weight=events_tt.event_weight[events_tt.channel_id == i])

    hh.fill(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i],weight=events_hh.event_weight[events_hh.channel_id == i])

    plt.yscale('log')    #Achse logarithmisch skalieren 

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal") #technically zerfallskanal+tt als korrekter name
    stack.plot(stack=True, histtype="fill") # 'stack=True' ist entscheidend!

    hh.plot(label=r"$HH$")

    plt.legend()
    plt.ylabel("number of events (weighted)")
    plt.xlabel("Di-Higgs-outputnode of the DNN")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins - {channelname_r[i-1]}-channel")
    plt.savefig(f"plots/hist_hhnode_stacked-tt/{channelname[i-1]}-channel.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    tt.reset()
    hh.reset()
