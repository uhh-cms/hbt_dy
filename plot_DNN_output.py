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


#Histogramme nach channel aufteilen, fillen (für dy nach Zerfallskanal aufteilen + stacken), plotten.
for i in [1,2,3]:
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal="e", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal="mu", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)],Zerfallskanal="tau", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 15)])

    tt.fill(events_tt.run3_dnn_moe_hh[events_tt.channel_id == i],weight=events_tt.event_weight[events_tt.channel_id == i])
    hh.fill(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i],weight=events_hh.event_weight[events_hh.channel_id == i])

    plt.yscale('log')    #Achse logarithmisch skalieren 

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal")
    stack.plot(stack=True, histtype="fill") # 'stack=True' ist entscheidend!

    #dy.plot(label="dy")
    tt.plot(label="tt")
    hh.plot(label="hh")

    plt.legend()
    plt.ylabel("number of events (weighted)")
    plt.title(f"DNN outputnode hh for dy,tt and hh simulatioins - channel{i}")
    plt.savefig(f"plots/hist_hhnode_channel{i}.png", dpi=300, bbox_inches='tight')
    plt.figure()



    #gen_ll_pdgid unterteilt die dy events in die tatsächlichen zerfallskanäle(?)
    #11=e 13=mu 15=tau
    #stack
    #dy in ee mu,mu und tt unterteilen