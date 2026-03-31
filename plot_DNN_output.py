import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data


dy = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))
tt = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))
hh = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))


#masken für channel-ID erstellen:
id_1dy = (events_dy.channel_id == 1)
id_2dy = (events_dy.channel_id == 2)
id_3dy = (events_dy.channel_id == 3)

id_1tt = (events_tt.channel_id == 1)
id_2tt = (events_tt.channel_id == 2)
id_3tt = (events_tt.channel_id == 3)

id_1hh = (events_hh.channel_id == 1)
id_2hh = (events_hh.channel_id == 2)
id_3hh = (events_hh.channel_id == 3)


for i in [1,2,3]:
    dy.fill(events_dy.run3_dnn_moe_hh[events_dy.channel_id == i],weight=events_dy.event_weight[events_dy.channel_id == i])
    tt.fill(events_tt.run3_dnn_moe_hh[events_tt.channel_id == i],weight=events_tt.event_weight[events_tt.channel_id == i])
    hh.fill(events_hh.run3_dnn_moe_hh[events_hh.channel_id == i],weight=events_hh.event_weight[events_hh.channel_id == i])

    plt.yscale('log')    #Achse logarithmisch skalieren 

    dy.plot(label="dy")
    tt.plot(label="tt")
    hh.plot(label="hh")

    plt.legend()
    plt.ylabel("number of events (weighted)")
    plt.title(f"DNN outputnode hh for dy,tt and hh simulatioins - channel{i}")
    plt.savefig(f"plots/hist_hhnode_channel{i}.png", dpi=300, bbox_inches='tight')
    plt.figure()
