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


dy.fill(events_dy.run3_dnn_moe_hh,weight=events_dy.event_weight)
tt.fill(events_tt.run3_dnn_moe_hh,weight=events_tt.event_weight)
hh.fill(events_hh.run3_dnn_moe_hh,weight=events_hh.event_weight)

plt.yscale('log')    #Achse logarithmisch skalieren 

dy.plot(label="dy")
tt.plot(label="tt")
hh.plot(label="hh")

plt.legend()
plt.ylabel("number of events (weighted)")
plt.title("DNN outputnode hh for dy,tt and hh simulatioins")
plt.savefig("plots/hist_hhnode.png", dpi=300, bbox_inches='tight')