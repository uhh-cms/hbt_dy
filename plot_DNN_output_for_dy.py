import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

print("hello")

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data

hh = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))
dy = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))
tt = Hist(hist.axis.Regular(bins=100, start=0, stop=1, name="x"))


hh.fill(events_dy.run3_dnn_moe_hh)
dy.fill(events_dy.run3_dnn_moe_dy)
tt.fill(events_dy.run3_dnn_moe_tt)

hh.plot(label="hh")
dy.plot(label="dy")
tt.plot(label="tt")

plt.legend()
plt.title("Outputnodes des DNN für DY simulationen")
plt.savefig("plots/hist_hh_dy_tt.png", dpi=300, bbox_inches='tight')
