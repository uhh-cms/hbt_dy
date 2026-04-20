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

#Namen der decay channel definieren:
channelname=["e-tau", "mu-tau", "tau-tau"]
channelname_r=[r"$\tau_e\tau_h$",r"$\tau_\mu\tau_h$",r"$\tau_h\tau_h$"]


#array mit anzahl der el bzw pos pro zerfall (Zahl zwischen 0 und 2)
maske_el = (events_dy.gen_dy_tau_decayproducts == 11)
zahlen_array_el = maske_el*1
maske_pos = (events_dy.gen_dy_tau_decayproducts == -11)
zahlen_array_pos = maske_pos*1

result_el = np.sum(zahlen_array_el,axis=2)
result_el = np.sum(result_el,axis=1)
result_pos = np.sum(zahlen_array_pos,axis=2)
result_pos = np.sum(result_pos,axis=1)
el_number = result_el + result_pos
el_number = el_number[(events_dy.gen_ll_pdgid == 15)]   #maske für ausschließlich tau tau zerfälle

#array mit anzahl der el bzw pos pro zerfall (Zahl zwischen 0 und 2)
maske_mu = (events_dy.gen_dy_tau_decayproducts == 13)
zahlen_array_mu = maske_mu*1
maske_antimu = (events_dy.gen_dy_tau_decayproducts == -13)
zahlen_array_antimu = maske_antimu*1

result_mu = np.sum(zahlen_array_mu,axis=2)
result_mu = np.sum(result_mu,axis=1)
result_antimu = np.sum(zahlen_array_antimu,axis=2)
result_antimu = np.sum(result_antimu,axis=1)
mu_number = result_mu + result_antimu
mu_number = mu_number[(events_dy.gen_ll_pdgid == 15)]

#für die Anzahl hadronischer taus pro event erst die anderen zerfälle (oben) von np.twos abziehen.
#Aber nicht vergessen, die tautau-maske noch drüberzusetzen, weil sonst alle anderen dy-subprozesse auch eine 2 zugewiesen bekommen.
hadron_number=np.ones([np.sum(events_dy.gen_ll_pdgid == 15)]) *2 - el_number - mu_number

#Zwischenergebnis (3dim. array, dass jedem event #el,#mu,#tau zerfälle zuordnet)
tau_zerfallskanäle=np.array([el_number,mu_number,hadron_number])



#6.1 stacked hist weiter in subprozesse unterteilen (tau tau weiter unterteilen wie bei 5. im bar chart)
for i in [1,2,3]: #i steht für den channel
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)],Zerfallskanal=r"$e^+e^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 11)])    #maske für channel (und bei dy Zerfallskanal) in eckigen Klammern
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)],Zerfallskanal=r"$\mu^+\mu^-$", weight=events_dy.event_weight[(events_dy.channel_id == i) & (events_dy.gen_ll_pdgid == 13)])


    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==2) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==0)],Zerfallskanal=r"$\tau_e\tau_e$", weight=events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==2) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==0)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==2) & (tau_zerfallskanäle[2]==0)],Zerfallskanal=r"$\tau_\mu\tau_\mu$", weight=events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==2) & (tau_zerfallskanäle[2]==0)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==2)],Zerfallskanal=r"$\tau_h\tau_h$", weight=events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==2)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==1) & (tau_zerfallskanäle[1]==1) & (tau_zerfallskanäle[2]==0)],Zerfallskanal=r"$\tau_e\tau_\mu$", weight=events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==1) & (tau_zerfallskanäle[1]==1) & (tau_zerfallskanäle[2]==0)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==1) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==1)],Zerfallskanal=r"$\tau_e\tau_h$", weight=events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==1) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==1)])
    dy.fill(x=events_dy.run3_dnn_moe_hh[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==1) & (tau_zerfallskanäle[2]==1)],Zerfallskanal=r"$\tau_\mu\tau_h$", weight=events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == i) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==1) & (tau_zerfallskanäle[2]==1)])


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
    plt.savefig(f"plots/hist_hhnode_stacked-tt/further_subdivision/{channelname[i-1]}-channel.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    tt.reset()
    hh.reset()


#7.2 hists für category_ids statt channels
#masken erstellen
IDs=["etau__res1b__os__iso","etau__res2b__os__iso","mutau__res1b__os__iso","mutau__res2b__os__iso","tautau__res1b__os__iso","tautau__res2b__os__iso"]
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

    plt.yscale('log')    #Achse logarithmisch skalieren 

    # Stack-Plot erstellen
    stack = dy.stack("Zerfallskanal") #technically zerfallskanal+tt als korrekter name
    stack.plot(stack=True, histtype="fill") # 'stack=True' ist entscheidend!

    hh.plot(label=r"$HH$")

    plt.legend()
    plt.ylabel("number of events (weighted)")
    plt.xlabel("Di-Higgs-outputnode of the DNN")
    plt.title(f"Histogram of DNN-outputnode $HH$ for dy,tt and hh simulatioins -{IDs[i]}- cat_id:{id}")
    plt.savefig(f"plots/hist_hhnode_stacked-tt/channel_unterteilung/{id}-cat_id.png", dpi=300, bbox_inches='tight')
    plt.figure()

    #histogramme für nächste iteration clearen
    dy.reset()
    hh.reset()

    #noch name im titel anzeigen lassen