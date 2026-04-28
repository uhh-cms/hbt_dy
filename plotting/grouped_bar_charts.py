import awkward as ak
import hist
from hist import Hist
import numpy as np
import matplotlib.pyplot as plt

events_dy = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/dy_22pre_v14.parquet")  # dy simulation data
events_tt = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/tt_22pre_v14.parquet")  # tt simulation data
events_hh = ak.from_parquet("/data/dust/user/wolfmor/hh2bbtautau/vincent/hh_22pre_v14.parquet")  # hh simulation data


#4.1 grouped bar chart for seeing which DY subprocesses go to which channel
categories = [r"gen: DY $\to e^+e^-$", r"gen: DY $\to \mu^+\mu^-$", r"gen: DY $\to \tau^+\tau^-$"]
channel_1 = np.array([]) #channel 1 werte (jeweils linker balken)
channel_2 = np.array([])
channel_3 = np.array([])

dictionary={"channel_1":channel_1,"channel_2":channel_2,"channel_3":channel_3}

#Channel arrays mit information füllen
for channel in [1,2,3]:
    for Zerfallskanal in [11,13,15]:
        dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.channel_id == channel) & (events_dy.gen_ll_pdgid == Zerfallskanal)]))

x = np.arange(len(categories))  # Positionen der Gruppen    
width = 0.2  # Breite der einzelnen Balken

#plotten
plt.bar(x - width, dictionary["channel_1"], width, label=r'$\tau_e\tau_h$ (chan. 1)')
plt.bar(x, dictionary["channel_2"], width, label=r'$\tau_\mu\tau_h$ (chan. 2)')
plt.bar(x + width, dictionary["channel_3"], width, label=r'$\tau_h\tau_h$ (chan. 3)') 
plt.xticks(x, categories)
plt.legend()
plt.ylabel("Events assigned to channel")
plt.xlabel("Drell-Yan-Subprocess")
plt.title("Channel assignment of the different DY-subprocesses")
plt.savefig("plots/grouped_bar-chart_DY-subprocesses/DY_channel-assignent.png", dpi=300, bbox_inches='tight')
plt.figure()

#nochmal logarithmisch plotten
plt.bar(x - width, dictionary["channel_1"], width, label=r'$\tau_e\tau_h$ (chan. 1)')
plt.bar(x, dictionary["channel_2"], width, label=r'$\tau_\mu\tau_h$ (chan. 2)')
plt.bar(x + width, dictionary["channel_3"], width, label=r'$\tau_h\tau_h$ (chan. 3)') 
plt.yscale('log')    #Achse logarithmisch skalieren
plt.xticks(x, categories)
plt.legend()
plt.ylabel("Events assigned to channel")
plt.xlabel("Drell-Yan-Subprocess")
plt.title("Channel assignment of the different DY-subprocesses")
plt.savefig("plots/grouped_bar-chart_DY-subprocesses/DY_channel-assignent_log.png", dpi=300, bbox_inches='tight')
plt.figure()



#5.2 grouped bar chart for seeing which DY subprocesses go to which channel, tau-tau-channel weiter unterteilt in Zerfallsart
#oben definierte tau_zerfallskanäle nutzen

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

categories = [r"$e^+e^-$", r"$\mu^+\mu^-$", r"$\tau_e\tau_e$", r"$\tau_\mu\tau_\mu$", r"$\tau_h\tau_h$", r"$\tau_e\tau_\mu$", r"$\tau_e\tau_h$", r"$\tau_\mu\tau_h$"]
channel_1 = np.array([]) #channel 1 werte (jeweils linker balken)
channel_2 = np.array([])
channel_3 = np.array([])

dictionary={"channel_1":channel_1,"channel_2":channel_2,"channel_3":channel_3}

#Channel arrays mit information füllen
for channel in [1,2,3]:
    for Zerfallskanal in [11,13]:
        dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.channel_id == channel) & (events_dy.gen_ll_pdgid == Zerfallskanal)]))
    #Channel arrays für Zerfallskanal 15 auch:
    dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == channel) & (tau_zerfallskanäle[0]==2) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==0)])) #achtung, verschachtelte Masken. manche masken redundant
    dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == channel) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==2) & (tau_zerfallskanäle[2]==0)]))
    dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == channel) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==2)]))
    dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == channel) & (tau_zerfallskanäle[0]==1) & (tau_zerfallskanäle[1]==1) & (tau_zerfallskanäle[2]==0)]))
    dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == channel) & (tau_zerfallskanäle[0]==1) & (tau_zerfallskanäle[1]==0) & (tau_zerfallskanäle[2]==1)]))
    dictionary[f"channel_{channel}"]=np.append(dictionary[f"channel_{channel}"],np.sum(events_dy.event_weight[(events_dy.gen_ll_pdgid == 15)][(events_dy.channel_id[events_dy.gen_ll_pdgid == 15] == channel) & (tau_zerfallskanäle[0]==0) & (tau_zerfallskanäle[1]==1) & (tau_zerfallskanäle[2]==1)]))
#Reihenfolge: el, mu, hadr
#Reihenfolge categories: e,mu,hadr,e+mu,e+hadr,mu+hadr

x = np.arange(len(categories))  # Positionen der Gruppen    
width = 0.2  # Breite der einzelnen Balken

#plotten
plt.figure(figsize=(15, 6))
plt.bar(x - width, dictionary["channel_1"], width, label=r'$\tau_e\tau_h$')
plt.bar(x, dictionary["channel_2"], width, label=r'$\tau_\mu\tau_h$')
plt.bar(x + width, dictionary["channel_3"], width, label=r'$\tau_h\tau_h$') 
plt.xticks(x, categories)
plt.legend()
plt.ylabel("Events assigned to channel")
plt.xlabel("gen: DY-subprocesses")
plt.figtext(0.1, 0.01,r"Different gen: DY decay-channels and their assignment to higgs decay-channels. The $gen: DY\to \tau\tau$-channel is further segmented into the $\tau$ decay-modes")
plt.title("Channel assignment of the different DY-subprocesses")
plt.savefig("plots/grouped_bar-chart_DY-subprocesses/further_disection/DY_channel-assignent.png", dpi=300, bbox_inches='tight')
plt.figure()

#nochmal logarithmisch plotten
plt.figure(figsize=(15, 6))
plt.bar(x - width, dictionary["channel_1"], width, label=r'$\tau_e\tau_h$ (chan. 1)')
plt.bar(x, dictionary["channel_2"], width, label=r'$\tau_\mu\tau_h$ (chan. 2)')
plt.bar(x + width, dictionary["channel_3"], width, label=r'$\tau_h\tau_h$ (chan. 3)') 
plt.yscale('log')    #Achse logarithmisch skalieren
plt.xticks(x, categories)
plt.legend()
plt.ylabel("Events assigned to channel")
plt.xlabel("gen: DY-subprocesses")
plt.figtext(0.1, 0.01,r"Different gen: DY decay-channels and their assignment to higgs decay-channels. The $gen: DY\to \tau\tau$-channel is further segmented into the $\tau$ decay-modes")
plt.title("Channel assignment of the different DY-subprocesses")
plt.savefig("plots/grouped_bar-chart_DY-subprocesses/further_disection/DY_channel-assignent_log.png", dpi=300, bbox_inches='tight')
plt.figure()