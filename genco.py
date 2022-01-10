# with latent data
import numpy as np
import pandas as pd
import pickle
import torch

#%%

with open("data/positive_graphs_timeseries.pkl", "rb") as handle:
    pos_graphs = pickle.load(handle)

with open("data/negative_graphs_timeseries.pkl", "rb") as handle:
    neg_graphs = pickle.load(handle)
#%%
for g in pos_graphs:
    mn = g.ndata["features"].mean(axis=1)
    maxmin = (
        g.ndata["features"].max(axis=1).values - g.ndata["features"].min(axis=1).values
    )
    gdata = torch.cat((maxmin, mn), axis=1)
    g.ndata["latent"] = gdata
#%%
for g in neg_graphs:
    mn = g.ndata["features"].mean(axis=1)
    maxmin = (
        g.ndata["features"].max(axis=1).values - g.ndata["features"].min(axis=1).values
    )
    gdata = torch.cat((maxmin, mn), axis=1)
    g.ndata["latent"] = gdata
#%%
pmus = [2, 8, 19, 23]
selected_latents = []
for g in pos_graphs:
    selected_latents.append(torch.ravel(g.ndata["latent"][pmus]).detach().cpu().numpy())
selected_latents = np.array(selected_latents)
#%%
bus_data = pd.read_excel("data/ss.xlsx")
network = pd.read_excel("data/edges.xlsx")
# print(bus_data.head())
selected_pmus = [806, 824, 836, 846]
for i in selected_pmus:
    if i in bus_data["bus"].values:
        index = bus_data.index[bus_data["bus"] == i][0]
        print(i, "the id is: ", bus_data["id"][index])

# get the data for selected pmus and fill the data for the rest
per_unit = np.load("data/new_aug_all_per_unit_806_824_836_846.npy")
labels = np.load("data/new_aug_labels_806_824_836_846.npy")

# with pmu data [806, 824, 836, 846]
known_data = np.array_split(per_unit, 4, axis=-1)
# initial values for not pmu buses
event_numbers, timesteps, _ = per_unit.shape
feature_num = 9
mva_base = 1  # mva
kvll = 24.9
base_voltage = (kvll) / np.sqrt(3)  # kv
base_current = mva_base / (base_voltage)

whole_data_without_ss = {}
whole_data_ss = {}
# all 1
temp_value = np.ones((event_numbers, timesteps, feature_num))
count = 0
features = bus_data.keys()[2:]
#%%
for bus in bus_data["bus"]:
    if bus in selected_pmus:
        whole_data_without_ss[bus] = known_data[count]
        whole_data_ss[bus] = known_data[count]
        count += 1
    else:
        whole_data_without_ss[bus] = temp_value

        ss_data_temp = np.ones((event_numbers, timesteps, feature_num))
        for i, f in enumerate(features):
            b_index = bus_data.index[bus_data["bus"] == bus][0]
            if i in range(0, 3):  # per unit voltage
                ss_data_temp[:, :, i] = (
                    ss_data_temp[:, :, i] * bus_data[f][b_index] / base_voltage
                )
            elif i in range(3, 6):  # per unit current
                ss_data_temp[:, :, i] = (
                    ss_data_temp[:, :, i] * bus_data[f][b_index] / base_current
                )
            elif i in range(6, 9):  # per unit current
                ss_data_temp[:, :, i] = np.cos(
                    ss_data_temp[:, :, i] * bus_data[f][b_index] * np.pi / 180
                )

        whole_data_ss[bus] = ss_data_temp

#%%
with open("data/whole_data_ss.pkl", "wb") as pkl_handle:
    pickle.dump(whole_data_ss, pkl_handle)

with open("data/whole_data_without_ss.pkl", "wb") as pkl_handle:
    pickle.dump(whole_data_without_ss, pkl_handle)
