import numpy as np
import pandas as pd
#%%

# make the complete graph for IEEE 34 bus distribution test system
bus_data = pd.read_excel('data/ss.xlsx')
network = pd.read_excel('data/edges.xlsx')
# print(bus_data.head())
selected_pmus = [806, 824, 836, 846]
for i in selected_pmus:
    if i in bus_data['bus'].values:
        index = bus_data.index[bus_data['bus'] == i][0]
        print(i, 'the id is: ', bus_data['id'][index])

# get the data for selected pmus and fill the data for the rest
per_unit = np.load('data/aug_all_per_unit_806_824_836_846.npy')
labels = np.load('data/aug_labels_806_824_836_846.npy')

# with pmu data [806, 824, 836, 846]
known_data = np.array_split(per_unit, 4, axis=-1)
# initial values for not pmu buses
event_numbers, timesteps, _ = per_unit.shape
feature_num = 9

whole_data = {}
    # all 1
temp_value = np.ones((event_numbers, timesteps, feature_num))
count = 0
for bus in bus_data['bus']:
    if bus in selected_pmus:
        whole_data[bus] = known_data[count]
        count += 1
    else:
        whole_data[bus] = temp_value

