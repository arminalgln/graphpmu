import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
import pscadimport
from pscadimport import PSCADVar
import pickle
#%%
# read the data


def read_data(path, file_name):


    csv_path = PSCADVar(path, file_name, del_out = False)
    data = pd.read_csv(csv_path)

    # just PMUs
    reg = re.compile("([a-zA-Z]+)([0-9]+)")
    given_keys = ['vpha', 'ipha', 'vphb', 'iphb', 'vphc', 'iphc', 'vma', 'ima', 'vmb', 'imb', 'vmc', 'imc']#for PMU data
    given_keys = ['va', 'ia', 'vb', 'ib', 'vc', 'ic', 'vam', 'vbm', 'vcm','vaph', 'vbph', 'vcph',
                  'iam', 'ibm', 'icm','iaph', 'ibph', 'icph']#for wave data
    main_keys = []
    buses = []
    for k in data.keys():
        mat = reg.match(k)
        if mat:
            if mat.group(1) in given_keys:
                main_keys.append(k)
                if not mat.group(2) in buses:
                    buses.append(mat.group(2))
    main_keys.append('time')
    data = data[main_keys]
    return data, buses
#%%

ev = 'onephase858'
path = r"models\pscadmodel\{}\{}.gf42".format(ev, ev)  # use r to avoid unicode problems
file_name = "{}_2".format(ev)

data, buses = read_data(path, file_name)

# capbank848 events
start_time = 1.5
end_time = 50
report_dist = data['time'][1] -data['time'][0]
starting_index = int(start_time/report_dist)
end_index = int(end_time/report_dist)

event_size = int(1/report_dist)
cap_on_events = {}
cap_off_events = {}
on_counter = 0
off_counter = 0
ON = True
while starting_index < end_index:
    if ON:
        cap_on_events[on_counter] = data.iloc[starting_index:starting_index + event_size]
        ON = False
        starting_index += event_size
        on_counter += 1
    else:
        cap_off_events[off_counter] = data.iloc[starting_index:starting_index + event_size]
        ON = True
        starting_index += event_size
        off_counter += 1

with open('data/events/{}/{}_on.p'.format(ev, file_name), 'wb') as fp:
    pickle.dump(cap_on_events, fp, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/events/{}/{}_off.p'.format(ev, file_name), 'wb') as fp:
    pickle.dump(cap_off_events, fp, protocol=pickle.HIGHEST_PROTOCOL)
#%%
ev = 'fault'
path = r"PSCAD_files\{}\{}.gf42".format(ev, ev)  # use r to avoid unicode problems
file_name = "{}_2".format(ev)

data, buses = read_data(path, file_name)

# faults
start_time = 0.5
end_time = 50
report_dist = data['time'][1] -data['time'][0]
starting_index = int(start_time/report_dist)
end_index = int(end_time/report_dist)

event_size = int(1/report_dist)
cap_on_events = {}
cap_off_events = {}
on_counter = 0
off_counter = 0
ON = True
while starting_index < end_index:
    if ON:
        cap_on_events[on_counter] = data.iloc[starting_index:starting_index + event_size]
        # ON = False
        starting_index += event_size
        on_counter += 1
    # else:
    #     cap_off_events[off_counter] = data.iloc[starting_index:starting_index + event_size]
    #     ON = True
    #     starting_index += event_size
    #     off_counter += 1
files = os.listdir('data/events')
if ev not in files:
    os.mkdir('data/events/{}'.format(ev))
with open('data/events/{}/{}.pkl'.format(ev, file_name), 'wb') as fp:
    pickle.dump(cap_on_events, fp, protocol=pickle.HIGHEST_PROTOCOL)

print(cap_on_events[0].keys())
print(cap_on_events[0].keys().shape)
#%%
event_files = os.listdir('PSCAD_files')
event_files.pop(int([i for i,c in enumerate(event_files) if c == 'all events.pswx'][0]))
for ev in event_files:
    path = r"PSCAD_files\{}\{}.gf42".format(ev, ev)  # use r to avoid unicode problems
    for sub_fil in [1, 2]:

        file_name = "{}_{}".format(ev, sub_fil)

        data, buses = read_data(path, file_name)

        # faults
        start_time = 0.5
        end_time = 50
        report_dist = data['time'][1] - data['time'][0]
        starting_index = int(start_time / report_dist)
        end_index = int(end_time / report_dist)

        event_size = int(1 / report_dist)
        cap_on_events = {}
        cap_off_events = {}
        on_counter = 0
        off_counter = 0
        ON = True
        while starting_index < end_index:
            if ON:
                cap_on_events[on_counter] = data.iloc[starting_index:starting_index + event_size]
                # ON = False
                starting_index += event_size
                on_counter += 1
            # else:
            #     cap_off_events[off_counter] = data.iloc[starting_index:starting_index + event_size]
            #     ON = True
            #     starting_index += event_size
            #     off_counter += 1
        files = os.listdir('data/events')
        if ev not in files:
            os.mkdir('data/events/{}'.format(ev))
        with open('data/events/{}/{}.pkl'.format(ev, file_name), 'wb') as fp:
            pickle.dump(cap_on_events, fp, protocol=pickle.HIGHEST_PROTOCOL)

        print('data/events/{}/{}.pkl'.format(ev, file_name))
        print(cap_on_events[0].keys().shape)











#%%



plt.plot(cap_on_events[1]['vam5_846'][0:1000])
plt.plot(cap_on_events[1]['vbm5_846'][0:1000])
plt.plot(cap_on_events[1]['vcm5_846'][0:1000])
plt.show()
#%%
plt.plot(data['va802'][1000:1500])
plt.plot(data['vb802'][1000:1500])
plt.plot(data['vc802'][1000:1500])
plt.show()
