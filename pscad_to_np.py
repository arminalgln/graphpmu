import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tslearn
import os
from scipy.interpolate import InterpolatedUnivariateSpline
#%%
all_pmus = pd.read_csv('data/buses.csv')
selected_pmus = ['806', '824', '854', '834', '836', '846']


def data_arrangement(path_to_events, selected_pmus):

    mva_base = 1 #mva
    kvll = 24.9
    base_voltage = (kvll) / np.sqrt(3) #kv
    base_current = mva_base/(base_voltage)
    base_phase = 90
    given_keys = ['vpha', 'vphb', 'vphc', 'ipha', 'iphb',  'iphc', 'vmc', 'vma', 'vmb', 'ima', 'imb', 'imc']
    desired_keys = []
    for pmu in selected_pmus: # all desired pmu data
        for k in given_keys:
            desired_keys.append(k+pmu)
    data = pd.read_pickle(path_to_events)
    data_arr = []
    per_unit_data_arr = []
    info_arr = {'event': path_to_events.split('/')[-1].split('\\')[0], 'pmus': selected_pmus}
    print(info_arr)

    for id in list(data.keys())[:-1]:
        print('here')
        temp_data = data[id][desired_keys].copy()
        sorted_cols = []
        per_unit_data = {}
        for pmu in selected_pmus:
            print(pmu)
            vm_keys = []
            im_keys = []
            t_keys = []
            for phase in ['a', 'b', 'c']: # making phase difference between voltage and current as a feature
                theta = temp_data['vph' + phase + pmu].values - temp_data['iph' + phase + pmu].values
                # desired_keys.append('t' + phase + pmu)
                temp_data['t' + phase + pmu] = theta

                vm_keys.append('vm' + phase + pmu)
                im_keys.append('im' + phase + pmu)
                t_keys.append('t' + phase + pmu)

                per_unit_data['vm' + phase + pmu] = temp_data['vm' + phase + pmu]/base_voltage
                per_unit_data['im' + phase + pmu] = temp_data['im' + phase + pmu]/base_current
                per_unit_data['t' + phase + pmu] = temp_data['t' + phase + pmu]/base_phase

                del temp_data['vph' + phase + pmu]
                del temp_data['iph' + phase + pmu]

            sorted_cols.append([vm_keys, im_keys, t_keys])
            print(np.ravel(sorted_cols))

        per_unit_data = pd.DataFrame(per_unit_data)
        per_unit_data_arr.append(per_unit_data[np.ravel(sorted_cols)].values)
        data_arr.append(temp_data[np.ravel(sorted_cols)].values)

    info_arr['features'] = np.ravel(sorted_cols)
    # data = to_time_series_dataset(data_arr)
    return np.stack(data_arr, axis=0), np.stack(per_unit_data_arr, axis=0), info_arr

#%%
# all_event_data = np.array([])
label = np.array([])
first = True
# all event ON gathering
for event in os.listdir('data/events'):
    path_to_events = 'data/events/' + event
    for filename in os.listdir(path_to_events):
        if 'on.p' in filename.split('_'):
            event_path = os.path.join(path_to_events, filename)
            data, per_unit, info = data_arrangement(event_path, selected_pmus)
            if first:
                all_event_data = data
                all_per_unit = per_unit
                first = False
            else:
                all_event_data = np.concatenate((all_event_data, data))
                all_per_unit = np.concatenate((all_per_unit, per_unit))
            label = np.concatenate((label, np.array([info['event']] * data.shape[0])))
label.shape
all_event_data.shape
print(info)
#%%
# np.save('data/features', info['features'])
np.save('data/labels' + ''.join(['_' + i for i in selected_pmus]), label)
np.save('data/all_event_data' + ''.join(['_' + i for i in selected_pmus]), all_event_data)
np.save('data/all_per_unit' + ''.join(['_' + i for i in selected_pmus]), all_per_unit)
#%%
#generate noise based on the maximum change in the time series

def get_noisy(ts):
    temp = np.roll(ts,-1)
    residue = np.abs(ts[0:-1]-temp[0:-1])
    eps = np.mean(residue)
    #np.random.seed(0)
    noise = np.random.normal(0, eps, ts.shape[0])
    ts_noisy = ts + noise
    return ts_noisy

#%%
#data augmentation
labels = np.load('data/labels' + ''.join(['_' + i for i in selected_pmus]) + '.npy')
per_unit = np.load('data/all_per_unit' + ''.join(['_' + i for i in selected_pmus]) + '.npy')
def augmentation(ev, label, shift, noise_number, order):

    # e = Event(ev, 0, -1, 'resampled')
    sample_horizon = np.arange(per_unit.shape[1])
    indexes = np.arange(ev.shape[0])
    ns = indexes.shape[0]
    augmented_causes = pd.DataFrame(columns=['label'])
    all_events = []
    all_labels = []
    for i, shift in enumerate(np.arange(-shift, shift)):
        for n in range(noise_number):
            aug_event = {}
            for f in range(ev.shape[-1]):
                shifted_temp_data = np.roll(ev[:, f], shift)[max(0, shift): max(ns, ns - shift)]
                intpo = InterpolatedUnivariateSpline(
                    indexes[max(0, shift): max(ns, ns - shift)], shifted_temp_data, k=order
                )
                new_data = intpo(indexes)
                aug_event[f] = get_noisy(new_data)
            aug_event = pd.DataFrame(aug_event)
            all_events.append(aug_event)
            all_labels.append(label)
            # new_id = cause['id'].values[0] + '_' + str(i) + '_' + str(n)
            # augmented_causes = augmented_causes.append({'label': label}, ignore_index=True)
            # saving_path = 'data/augmented_data/{}.pkl'.format(new_id)
            # aug_event.to_pickle(saving_path)
            # print('I saved {}'.format(new_id))
    all_events = np.array(all_events)
    all_labels = np.array(all_labels)
    return all_events, all_labels


#%%
#roll the time series
# labels = np.load('data/labels.npy')
# per_unit = np.load('data/all_per_unit.npy')
noise_number = 2
shift = 10
order = 2

for i, ev in enumerate(per_unit):
    agc, lbl = augmentation(ev, labels[i], shift, noise_number, order)
    per_unit = np.append(per_unit, agc, axis=0)
    labels = np.append(labels, lbl, axis=0)
    print(per_unit.shape, labels.shape)
# whole_agc.to_pickle('data/whole_agc.pkl')
#%%
np.save('data/aug_labels' + ''.join(['_' + i for i in selected_pmus]), labels)
np.save('data/aug_all_per_unit' + ''.join(['_' + i for i in selected_pmus]), per_unit)

#%%
np.save('data/features', info['features'])
np.save('data/labels', label)
np.save('data/all_event_data', all_event_data)
np.save('data/all_per_unit', all_per_unit)

#%%
