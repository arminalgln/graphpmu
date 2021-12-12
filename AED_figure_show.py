import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch

def show_detail(data, pmu, type):
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)
  for k in range(3):
    ax0.plot(data[5:, pmu*9 + k])
    ax1.plot(data[5:, pmu*9 + k + 3])
    ax2.plot(data[5:, pmu*9 + k + 6])

  ax0.set_xlabel('timesteps')
  ax0.set_ylabel('voltage magnitude')
  ax0.legend(['v1', 'v2', 'v3'])

  ax1.set_xlabel('timesteps')
  ax1.set_ylabel('current magnitude')
  ax1.legend(['i1', 'i2', 'i3'])

  ax2.set_xlabel('timesteps')
  ax2.set_ylabel('angle diff')
  ax2.legend(['t1', 't2', 't3'])

  fig.title = 'real'
  if type == 'pred':
    fig.title = 'pred'

  return fig
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#load the data
per_unit = np.load('data/whole_data_ss.pkl', allow_pickle=True)
labels = np.load('data/new_aug_labels_806_824_836_846.npy')
bus_data = pd.read_excel('data/ss.xlsx')
network = pd.read_excel('data/edges.xlsx')
# normalize data
new_data = []

for f in per_unit:
  if f == 806:
    concat_data = per_unit[f]
  elif f in [824, 836, 846]:
    concat_data = np.concatenate((concat_data, per_unit[f]))
print(concat_data.shape)
per_unit = concat_data
n_seq, seq_len, n_features = per_unit.shape
ev_nums = int(per_unit.shape[0]/4)
#%%
#0 capbank840
#50 capbank848
#100 faultAB862
#200 faultABC816
#250 'faultABC816'
#300 'faultC852'
#400 'loada836'
#450 'motormed812'
#500 'motorsmall828'
#550 'onephase858'
model_path = 'models/AED/806_824_836_846_with_complete_network_just_pmus_9features_flex'
model = torch.load(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
#pmus={0:'806', 1:'824', 2:'836', 3:'846'}
ev = 10
pmu = 0


for ev in [25]:
  for pmu in range(4):
    # selected_data = per_unit[ev]
    selected_data = per_unit[pmu * ev_nums + ev]
    selected_data = torch.from_numpy(selected_data).to(device).reshape(1, selected_data.shape[0], selected_data.shape[1])
    pred = model(selected_data)
    def torch_to_numpy_cpu(data):
      return data.cpu()[0].detach().numpy()

    selected_data = torch_to_numpy_cpu(selected_data)
    pred = torch_to_numpy_cpu(pred)

    fig1 = show_detail(selected_data, 0, 'real')
    plt.title(labels[ev])
    plt.show()

    # pred = pred.reshape(1, pred.shape[0], pred.shape[1])
    fig1 = show_detail(pred, 0, 'pred')  
    plt.title(labels[ev])
    plt.show()
    # fig2 = show_detail(pred, pmu, 'pred')
    # plt.show()
