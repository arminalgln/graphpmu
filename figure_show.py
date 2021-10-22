import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def show_detail(data, pmu, type):
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)
  for k in range(3):
    ax0.plot(data[5:,pmu * k])
    ax1.plot(data[5:,pmu * k + 3])
    ax2.plot(data[5:,pmu * k + 6])

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
per_unit = np.load('data/new_aug_all_per_unit_806_824_836_846.npy')
labels = np.load('data/new_aug_labels_806_824_836_846.npy')
#%%
pmu = 1

for ev in range(3):
# ev = 100
  data = per_unit[ev]
  # data = torch.from_numpy(data).to(device).reshape(1, data.shape[0], data.shape[1])
  # pred = model(data)
  def torch_to_numpy_cpu(data):
    return data.cpu()[0].detach().numpy()

  # data = torch_to_numpy_cpu(data)
  # pred = torch_to_numpy_cpu(pred)

  fig1 = show_detail(data, pmu, 'real')
  plt.show()
  # fig2 = show_detail(pred, pmu, 'pred')
  # plt.show()
