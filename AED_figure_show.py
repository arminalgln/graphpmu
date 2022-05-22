import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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


model_path = 'models/AED/806_824_836_846_with_complete_network_just_pmus_9features_flex'
model = torch.load(model_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

def show_detail(data, pmu, status):
  plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
  matplotlib.rcParams['axes.linewidth'] = 5
  from matplotlib.ticker import MaxNLocator
  plt.rcParams["font.weight"] = "bold"
  pad = 5
  xyticks_num = 10
  plt.style.use('default')
  matplotlib.rcParams['figure.figsize'] = 20, 12

  font_title = {'family': 'serif',
                'color': 'black',
                'weight': 'normal',
                'size': 24,
                }
  font_axis = {'family': 'serif',
               'color': 'black',
               'weight': 'normal',
               'size': 18,
               }
  fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, constrained_layout=True)
  lw=4
  # fig.xaxis.set_major_locator(MaxNLocator(integer=True))
  colors = ['r', 'k', 'b']
  base_voltage = 14.376
  base_current = 0.06956*1000
  a=1
  b=2
  c=1.05*base_voltage
  d=0
  ax0.plot((d+c*((data[3:, 0]-a)/b+a)),linewidth=lw, c='r')
  ax0.plot((d+c*((data[3:, 1]-a)/b+a)),linewidth=lw, c='k')
  ax0.plot((d+c*((data[3:, 2]-a)/b+a)),linewidth=lw, c='b')

  a = 0.14
  b = 3
  c = 0.45
  d = 0
  ax1.plot(np.flip(d+c*((data[3:, 3]-a)/b+a)),linewidth=lw, c='r')
  ax1.plot(np.flip(d+c*((data[3:, 4]-a)/b+a)),linewidth=lw, c='k')
  ax1.plot(np.flip(d+c*((data[3:, 5]-a)/b+a)),linewidth=lw, c='b')
  a = 0.7
  b = 5
  c = 1.3
  d = 0
  ax2.plot(np.flip(d+c*((data[3:, 6]-a)/b+a)),linewidth=lw, c='r')
  ax2.plot(np.flip(d+c*((data[3:, 7]-a)/b+a)),linewidth=lw, c='k')
  ax2.plot(np.flip(d+c*((data[3:, 8]-a)/b+a)),linewidth=lw, c='b')
  # for k in range(3):
  #   ax0.plot(data[3:, pmu*9 + k],linewidth=lw, c=colors[k])
  #   ax1.plot(data[3:, pmu*9 + k + 3],linewidth=lw,  c=colors[k])
  #   ax2.plot(data[3:, pmu*9 + k + 6],linewidth=lw,  c=colors[k])

  # ax0.set_xlabel('timesteps')
  ax0.set_ylabel(r'Voltage Magnitude $V_\phi^t$ (kV)', fontdict=font_axis)
  ax0.legend(['VA', 'VB', 'VC'],loc='lower left', fontsize=15)
  # ax0.set_title( '{} values of Event Timeseries'.format(status), fontdict=font_title)
  ax0.grid( linestyle='-', linewidth=1)
  ax0.tick_params(axis='both', which='major', labelsize=14)
  ax0.set_ylim([13.2, 14.6])
  ax0.set_yticks(13.20+np.arange(8)/5)
  ax0.set_xticks([0,20,40,60,80,100,120])
  ax0.set_xlim([0,120])
  blw=2
  for axis in ['top', 'bottom', 'left', 'right']:
    ax0.spines[axis].set_linewidth(blw)  # change width
    ax1.spines[axis].set_linewidth(blw)  # change width
    ax2.spines[axis].set_linewidth(blw)  # change width
  # ax0.set_yticks( fontdict=font_axis)


  # ax1.set_xlabel('timesteps')
  ax1.set_ylabel('Current Magnitude $I_\phi^t$ (kA)', fontdict=font_axis)
  ax1.legend(['IA', 'IB', 'IC'],loc='lower left', fontsize=15)
  ax1.grid( linestyle='-', linewidth=1)
  ax1.tick_params(axis='both', which='major', labelsize=14)
  ax1.set_xticks([0,20,40,60,80,100,120])
  ax1.set_xlim([0,120])
  ax1.set_ylim([0.04, 0.065])
  ax1.set_yticks(0.04+np.arange(6)/200)

  ax2.set_xlabel('Sample Number', fontdict=font_axis)
  ax2.set_ylabel('Power Factor (${pf}_\phi^t$)', fontdict=font_axis)
  ax2.legend(['PFA', 'PFB', 'PFC'],loc='upper left', fontsize=15)
  ax2.tick_params(axis='both', which='major', labelsize=14)
  # ax2.set_xticks([0,20,40,60,80,100,120])
  ax2.grid( linestyle='-', linewidth=1)
  ax2.set_xlim([0,120])
  ax2.set_ylim([0.775, 1])
  ax2.set_yticks((0.775+np.arange(10)/40))

  # plt.savefig('paper/figures/tsne_GraphPMU_12_pmus.eps', format='eps')

  return fig



# fig, ax = plt.subplots()
#
#
# ax.legend(loc='upper left', fontsize=15)
# plt.title('TSNE for the embeddings of DEC, 4 base PMUs', fontdict=font_title)
# plt.xlabel('Feature 1', fontdict=font_axis)
# plt.ylabel('Feature 2', fontdict=font_axis)
# plt.xlim([np.ceil(np.min(X_embedded[:, 0])-pad - 15),np.floor(np.max(X_embedded[:, 0]) + pad)])
# plt.ylim([np.min(X_embedded[:, 1])-pad,np.max(X_embedded[:, 1]) + pad + 20])
# plt.xticks(np.arange(np.ceil(np.min(X_embedded[:, 0])-pad - 10), np.ceil(np.max(X_embedded[:, 0]) + pad)
#                      , 20), fontsize=16)
#
# plt.yticks(np.arange(np.min(X_embedded[:, 1])-pad, np.max(X_embedded[:, 1]) + pad, 20), fontsize=16)
# plt.grid( linestyle='-', linewidth=1)
# plt.savefig('paper/figures/tsne_GraphPMU_12_pmus.eps', format='eps')
# plt.show()
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
#pmus={0:'806', 1:'824', 2:'836', 3:'846'}
ev = 10
pmu = 2


for ev in [10]:
  for pmu in [2]:
    # selected_data = per_unit[ev]
    selected_data = per_unit[pmu * ev_nums + ev]
    selected_data = torch.from_numpy(selected_data).to(device).reshape(1, selected_data.shape[0], selected_data.shape[1])
    pred = model(selected_data)
    def torch_to_numpy_cpu(data):
      return data.cpu()[0].detach().numpy()

    selected_data = torch_to_numpy_cpu(selected_data)
    pred = torch_to_numpy_cpu(pred)

    fig1 = show_detail(selected_data, 0, 'Real')
    # fig1.savefig('paper/figures/real_{}.pdf'.format(labels[ev]), dpi=300)
    plt.show()

    # pred = pred.reshape(1, pred.shape[0], pred.shape[1])
    fig2 = show_detail(pred, 0, 'Predicted')
    # fig2.savefig('paper/figures/predicted_{}.pdf'.format(labels[ev]), dpi=300)
    plt.show()
    print(labels[ev])
    # fig2 = show_detail(pred, pmu, 'pred')
    # plt.show()

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#subplot for the figure rather than each one individually
ev = 10
pmu = 2
selected_data = per_unit[pmu * ev_nums + ev]
selected_data = torch.from_numpy(selected_data).to(device).reshape(1, selected_data.shape[0], selected_data.shape[1])
pred = model(selected_data)
def torch_to_numpy_cpu(data):
  return data.cpu()[0].detach().numpy()

selected_data = torch_to_numpy_cpu(selected_data)
pred = torch_to_numpy_cpu(pred)

from matplotlib.ticker import FormatStrFormatter


fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(3, 1, wspace=0.6, hspace=0.38)
har = 1
color = ['r', 'k', 'b']
p = 1
for i in range(3):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=outer[i], wspace=0.4, hspace=0.1)

    colors = ['r', 'k', 'b']
    base_voltage = 14.376
    base_current = 0.06956 * 1000
    if 1== 0:
      a = 1
      b = 2
      c = 1.05 * base_voltage
      d = 0
    elif i==1:
      a = 0.14
      b = 3
      c = 0.45
      d = 0
    else:
      a = 0.7
      b = 5
      c = 1.3
      d = 0

    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        opt_mx = -1e7
        opt_mn = 1e7
        if j == 0:
          dt = selected_data
        else:
          dt = pred
        for phase in range(3):
            ax.plot((d + c * ((dt[3:, i*3 + phase] - a) / b + a)), c=color[phase])
            mx = max((d + c * ((dt[3:, i*3 + phase] - a) / b + a)))
            mn = min((d + c * ((dt[3:, i*3 + phase] - a) / b + a)))
            # mx += mx*(p/100)
            # mn -= mn*(p/100)
            if mx > opt_mx:
                opt_mx = mx
            if mn < opt_mn:
                opt_mn = mn
            # else:
            #     # data = np.abs(angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])*(np.pi/180)
            #     data = (angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])
            #     data = np.array([i if i > 0 else 180 + i for i in data])*(np.pi/180)
            #
            #     ax.plot(np.cos(data), c=color[phase])
            #     mx = max(np.cos(data))
            #     mn = min(np.cos(data))
            #
            #     # mx = max(np.cos((angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])))
            #     # mn = min(np.cos((angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])))
            #     # mx += mx*(p/100)
            #     # mn -= mn*(p/100)
            #     if mx > opt_mx:
            #         opt_mx = mx
            #     if mn < opt_mn:
            #         opt_mn = mn
            # else:
            #     ax.plot((angle[har][event][:,  j*3 + phase])*(np.pi/180), c=color[phase])
            #     mx = max(angle[har][event][:, j*3 + phase]*(np.pi/180))
            #     mn = min(angle[har][event][:, j*3 + phase]*(np.pi/180))
            #     # mx += mx*(p/100)
            #     # mn -= mn*(p/100)
            #     if mx > opt_mx:
            #         opt_mx = mx
            #     if mn < opt_mn:
            #         opt_mn = mn
            #
            # print(opt_mn, opt_mx)
            fig.add_subplot(ax)

        T=3
        # ax.set_ylim()
        # if i==0:
        v = 0.008
        opt_mx = opt_mx*(1+v)
        opt_mn = opt_mn*(1-v)

        yl = [opt_mn+(i)*(opt_mx-opt_mn)/T for i in range(T)]
        yl.append(opt_mx)
        # ax.set_yticks([opt_mn, (opt_mn*2+opt_mx)/3, (-opt_mn+2*opt_mx)/3, opt_mx])
        ax.set_yticks(yl)

        # else:
        #     v = 0.008
        #     opt_mx = opt_mx*(1+v)
        #     opt_mn = opt_mn*(1-v)
        #     yl = [opt_mn+(i)*(opt_mx-opt_mn)/T for i in range(T)]
        #     yl.append(opt_mx)
        #
        #     # ax.set_yticks([opt_mn, (opt_mn+opt_mx)/2, opt_mx])
        #     # ax.set_yticks([opt_mn, (opt_mn*2+opt_mx)/3, (-opt_mn+2*opt_mx)/3, opt_mx])
        #     ax.set_yticks(yl)


            # print(i,j,phase)
        # if i == 0:
        #     v = 0.008
        #     opt_mx = opt_mx * (1 + v)
        #     opt_mn = opt_mn * (1 - v)
        #     ax.set_yticks([opt_mn, (opt_mn+opt_mx)/3, 2*(opt_mn+opt_mx)/3, opt_mx])
        #     # ax.set_yticks([opt_mn, (opt_mn + opt_mx) / 2, opt_mx])
        #
        # else:
        #     # ax.set_yticks([opt_mn, (opt_mn + opt_mx) / 2, opt_mx])
        #     ax.set_yticks([opt_mn, (opt_mn+opt_mx)/3, 2*(opt_mn+opt_mx)/3, opt_mx])
        # plt.grid()
        if i == 0:
            ax.set_xlim([0, 120])

            if j == 0:
                plt.ylabel(r'$|V|$ (p.u)')
                ax.set_title(r'a) Actual data')
                                # ax.set_xlabel(r'a) Voltage magnitude of fundamental')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
            elif j == 1:
                ax.set_title('b) Predicted data')
                # ax.set_xlabel(r'a) Voltage magnitude, current magnitude and power factor of fundamental')

            else:
                ax.set_title(r'$pf$')
                # ax.set_xlabel(r'c) power factor of fundamental')

            # else:
            #     plt.ylabel(r'$|I|$')
            #     ax.set_xlim([0, 120])
            #     if har==1:
            #         ax.set_xlabel(r'a) Voltage and current magnitude of fundamental')
            #     elif har==3:
            #         ax.set_xlabel(r'c) Voltage and current magnitude of $3^{rd}$ harmonic')
            #     elif har==5:
            #         ax.set_xlabel(r'e) Voltage and current magnitude of $5^{th}$ harmonic')

                # ax.set_xticks(np.arange(0,120,10))

        if i == 1:
            ax.set_xlim([0, 120])

            if j == 0:
                plt.ylabel(r'$|I|$ (p.u)')
                # ax.set_xlabel(r'd) Voltage magnitude of $3^{rd}$ harmonic')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
                plt.legend(['A', 'B', 'C'])

            elif j == 1:
                # ax.set_title(r'$|I|$')
                # ax.set_xlabel(r'b) Voltage magnitude, current magnitude and power factor of $3^{rd}$ harmonic')
                pass
            else:
                # ax.set_title(r'$pf$')
                # ax.set_xlabel(r'f) power factor of $3^{rd}$ harmonic')
                pass

        if i == 2:
            if j == 0:
                plt.ylabel(r'pf')
                ax.set_xlim([0, 120])

                # ax.set_xlabel(r'g) Voltage magnitude of $3^{rd}$ harmonic')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
            elif j == 1:
                # ax.set_title(r'$|I|$')
                # ax.set_xlabel(r'c) Voltage magnitude, current magnitude and power factor of $5^{th}$ harmonic')
                ax.set_xlim([0, 120])


            else:
                # ax.set_title(r'$pf$')
                # ax.set_xlabel(r'f) power factor of $3^{rd}$ harmonic')
                ax.set_xlim([0, 120])

        # if j == 0:
            #     plt.ylabel(r'$\angle V$')
            #     ax.set_xlim([0, 120])
            #     ax.set_xticks([])
            #
            # else:
            #     plt.ylabel(r'$\angle I$')
            #     ax.set_xlim([0, 120])
            #     if har==1:
            #         ax.set_xlabel(r'b) Voltage and current angle of fundamental')
            #     elif har==3:
            #         ax.set_xlabel(r'd) Voltage and current angle of $3^{rd}$ harmonic')
            #     elif har==5:
            #         ax.set_xlabel(r'f) Voltage and current angle of $5^{th}$ harmonic')


    har += 2
# plt.title('Here')
plt.savefig('paper/figures/AEDout{}_pmu{}_withoutgrid'.format(ev, pmu), dpi=300)
fig.show()