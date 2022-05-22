import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#revive the data
phasor_data = pd.read_pickle('data/wave/actual_events_harmonics_1_3_5.pickle')
features = pd.read_pickle('data/wave/features.pickle')
pmus = features['pmus']
features = features['features']
magnitude = phasor_data['mag']
angle = phasor_data['angle']

event_numbers, window, features_num = magnitude[1].shape
pmu_name = '824'# '836'
pmu_index = pmus.index(pmu_name)
selected_features = [i for i in features if pmu_name in i]
selected_features_index = np.arange(pmu_index*6, (pmu_index+1)*6)
event = 400 #10, 15, 450
for k in magnitude:
    a = magnitude[k][:, 116:]
    b = np.copy(magnitude[k])
    magnitude[k] = np.concatenate((b, a), axis=1)
    magnitude[k] = magnitude[k][:, :, selected_features_index]

    a = angle[k][:, 116:]
    b = np.copy(angle[k])
    angle[k] = np.concatenate((b, a), axis=1)
    angle[k] = angle[k][:, :, selected_features_index]



from matplotlib.ticker import FormatStrFormatter


fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(3, 2, wspace=0.4, hspace=0.38)
har = 1
color = ['r', 'k', 'b']
p = 1
for i in range(6):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        opt_mx = -1e7
        opt_mn = 1e7
        for phase in range(3):
            if i%2 == 0:
                ax.plot(magnitude[har][event][:, j*3 + phase], c=color[phase])
                mx = max(magnitude[har][event][:, j*3 + phase])
                mn = min(magnitude[har][event][:, j*3 + phase])
                # mx += mx*(p/100)
                # mn -= mn*(p/100)
                if mx > opt_mx:
                    opt_mx = mx
                if mn < opt_mn:
                    opt_mn = mn
            else:
                ax.plot((angle[har][event][:,  j*3 + phase])*(np.pi/180), c=color[phase])
                mx = max(angle[har][event][:, j*3 + phase]*(np.pi/180))
                mn = min(angle[har][event][:, j*3 + phase]*(np.pi/180))
                # mx += mx*(p/100)
                # mn -= mn*(p/100)
                if mx > opt_mx:
                    opt_mx = mx
                if mn < opt_mn:
                    opt_mn = mn

            # print(opt_mn, opt_mx)


            # ax.set_ylim()
            ax.set_yticks([opt_mn, (opt_mn+opt_mx)/2, opt_mx])

            fig.add_subplot(ax)
            # plt.grid()
        if i % 2 == 0:
            if j == 0:
                plt.ylabel(r'$|V|$')
                ax.set_xlim([0, 120])
                ax.set_xticks([])

            else:
                plt.ylabel(r'$|I|$')
                ax.set_xlim([0, 120])
                if har==1:
                    ax.set_xlabel(r'a) Voltage and current magnitude of fundamental')
                elif har==3:
                    ax.set_xlabel(r'c) Voltage and current magnitude of $3^{rd}$ harmonic')
                elif har==5:
                    ax.set_xlabel(r'e) Voltage and current magnitude of $5^{th}$ harmonic')

                # ax.set_xticks(np.arange(0,120,10))

        if i % 2 == 1:
            if j == 0:
                plt.ylabel(r'$\angle V$')
                ax.set_xlim([0, 120])
                ax.set_xticks([])

            else:
                plt.ylabel(r'$\angle I$')
                ax.set_xlim([0, 120])
                if har==1:
                    ax.set_xlabel(r'b) Voltage and current angle of fundamental')
                elif har==3:
                    ax.set_xlabel(r'd) Voltage and current angle of $3^{rd}$ harmonic')
                elif har==5:
                    ax.set_xlabel(r'f) Voltage and current angle of $5^{th}$ harmonic')


    if i%2 == 1:
        har += 2
# plt.title('Here')
# plt.savefig('paper/figures/harmonics/3by2_event{}_pmu{}'.format(event, pmu_name), dpi=300)

fig.show()


#another version with pf instead of just angles



# fig, ax = plt.subplots(3, 1, constrained_layout=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter


fig = plt.figure(figsize=(10, 8))
outer = gridspec.GridSpec(3, 1, wspace=0.6, hspace=0.38)
har = 1
color = ['r', 'k', 'b']
p = 1
for i in range(3):
    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[i], wspace=0.4, hspace=0.1)

    for j in range(3):
        ax = plt.Subplot(fig, inner[j])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        opt_mx = -1e7
        opt_mn = 1e7
        for phase in range(3):
            if j in [0, 1]:
                ax.plot(magnitude[har][event][:, j*3 + phase], c=color[phase])
                mx = max(magnitude[har][event][:, j*3 + phase])
                mn = min(magnitude[har][event][:, j*3 + phase])
                # mx += mx*(p/100)
                # mn -= mn*(p/100)
                if mx > opt_mx:
                    opt_mx = mx
                if mn < opt_mn:
                    opt_mn = mn
            else:
                # data = np.abs(angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])*(np.pi/180)
                data = (angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])
                data = np.array([i if i > 0 else 180 + i for i in data])*(np.pi/180)

                ax.plot(np.cos(data), c=color[phase])
                mx = max(np.cos(data))
                mn = min(np.cos(data))

                # mx = max(np.cos((angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])))
                # mn = min(np.cos((angle[har][event][:, j - 2 + phase] - angle[har][event][:, j + 1 + phase])))
                # mx += mx*(p/100)
                # mn -= mn*(p/100)
                if mx > opt_mx:
                    opt_mx = mx
                if mn < opt_mn:
                    opt_mn = mn
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


            # ax.set_ylim()
            ax.set_yticks([opt_mn, (opt_mn+opt_mx)/2, opt_mx])

            fig.add_subplot(ax)
            # print(i,j,phase)
            # plt.grid()
        if i == 0:
            ax.set_xlim([0, 120])

            if j == 0:
                plt.ylabel(r'Fundamental')
                ax.set_title(r'$|V|$ (kV)')
                plt.legend(['A', 'B', 'C'])
                # ax.set_xlabel(r'a) Voltage magnitude of fundamental')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
            elif j == 1:
                ax.set_title(r'$|I|$ (kA)')
                ax.set_xlabel(r'a) Voltage magnitude, current magnitude and power factor of fundamental')

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
                plt.ylabel(r'$3^{rd}$ harmonic')
                # ax.set_xlabel(r'd) Voltage magnitude of $3^{rd}$ harmonic')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
            elif j == 1:
                # ax.set_title(r'$|I|$')
                ax.set_xlabel(r'b) Voltage magnitude, current magnitude and power factor of $3^{rd}$ harmonic')

            else:
                # ax.set_title(r'$pf$')
                # ax.set_xlabel(r'f) power factor of $3^{rd}$ harmonic')
                pass

        if i == 2:
            if j == 0:
                plt.ylabel(r'$5^{th}$ harmonic')
                ax.set_xlim([0, 120])

                # ax.set_xlabel(r'g) Voltage magnitude of $3^{rd}$ harmonic')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
            elif j == 1:
                # ax.set_title(r'$|I|$')
                ax.set_xlabel(r'c) Voltage magnitude, current magnitude and power factor of $5^{th}$ harmonic')
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
plt.savefig('paper/figures/harmonics/3by3_event{}_pmu{}'.format(event, pmu_name), dpi=300)

fig.show()

#%%%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

#revive the data
phasor_data = pd.read_pickle('data/wave/actual_events_harmonics_1_3_5.pickle')
features = pd.read_pickle('data/wave/features.pickle')
pmus = features['pmus']
features = features['features']
magnitude = phasor_data['mag']
angle = phasor_data['angle']

event_numbers, window, features_num = magnitude[1].shape
pmu_name = '824'# '836'
pmu_index = pmus.index(pmu_name)
selected_features = [i for i in features if pmu_name in i]
selected_features_index = np.arange(pmu_index*6, (pmu_index+1)*6)
event = 400 #10, 15, 450
for k in magnitude:
    a = magnitude[k][:, 116:]
    b = np.copy(magnitude[k])
    magnitude[k] = np.concatenate((b, a), axis=1)
    magnitude[k] = magnitude[k][:, :, selected_features_index]

    a = angle[k][:, 116:]
    b = np.copy(angle[k])
    angle[k] = np.concatenate((b, a), axis=1)
    angle[k] = angle[k][:, :, selected_features_index]

#%%
mva_base = 1 #mva
kvll = 23.9
base_voltage = (kvll) / np.sqrt(3) #kv
base_current = mva_base/(base_voltage)

for i in magnitude:
    for j in range(6):
        if j<3:
            magnitude[i][:, :, j] =  magnitude[i][:, :, j]/base_voltage
        else:
            magnitude[i][:, :, j] = magnitude[i][:, :, j] /base_current

#%%


from matplotlib.ticker import FormatStrFormatter
import matplotlib
fig = plt.figure(figsize=(14, 8))
outer = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.18)#rows
har = 1
color = ['r', 'k', 'b']
p = 1
# plt.rcParams.update({'font.size': 14})
font = {'size'   : 13}

matplotlib.rc('font', **font)
for i in range(2):#rows
    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[i], wspace=0.3, hspace=0.1)#cols

    for j in range(3):#cols
        ax = plt.Subplot(fig, inner[j])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        opt_mx = -1e7
        opt_mn = 1e7
        for phase in range(3):#phases

            ax.plot(magnitude[2*j+1][event][:, i*3 + phase], c=color[phase])
            mx = max(magnitude[2*j+1][event][:, i*3 + phase])
            mn = min(magnitude[2*j+1][event][:, i*3 + phase])
            # mx += mx*(p/100)
            # mn -= mn*(p/100)
            if mx > opt_mx:
                opt_mx = mx
            if mn < opt_mn:
                opt_mn = mn

            # ax.set_yticks([opt_mn, (opt_mn+opt_mx)/2, opt_mx])

            fig.add_subplot(ax)
            ax.set_xticks([0, 30, 60, 90, 120])
        T=6
        # ax.set_ylim()
        # if i==0:
        v = 0.008
        opt_mx = opt_mx*(1+v)
        opt_mn = opt_mn*(1-v)

        yl = [opt_mn+(i)*(opt_mx-opt_mn)/T for i in range(T)]
        yl.append(opt_mx)
        # ax.set_yticks([opt_mn, (opt_mn*2+opt_mx)/3, (-opt_mn+2*opt_mx)/3, opt_mx])
        ax.set_yticks(yl)
        plt.grid()
        if i == 0:
            ax.set_xlim([0, 120])

            if j == 0:
                plt.ylabel(r'$|V|$ (p.u.)', fontsize=18)
                ax.set_title(r'a) Fundamental', fontsize=18)
                plt.legend(['A', 'B', 'C'], fontsize=14)
                # ax.set_xlabel(r'a) Voltage magnitude of fundamental')
                # ax.set_xlim([0, 120])
                # ax.set_xticks([])
            elif j == 1:
                ax.set_title(r'3rd Harmonic', fontsize=18)
                # ax.set_xlabel(r'a) Voltage magnitude, current magnitude and power factor of fundamental')

            else:
                ax.set_title(r'5th Harmonic', fontsize=18)
                # ax.set_xlabel(r'c) power factor of fundamental')
        else:
            plt.xlabel(r'Sample Number', fontsize=17)


        if i == 1:
            if j == 0:
                plt.ylabel(r'$|I|$ (p.u.)', fontsize=18)
        ax.set_xlim([0, 120])



plt.savefig('paper/figures/harmonics/2by3_event{}_pmu{}'.format(event, pmu_name), dpi=300)

fig.show()
