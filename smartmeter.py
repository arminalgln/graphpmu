import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

data = pd.read_excel('data/sm/elec_consumption_15.xlsx')

dt = data.loc[(data['local_15min']>'2013-06-21') & (data['local_15min']<'2013-07-22')]
#%%
s = dt['local_15min'].shape[0]
n = 10
tks = [dt['local_15min'].iloc[int(np.round(i*s/n))] for i in range(n)]
tks.append(dt['local_15min'].iloc[-1])
# plt.figure(figsize=(14,7))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 10}
# matplotlib.rc('font', **font)
plt.gcf().subplots_adjust(bottom=0.25)
plt.plot(dt['local_15min'], dt['use'])
plt.plot(evs['local_15min'], evs['use'])
plt.xticks(tks, rotation=90,fontsize=10)
plt.xlabel('Days',fontsize=12, weight='bold')
plt.ylim([0, dt['use'].max()*1.05])
plt.ylabel('Power Consumption (kW)', fontsize=12, weight='bold')
# plt.plot(data['local_15min'][0:2000], data['use'][0:2000])
plt.savefig('data/sm/onemonth.png',dpi=300)
plt.show()

#%%
start = '2013-06-29'
end = '2013-07-02'
event = dt.loc[(data['local_15min']>start) & (data['local_15min']<end)]
plt.plot(event['local_15min'], event['use'])
# plt.plot(data['local_15min'][0:2000], data['use'][0:2000])
plt.show()


s = event['local_15min'].shape[0]
n = 2
tks = [event['local_15min'].iloc[int(np.round(i*s/n))] for i in range(n)]
tks.append(event['local_15min'].iloc[-1])
# plt.figure(figsize=(14,7))
# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 10}
# matplotlib.rc('font', **font)
# plt.gcf().subplots_adjust(bottom=0.25)
evs = event.iloc[35:155]
plt.plot(event['local_15min'], event['use'])
plt.plot(evs['local_15min'],evs['use'])
plt.xticks(tks, rotation=0,fontsize=10)
plt.xlabel('Days',fontsize=12, weight='bold')
plt.ylim([0, event['use'].max()*1.05])
plt.ylabel('Power Consumption (kW)', fontsize=12, weight='bold')
# plt.plot(data['local_15min'][0:2000], data['use'][0:2000])
plt.savefig('data/sm/event.png', dpi=300)

plt.show()
#%%%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os


data = pd.read_pickle('data/pmu/events_data.pkl')
data = data['July_03'][351]



fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6,6), constrained_layout=True)
plt.subplots_adjust( wspace=1, hspace=1)
for k in range(3):
    ax0.plot(data[k]*10)
    ax1.plot(data[k + 3])
    ax2.plot((data[k + 6]))

# ax0.set_xlabel('timesteps')
ax0.set_ylabel('Voltage Magnitude (kV)')
ax0.legend(['vA', 'vB', 'vC'])
# ax1.set_xlabel('timesteps')
ax1.set_ylabel('Current Magnitude (kA)')
ax1.legend(['iA', 'iB', 'iC'])

ax2.set_xlabel('Timesteps')
ax2.set_ylabel('Power Factor')
ax2.legend(['pfA', 'pfB', 'pfC'])

plt.show()

#%%
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

data = pd.read_pickle('data/pmu/rawdata3.pkl')
v1 = data['1224']['L1MAG']
v2 = data['1224']['L2MAG']
v3 = data['1224']['L3MAG']



start = 350*20-10
end = 353*20
t = np.arange(start,end)
plt.plot(t,v1[start:end])
plt.plot(t,v2[start:end])
plt.plot(t,v3[start:end])
plt.xlim([start,end])
plt.ylim([7220,7280])
plt.xticks(np.arange(start,end+1,10))
plt.yticks(np.arange(7220,7280+1,10))
plt.xlabel('Sample Number',fontsize=12, weight='bold')
plt.ylabel('Voltage Magnitude (V)', fontsize=12, weight='bold')
plt.savefig('data/pmu/eventpmu.png', dpi=300)
plt.show()
#%%

fig, ax = plt.subplots()
end = 30000
t = np.arange(0,end)
ax.plot(v1[0:end])
ax.plot(v2[0:end])
ax.plot(v3[0:end])
plt.xlim([0,end])
plt.ylim([7220,7330])
plt.xticks(np.arange(0,end+1,5000))
plt.yticks(np.arange(7220,7331,20))
plt.xlabel('Sample Number',fontsize=12, weight='bold')
plt.ylabel('Voltage Magnitude (V)', fontsize=12, weight='bold')

ellipse = mpatches.Ellipse((7000, 7254), 1075, 57,angle=0, facecolor='y', alpha=0.4, lw=1,edgecolor='y')
ax.add_patch(ellipse)
plt.savefig('data/pmu/pmudata.png', dpi=300)
# plt.xlim([0,end])

# start = 350*20
# end = 353*20
# plt.plot(t,v1[start:end], color='r')
# plt.plot(t,v2[start:end], color='r')
# plt.plot(t,v3[start:end], color='r')
fig.show()

