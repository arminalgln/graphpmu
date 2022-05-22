import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%%
accuracy = {'kshape_Tslearn': 0.2371, 'kmeans_Tslearn': 0.3435, 'kernel_Tslearn': 0.4181,
            'GMM_AED': 0.4735, 'GMM_DEC_AED': 0.5248}

pmus_ordered = [806, 824, 836, 846, 800, 814,  834, 854, 832, 848, 808, 840]

number_if_pmus = [4,6,8,10]
methods = ['kernel_Tslearn', 'kmeans_Tslearn', 'kshape_Tslearn', 'AED', 'DEC',
           'GraphPMU(ss)', 'GraphPMU(random)', 'GraphPMU(fixed)', 'GraphPMU(just global)', 'Just PMU buses graph']
Total_accuracy = {'AED':[0.4735,0.5089, 0.546, 0.568, 0.573],
    'DEC':[0.5248,0.678, 0.67903,0.7091,0.721],
    'kernel_Tslearn': [0.2371,0.422,0.419,0.399,0.408], 'kshape_Tslearn':[0.4181,0.622,0.634,0.631,0.637],
    'kmeans_Tslearn':[0.3435,0.4413,0.456,0.5221,0.5532],
    'TS + N/G + NL':[0.487,0.532,0.581,0.589,0.603],
    'AED + N/G': [0.423, 0.472, 0.552, 0.665, 0.725],
    'AED + N/G + RL':[0.533,0.6505,0.6720,0.6892,0.698],
    'AED + G + NL':[0.585, 0.6812, 0.7191, 0.759, 0.764],
    'AED + N/G + NL': [0.7205, 0.7732, 0.7889, 0.812, 0.819]

}

Total_accuracy = {'AED':[0.4735,0.5089, 0.546, 0.568],
    'DEC':[0.5248,0.678, 0.67903,0.7091],
    'kernel_Tslearn': [0.2371,0.422,0.419,0.399], 'kshape_Tslearn':[0.4181,0.622,0.634,0.631],
    'kmeans_Tslearn':[0.3435,0.4413,0.456,0.5221],
    'TS + N/G + NL':[0.487,0.532,0.581,0.589],
    'AED+N/G': [0.423, 0.472, 0.552, 0.665],
    'AED + N/G + RL':[0.533,0.6505,0.6720,0.6892],
    'AED + G + NL':[0.585, 0.6812, 0.7191, 0.759],
    'GraphPMU': [0.7205, 0.7732, 0.7889, 0.812]

}
#it was right infront of the        'Just PMU buses graph':[0.423,0.472,0.552,0.665,0.725]       ,  0.4879

# markers = {
#         'kernel_Tslearn': "o", 'kmeans_Tslearn':"v", 'kshape_Tslearn':"^", 'AED': "<",
#        'DEC': ">", 'GraphPMU(ss)': "s", 'GraphPMU(random)':"P",'GraphPMU(fixed)':"d", 'GraphPMU(just global with ss)':"*",
#        'Just PMU buses graph':"X"
# }

markers = {'AED': "<",
       'DEC': "s",
        'kernel_Tslearn': "o", 'kshape_Tslearn':"^",'kmeans_Tslearn':"v",
        'TS + N/G + NL': ">", 'AED+N/G':"P",'AED + N/G + RL':"d", 'AED + G + NL':"*",
       'GraphPMU':"X"
}
methods = list(markers.keys())
cs = ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',
            'tab:gray','tab:olive','tab:cyan']
cs = ['tab:blue', 'tab:orange','tab:red','tab:gray']
colors = {}

#%%
import matplotlib.lines as mlines
ax = plt.gca()
leg = ax.get_legend()

plt.rcParams["font.weight"] = "normal"
font_title = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }
font_axis = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
matplotlib.rcParams['figure.figsize'] = 20, 12
count=0
for mdl in Total_accuracy:
    if mdl in ['AED','DEC', 'AED+N/G', 'GraphPMU']:
        ax.scatter(number_if_pmus, Total_accuracy[mdl], marker=markers[mdl], c=cs[count] ,s=600)
        ax.plot(number_if_pmus, Total_accuracy[mdl], c=cs[count], linewidth=3.0)
        count += 1
# plt.legend(methods, loc='lower right', fontsize=15)
plt.xlabel('Number of PMUs', fontdict=font_axis)

plt.ylabel('ARI Score', fontdict=font_axis)
plt.xlim([3.5, 10.5])
plt.ylim([0.4, 0.85])
plt.yticks([0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85], fontsize=16)
plt.xticks([4, 6, 8, 10], fontsize=16)


lgs = []
count = 0
for c,i in enumerate(methods):
    if i in ['AED', 'DEC', 'AED+N/G', 'GraphPMU']:
        lgs.append(mlines.Line2D([], [], color=cs[count], marker=markers[i], linestyle='None',
                          markersize=22, label=i))
        count += 1


plt.legend(handles=lgs,loc='upper left', fontsize=22)

# plt.title('ARI score for all models with different number of available micro-PMUs', fontdict=font_title)
plt.grid()
plt.savefig('paper/figures/ARI_all_methods_10.pdf', dpi=300)
plt.show()
#

#%%


#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

number_if_pmus = [4,6,8,10]
methods = ['kernel_Tslearn', 'kmeans_Tslearn', 'kshape_Tslearn', 'AED', 'DEC',
           'GraphPMU(ss)', 'GraphPMU(random)', 'GraphPMU(fixed)', 'GraphPMU(just global)', 'Just PMU buses graph']
Total_accuracy = {'AED':[0.4735,0.5089, 0.546, 0.568],
    'DEC':[0.5248,0.678, 0.67903,0.7091],
    'AED + N/G': [0.423, 0.472, 0.552, 0.665],
    'AED + N/G + NL': [0.7205, 0.7732, 0.7889, 0.812]

}

Total_accuracy = {
    # 'AED':[0.4735,0.5089, 0.546, 0.568],
    # 'DEC':[0.5248,0.678, 0.67903,0.7091],
    # 'AED+N/G': [0.423, 0.472, 0.552, 0.665],
    # 'AED+N/G+NL': [0.7205, 0.7732, 0.7889, 0.812],
    'AED+harmonic':[0.6666, 0.69510, 0.7346, 0.7368],
    'DEC+harmonic':[0.6948, 0.752, 0.76903, 0.7791],
    'AED+N/G+harmonic': [0.5656, 0.612, 0.792, 0.8765],
    'GraphPMU+harmonic': [0.814, 0.8732, 0.9209, 0.9317]
}


# markers = {'AED': "<",
#        'DEC': "s",
#         'kernel_Tslearn': "o", 'kshape_Tslearn':"^",'kmeans_Tslearn':"v",
#         'TS + N/G + NL': ">", 'AED + N/G':"P",'AED + N/G + RL':"d", 'AED + G + NL':"*",
#        'AED + N/G + NL':"X"
# }

# markers = {'AED': "<",
#        'DEC': "s",
#         'AED+N/G': "o", 'AED+N/G+NL':"^",'AED+harmonic':"v",
#         'DEC+harmonic': ">", 'AED+N/G+harmonic':"P",'GraphPMU':"d"
# }

markers = {'AED+harmonic':"<",
        'DEC+harmonic': "s", 'AED+N/G+harmonic':"o",'GraphPMU+harmonic':"X"
}
methods = list(markers.keys())
cs = ['tab:blue', 'tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',
            'tab:gray']
cs = ['tab:blue', 'tab:orange','tab:red','tab:gray']
colors = {}


import matplotlib.lines as mlines
ax = plt.gca()
leg = ax.get_legend()

plt.rcParams["font.weight"] = "normal"
font_title = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 24,
        }
font_axis = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 22,
        }
matplotlib.rcParams['figure.figsize'] = 20, 12
count=0
for mdl in Total_accuracy:
    ax.scatter(number_if_pmus, Total_accuracy[mdl], marker=markers[mdl], c=cs[count] ,s=600)
    ax.plot(number_if_pmus, Total_accuracy[mdl], c=cs[count], linewidth=3.0)
    count += 1
# plt.legend(methods, loc='lower right', fontsize=15)
plt.xlabel('Number of PMUs', fontdict=font_axis)

plt.ylabel('ARI Score', fontdict=font_axis)
plt.xlim([3.5, 10.5])
plt.ylim([0.55, 0.95])
plt.yticks([0.55,0.6,0.65,0.7,0.75,0.8,0.85, 0.9, 0.95], fontsize=16)
plt.xticks([4, 6, 8, 10], fontsize=16)


lgs = []
count = 0
for c,i in enumerate(methods):
    lgs.append(mlines.Line2D([], [], color=cs[count], marker=markers[i], linestyle='None',
                      markersize=22, label=i))
    count += 1


plt.legend(handles=lgs,loc='upper left', fontsize=22)

# plt.title('ARI score for all models with different number of available micro-PMUs', fontdict=font_title)
plt.grid()
plt.savefig('paper/figures/ARI_all_methods_harm.pdf', dpi=300)
plt.show()

