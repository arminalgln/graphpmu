import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%%
accuracy = {'kshape_Tslearn': 0.2371, 'kmeans_Tslearn': 0.3435, 'kernel_Tslearn': 0.4181,
            'GMM_AED': 0.4735, 'GMM_DEC_AED': 0.5248}

pmus_ordered = [806, 824, 836, 846, 834, 854, 800, 814, 832, 848, 808, 840]

number_if_pmus = [4,6,8,10,12]
methods = ['kernel_Tslearn', 'kmeans_Tslearn', 'kshape_Tslearn', 'AED', 'DEC',
           'GraphPMU(ss)', 'GraphPMU(random)', 'GraphPMU(fixed)', 'GraphPMU(just global)', 'Just PMU buses graph']
Total_accuracy = {
    'kernel_Tslearn': [0.2371,0.422,0.419,0.399,0.408], 'kmeans_Tslearn':[0.3435,0.4413,0.456,0.5221,0.5632],
    'kshape_Tslearn':[0.4181,0.622,0.634,0.631,0.637], 'AED':[0.4735,0.5089, 0.546, 0.568, 0.573],
    'DEC':[0.5248,0.678, 0.67903,0.7091,0.721],'GraphPMU(ss)':[0.7205, 0.7732, 0.7889, 0.812, 0.819],
    'GraphPMU(random)':[0.533,0.6505,0.6720,0.6892,0.698],
    'GraphPMU(fixed)':[0.492,0.532,0.581,0.589,0.603],
    'GraphPMU(just global with ss)':[0.585, 0.6812, 0.7191, 0.759, 0.764],
    'Just PMU buses graph':[0.423,0.472,0.552,0.665,0.725] 0.4879
}

markers = {
        'kernel_Tslearn': "o", 'kmeans_Tslearn':"v", 'kshape_Tslearn':"^", 'AED': "<",
       'DEC': ">", 'GraphPMU(ss)': "s", 'GraphPMU(random)':"P",'GraphPMU(fixed)':"d", 'GraphPMU(just global with ss)':"*",
       'Just PMU buses graph':"X"
}
#%%
plt.rcParams["font.weight"] = "bold"
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
for mdl in Total_accuracy:
    plt.scatter(number_if_pmus, Total_accuracy[mdl], marker=markers[mdl], s=300)
    plt.plot(number_if_pmus, Total_accuracy[mdl], linewidth=5.0)
plt.legend(methods, loc='lower right', fontsize=15)
plt.xlabel('Number of PMUs', fontdict=font_axis)
plt.ylabel('ARI score', fontdict=font_axis)
plt.xlim([3.5, 12.5])
plt.ylim([0.1, 0.85])
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.85], fontsize=16)
plt.xticks([4, 6, 8, 10, 12], fontsize=16)
# plt.title('ARI score for all models with different number of available micro-PMUs', fontdict=font_title)
plt.grid()
plt.savefig('paper/figures/ARI_all_methods.pdf', dpi=300)
plt.show()

#%%
