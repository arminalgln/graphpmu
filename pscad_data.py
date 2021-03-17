import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%%
# data = pd.read_csv('pscadmodel/IEEE34_no_wind_main.gf42/capbank3to5.txt')
# from ImPSCAD import PSCADVar

path = r"pscadmodel\IEEE34_no_wind_main.gf42"  # use r to avoid unicode problems
file_name = "capbank3to5"

csv_path = PSCADVar(path, file_name, del_out = False)
#%%
variables = pd.read_csv(csv_path)
#%%
variables
#%%
plt.plot(variables['time'],variables['Pa1_x.1'])
plt.show()
#%%