import numpy as np
#%%
#cap bank
#Given the default values of C = 3.76 uF and L = 0.227 mH,
#this part of the bank represents a 25 MVAr leg.
#Note:
#25 MVAr = 230 kV / sqrt(3)^2 / Xc
#Xc = 705.333 ohm
#Therefore,
#C = 1 / w*Xc = 3.76 uF
f = 60
w = 2 * np.pi * f
v1ph = 14.376 #kV
# each leg of cap bank 100 kvar
q = 0.1 #Mvar
Xc = (v1ph**2)/q
C = 1/(w * Xc)
Cm = C * 10**6
#%%