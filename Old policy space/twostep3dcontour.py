#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 09:28:36 2018

@author: peter
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PARAMETERS TO CREATE POLICY SPACE
LOWER_CRATE_LIM = 3.6 # C rate, lower cutoff
UPPER_CRATE_LIM = 6   # C rate, upper cutoff
LOWER_SOC1_LIM  = 0  # [%], lower SOC1 cutoff
UPPER_SOC1_LIM  = 80  # [%], upper SOC1 cutoff
DENSITY         = 8   # Points per line cut
STEP            = 0.2 # initial distance from baseline policy
chargetime      = 10  # [=] minutes
FINAL_CUTOFF    = 80  # SOC cutoff
margin = 0.8 # plotting margin

# Calculate Q1(CC1, CC2) values for contour lines
CC1 = np.linspace(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,100)
CC2 = np.linspace(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,100)
[X,Y] = np.meshgrid(CC1,CC2)
Q1 = (100)*(chargetime - ((60*(FINAL_CUTOFF/100))/Y))/((60/X)-(60/Y))
Q1[np.where(Q1<0)]  = float('NaN')
Q1[np.where(Q1>80)] = float('NaN')
    
## Create contour plot
## Initialize plot 1: color = SOC1
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.set_cmap('viridis')

minn, maxx = 0, 80
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='winter')
m.set_array([])
fcolor = m.to_rgba(Q1)

C = ax.plot_surface(X,Y,Q1, facecolors=fcolor)
plt.clabel(C, fontsize=10,fmt='%1.0f')
ax.set_xlabel('CC1',fontsize=16)
ax.set_ylabel('CC2',fontsize=16)
ax.set_zlabel('Q1',fontsize=16)
plt.axis('square')
plt.xlim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
plt.ylim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
plt.colorbar(m)
ax.view_init(elev=45, azim=225)

# Make full screen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
