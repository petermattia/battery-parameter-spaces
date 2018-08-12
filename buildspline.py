#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 10:22:45 2018

@author: peter
"""

import numpy as np
from scipy.interpolate import Rbf
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

plt.close('all')

##############################################################################
# PARAMETERS TO CREATE POLICY SPACE
C1list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6, 8]
C3list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]

C4_LIMITS = [0.1, 4.81] # Lower and upper limits specifying valid C4s
FILENAME = 'real'

##############################################################################

one_step = 4.8
margin = 0.2 # plotting margin

colormap = 'plasma_r'

# IMPORT RESULTS
# Get folder path containing text files
cwd = os.getcwd()
data = pickle.load(open(cwd + '/mid_oed_tests/data_1to4.pkl', 'rb'))
minn = min_lifetime = np.min(data[:,4])
maxx = max_lifetime = np.max(data[:,4])

# CREATE RBF
hp = 0.03
rbf = Rbf(data[:,0], data[:,1], data[:,2], data[:,4],function='thin_plate',smooth=hp)

## INITIALIZE CONTOUR PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
plt.set_cmap(colormap)
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()

# Calculate C4(CC1, CC2) values for contour lines
C1_grid = np.arange(min(C1list)-margin,max(C1list) + margin,0.01)
C2_grid = np.arange(min(C1list)-margin,max(C1list) + margin,0.01)
[X,Y] = np.meshgrid(C1_grid,C2_grid)

## MAKE SUBPLOTS
for k, c3 in enumerate(C3list):
    plt.subplot(2,3,k+1)
    plt.axis('square')

# ADD COLORBAR
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(top=0.93)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])
cbar = fig.colorbar(m, cax=cbar_ax)
cbar.ax.set_title('Cycle life')

plt.suptitle('smooth = ' + str(hp))

for k, c3 in enumerate(C3list):
    plt.subplot(2,3,k+1)
    plt.axis('square')
    plt.cla()
    
    ## PLOT SURFACE
    ti = np.linspace(3.6-margin, 8.0+margin, 100)
    XI, YI = np.meshgrid(ti, ti)
    C3I = c3*np.ones((len(XI),len(YI)))
    lifetimeI = rbf(XI, YI, C3I)
    plt.pcolor(XI, YI, lifetimeI,vmin=minn,vmax=maxx)

    ## PLOT CONTOURS
    C4 = 0.2/(1/6 - (0.2/X + 0.2/Y + 0.2/c3))
    C4[np.where(C4<C4_LIMITS[0])] = float('NaN')
    C4[np.where(C4>C4_LIMITS[1])] = float('NaN')
    levels = np.arange(2.5,4.8,0.25)
    C = plt.contour(X,Y,C4,levels,zorder=1,colors='k')
    plt.clabel(C,fmt='%1.1f')
    
    ## PLOT POLICIES
    idx_subset = np.where(data[:,2]==c3)
    policy_subset = data[idx_subset,0:4][0]
    lifetime_subset = data[idx_subset,4]
    plt.scatter(policy_subset[:,0],policy_subset[:,1],vmin=minn,vmax=maxx,
                c=lifetime_subset.ravel(),zorder=2,s=100)
    
    ## BASELINE
    if c3 == one_step:
        plt.scatter(one_step,one_step,c='k',marker='s',zorder=3,s=100)
        
    ## ADD LABELS
    plt.title('C3=' + str(c3) + ': ' + str(len(policy_subset)) + ' policies',fontsize=16)
    plt.xlabel('C1')
    plt.ylabel('C2')
    plt.xlim((min(C1list)-margin, max(C1list)+margin))
    plt.ylim((min(C1list)-margin, max(C1list)+margin))
    
plt.savefig(cwd+'/mid_oed_tests/smooth_'+str(hp)+'.png', bbox_inches='tight')