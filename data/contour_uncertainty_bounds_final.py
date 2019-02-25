#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

NOTE: REQUIRES IMAGEMAGICK. https://www.imagemagick.org/script/download.php
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import pickle

plt.close('all')

##############################################################################
# PARAMETERS TO CREATE POLICY SPACE
min_policy_bound, max_policy_bound = 3.6, 8
C3list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]

C4_LIMITS = [0.1, 4.81] # Lower and upper limits specifying valid C4s
##############################################################################

one_step = 4.8
margin = 0.2 # plotting margin

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colormap = 'plasma_r'

# IMPORT RESULTS
# Get folder path containing text files
file_list = sorted(glob.glob('./bounds/4_bounds.pkl'))
data = []
min_bounds = 10000
max_bounds = -1
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        bounds = ub - lb
        data.append(mean)
        min_bounds = min(np.min(bounds),min_bounds)
        max_bounds = max(np.max(bounds),max_bounds)

# Find number of batches
batchnum = len(data)

## INITIALIZE CONTOUR PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
plt.rcParams.update({'font.size': 16})
plt.set_cmap(colormap)
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()
minn, maxx = min_bounds, max_bounds

# Calculate C4(CC1, CC2) values for contour lines
C1_grid = np.arange(min_policy_bound-margin,max_policy_bound + margin,0.01)
C2_grid = np.arange(min_policy_bound-margin,max_policy_bound + margin,0.01)
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
cbar.ax.set_title('Bounds on cycle life')

k2 = 0

# FUNCTION FOR LOOPING THROUGH BATCHES
for k, c3 in enumerate(C3list):
    plt.subplot(2,3,k+1)
    plt.axis('square')
    plt.cla()

    ## PLOT CONTOURS
    C4 = 0.2/(1/6 - (0.2/X + 0.2/Y + 0.2/c3))
    C4[np.where(C4<C4_LIMITS[0])]  = float('NaN')
    C4[np.where(C4>C4_LIMITS[1])] = float('NaN')

    levels = np.arange(2.5,4.8,0.25)
    C = plt.contour(X,Y,C4,levels,zorder=1,colors='k')
    plt.clabel(C,fmt='%1.1f')

    ## PLOT POLICIES
    idx_subset = np.where(param_space[:,2]==c3)
    policy_subset = param_space[idx_subset]
    bounds_subset = bounds[idx_subset]
    plt.scatter(policy_subset[:,0],policy_subset[:,1],vmin=minn,vmax=maxx,
                c=bounds_subset.ravel(),zorder=2,s=100)

    ## BASELINE
    if c3 == one_step:
        plt.scatter(one_step,one_step,c='k',marker='s',zorder=3,s=100)

    plt.title('CC3=' + str(c3) + ': ' + str(len(policy_subset)) + ' policies',fontsize=16)
    plt.xlabel('CC1')
    plt.ylabel('CC2')
    plt.xlim((min_policy_bound-margin, max_policy_bound+margin))
    plt.ylim((min_policy_bound-margin, max_policy_bound+margin))

## SAVE Plot

plt.savefig('plots/final_bounds_uncertainty.png', bbox_inches = 'tight')
plt.savefig('plots/final_bounds_uncertainty.pdf', bbox_inches = 'tight')