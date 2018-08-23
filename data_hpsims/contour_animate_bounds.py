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
from matplotlib import animation
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

colormap = 'plasma_r'

# IMPORT RESULTS
# Get folder path containing text files
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
data = []
min_lifetime = 10000
max_lifetime = -1
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        data.append(mean)
        min_lifetime = min(np.min(mean),min_lifetime)
        max_lifetime = max(np.max(mean),max_lifetime)

# Find number of batches
batchnum = len(data)

## INITIALIZE CONTOUR PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
plt.set_cmap(colormap)
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()
minn, maxx = min_lifetime, max_lifetime

# Calculate C4(CC1, CC2) values for contour lines
C1_grid = np.arange(min_policy_bound-margin,max_policy_bound + margin,0.01)
C2_grid = np.arange(min_policy_bound-margin,max_policy_bound + margin,0.01)
[X,Y] = np.meshgrid(C1_grid,C2_grid)

## MAKE SUBPLOTS
for k, c3 in enumerate(C3list):
    plt.subplot(2,3,k+1)
    plt.axis('square')

plt.suptitle('Batch ' + str(1))

# ADD COLORBAR
fig.subplots_adjust(right=0.8)
fig.subplots_adjust(top=0.93)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])
cbar = fig.colorbar(m, cax=cbar_ax)
cbar.ax.set_title('Cycle life')

# FUNCTION FOR LOOPING THROUGH BATCHES
def make_frame(k2):
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
        lifetime_subset = data[k2][idx_subset]
        plt.scatter(policy_subset[:,0],policy_subset[:,1],vmin=minn,vmax=maxx,
                    c=lifetime_subset.ravel(),zorder=2,s=100)

        ## BASELINE
        if c3 == one_step:
            plt.scatter(one_step,one_step,c='k',marker='s',zorder=3,s=100)

        plt.title('C3=' + str(c3) + ': ' + str(len(policy_subset)) + ' policies',fontsize=16)
        plt.xlabel('C1')
        plt.ylabel('C2')
        plt.xlim((min_policy_bound-margin, max_policy_bound+margin))
        plt.ylim((min_policy_bound-margin, max_policy_bound+margin))

    plt.suptitle('Before batch ' + str(k2+1))

    return fig

## SAVE ANIMATION
anim = animation.FuncAnimation(fig, make_frame, frames=batchnum,
                               interval=1000, blit=False)

anim.save('animation_bounds.gif', writer='imagemagick', fps=0.5)
