#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contour_points_general.py creates contour plot based on file_path
    This is useful for creating contour plots of the most recent OED recommendations

Peter Attia
Last modified August 22, 2018
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

min_policy_bound, max_policy_bound = 3.6, 8
C3 = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]
C4_LIMITS = [0.1, 4.81]

filename = '1' # saving
plot_file = 'data/plots/contour_' + filename + '.png'

# Import policies
file_path = '/Users/peter/Documents/GitHub/battery-parameter-spaces/data/batch/'+filename+'.csv';
policies = np.genfromtxt(file_path, delimiter=',')

COLOR_LIM = [2.5,4.8]

one_step = 4.8
margin = 0.2 # plotting margin

# Calculate C4(CC1, CC2) values for contour lines
C1_grid = np.arange(min_policy_bound-margin,max_policy_bound + margin,0.01)
C2_grid = np.arange(min_policy_bound-margin,max_policy_bound + margin,0.01)
[X,Y] = np.meshgrid(C1_grid,C2_grid)

## CREATE CONTOUR PLOT
fig = plt.figure() # x = C1, y = C2, cuts = C3, contours = C4
plt.style.use('classic')
plt.rcParams.update({'font.size': 16})
plt.set_cmap('viridis')
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()


## MAKE PLOT
for k, c3 in enumerate(C3):
    plt.subplot(2,3,k+1)
    plt.axis('square')

    C4 = 0.2/(1/6 - (0.2/X + 0.2/Y + 0.2/c3))
    C4[np.where(C4<C4_LIMITS[0])]  = float('NaN')
    C4[np.where(C4>C4_LIMITS[1])] = float('NaN')

    ## PLOT CONTOURS
    levels = np.arange(2.5,4.8,0.25)
    C = plt.contour(X,Y,C4,levels,zorder=1,vmin=COLOR_LIM[0],vmax=COLOR_LIM[1])
    plt.clabel(C,fmt='%1.1f')

    ## PLOT POLICIES
    if c3 == 4.8:
        plt.scatter(one_step,one_step,c='k',marker='s',zorder=3,s=50) ## BASELINE

    idx_subset = np.where(policies[:,2]==c3)
    policy_subset = policies[idx_subset,:][0]
    plt.scatter(policy_subset[:,0],policy_subset[:,1],c='k',zorder=2,s=50)

    plt.title('C3=' + str(c3) + ': ' + str(len(policy_subset)) + ' policies',fontsize=16)
    plt.xlabel('C1')
    plt.ylabel('C2')
    plt.xlim((min_policy_bound-margin, max_policy_bound+margin))
    plt.ylim((min_policy_bound-margin, max_policy_bound+margin))

plt.tight_layout()

# Add colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
minn, maxx = COLOR_LIM[0], COLOR_LIM[1]
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
m.set_array([])
cbar = plt.colorbar(m, cax=cbar_ax)
#fig.colorbar(m, cax=cbar_ax)
plt.clim(min(C4_LIMITS),max(C4_LIMITS))
cbar.ax.set_title('C4')

## SAVE FIGURE
plt.savefig(plot_file, bbox_inches='tight')
