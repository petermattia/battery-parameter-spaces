#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import glob
import pickle

plt.close('all')

##############################################################################
# PLOTTING PARAMETERS
batches_to_plot = [0,2,4]
labels = ['Initial','Middle','Final']

colormap = 'plasma_r'
el, az = 30, 240
point_size = 70
num_policies = 224
seed = 0
##############################################################################

# IMPORT RESULTS
# Get folder path containing pickle files
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


## INITIALIZE PLOT
# SETTINGS
fig = plt.figure(figsize=(20,5))
plt.style.use('classic')
plt.set_cmap(colormap)

# Randomly select indices to plot
np.random.seed(seed=seed)
indices = sorted(np.random.choice(len(param_space), num_policies, replace=False))

## MAKE SUBPLOTS
for k, batch_idx in enumerate(batches_to_plot):
    ax = fig.add_subplot(1, len(batches_to_plot), k+1, projection='3d')
    ax.set_aspect('equal')

    ## PLOT POLICIES
    CC1 = param_space[indices,0]
    CC2 = param_space[indices,1]
    CC3 = param_space[indices,2]
    lifetime = data[batch_idx][indices]
    ax.scatter(CC1,CC2,CC3, s=point_size, c=lifetime.ravel(),
               vmin=min_lifetime, vmax=max_lifetime)
    
    ax.set_xlabel('CC1'), ax.set_xlim([3, 8])
    ax.set_ylabel('CC2'), ax.set_ylim([3, 8])
    ax.set_zlabel('CC3'), ax.set_zlim([3, 8])
    #ax.set_title('Before batch '+str(batch_idx))
    ax.set_title(labels[k])
    
    ax.view_init(elev=el, azim=az)

# ADD COLORBAR
fig.subplots_adjust(left=0.01,right=0.85,bottom=0.02,top=0.98,wspace=0.000001)
cbar_ax = fig.add_axes([0.9, 0.05, 0.05, 0.85])
norm = matplotlib.colors.Normalize(min_lifetime, max_lifetime)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])
cbar = fig.colorbar(m, cax=cbar_ax)
cbar.ax.set_title('Estimated cycle life')


plt.savefig('plots/evolution.png')
plt.savefig('plots/evolution.pdf')