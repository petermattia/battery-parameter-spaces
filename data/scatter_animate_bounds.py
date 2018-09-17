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

# IMPORT RESULTS
# Get folder path containing text files
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
means, ubs, lbs = [], [], []
min_lifetime = 10000
max_lifetime = -1
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        means.append(mean)
        ubs.append(ub)
        lbs.append(lb)
        min_lifetime = min(np.min(lbs),min_lifetime)
        max_lifetime = max(np.max(ubs),max_lifetime)

# Find number of batches
batchnum = len(means)

## INITIALIZE CONTOUR PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
plt.set_cmap(colormap)
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()

plt.suptitle('Batch ' + str(1))

# FUNCTION FOR LOOPING THROUGH BATCHES
def make_frame(k2):
    plt.cla()
    
    ## PLOT POLICIES
    plt.errorbar(np.arange(224), means[k2],yerr=[means[k2]-lbs[k2],ubs[k2]-means[k2]],fmt='o')
    plt.xlim((0,224))
    plt.ylim((min_lifetime-10,max_lifetime+10))
    plt.xlabel('Policy index')
    plt.ylabel('Upper and lower bounds on lifetime')
    plt.suptitle('Before batch ' + str(k2+1))

    return fig

## SAVE ANIMATION
anim = animation.FuncAnimation(fig, make_frame, frames=batchnum,
                               interval=1000, blit=False)

anim.save('plots/scatter_animation_bounds.gif', writer='imagemagick', fps=0.5)
