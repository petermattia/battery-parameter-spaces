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

batchnum = len(means) # Find number of batches
polnum   = len(means[0]) # Find number of batches

## INITIALIZE LEADERBOARD PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()


# FUNCTION FOR LOOPING THROUGH BATCHES
def make_frame(k2):
    plt.cla()
    
    ## PLOT POLICIES
    plt.ylim((0,24))
    plt.xlim((0,10))
    plt.axis('off')
    
    plt.text(5,22,'Before batch ' + str(k2+1),fontsize=24,horizontalalignment='center')
    
    best_indices = np.flip(np.argsort(means[k2]),axis=0)
    
    for k3 in np.arange(10):
        k = best_indices[k3]
        plt.text(5,20-k3*2,str(k3+1) + ': Policy ' + str(k) + ' '
                 + str(param_space[k]) + 
                 ', ' + str(int(means[k2][k])) + 
                 ' (' + str(int(lbs[k2][k])) + ',' + str(int(ubs[k2][k])) + ')',
                 fontsize=20,
                 horizontalalignment='center',
                 verticalalignment='center')
    return fig

## SAVE ANIMATION
anim = animation.FuncAnimation(fig, make_frame, frames=batchnum,
                               interval=1000, blit=False)

anim.save('plots/leaderboard_animate.gif', writer='imagemagick', fps=0.5)
