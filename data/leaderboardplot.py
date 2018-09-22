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
polnum   = len(means[0]) # Find number of policies

## INITIALIZE LEADERBOARD PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()
    
## PLOT POLICIES
plt.xlim((0,batchnum))
plt.ylim((0,11))
plt.gca().invert_yaxis()
plt.xlabel('Batch index')
plt.ylabel('Policy ranking (1=best)')

for k in np.arange(batchnum):
    best_indices = np.flip(np.argsort(means[k]),axis=0)
    
    for k2 in np.arange(10):
        k3 = best_indices[k2]
        ranking_k = np.zeros(batchnum)
        for k4 in np.arange(batchnum):
            best_indices_2 = np.flip(np.argsort(means[k4]),axis=0)
            ranking_k[k4] = np.where(best_indices_2 == k3)[0][0]
        
        plt.plot(np.arange(batchnum),ranking_k+1)
        
        if k==batchnum-1:
            plt.text(4,k2+1,str(k3) + ': ' + str(param_space[k3]),
                     fontsize=16,verticalalignment='center')

plt.savefig('plots/leaderboardplot.png',bbox_inches='tight')