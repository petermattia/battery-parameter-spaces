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

addLines = True

plt.close('all')

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

colormap = 'plasma_r'

# IMPORT RESULTS
# Get folder path containing text files
file_list = sorted(glob.glob('./bounds/4_bounds.pkl'))
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
fig = plt.figure()
plt.style.use('classic')
plt.rcParams.update({'font.size': 16})
plt.set_cmap(colormap)

plt.hist(data, bins=12, range=(600,1200))
plt.xlabel('OED-estimated lifetimes')
plt.ylabel('Count')

# Add lines for good policies
if addLines:
    policies = np.asarray([[4.8,5.2,5.2],
                           [5.2,5.2,4.8],
                           [4.4,5.6,5.2],
                           [7,4.8,4.8],
                           [8,4.4,4.4],
                           [3.6,6,5.6],
                           [8,6,4.8],
                           [8,7,5.2],
                           [6,5.6,4.4]])
    for k,p in enumerate(policies):
        idx = np.where(np.sum(p==param_space,axis=1)==3)[0][0]
        life_p = mean[idx]
        if k < 3: c='k'
        elif k<7: c='r'
        else: c='g'
        plt.axvline(life_p, color=c, linestyle='dashed', linewidth=2)

## SAVE Plot
plt.savefig('plots/OED_estimates_histogram.png', bbox_inches = 'tight')
plt.savefig('plots/OED_estimates_histogram.png', bbox_inches = 'tight')
