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

## INITIALIZE CONTOUR PLOT
# SETTINGS
fig = plt.figure()
plt.style.use('classic')
plt.rcParams.update({'font.size': 16})
plt.set_cmap(colormap)

plt.hist(data, bins=12, range=(600,1200))
plt.xlabel('OED-estimated lifetimes')
plt.ylabel('Count')

## SAVE Plot
plt.savefig('plots/OED_estimates_histogram.png', bbox_inches = 'tight')
plt.savefig('plots/OED_estimates_histogram.png', bbox_inches = 'tight')
