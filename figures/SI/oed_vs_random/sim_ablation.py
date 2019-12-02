#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:12:13 2019

@author: peter

This script plots the results of simulations comparing the closed loop with OED
to random searching. The results were generated on a cluster
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

MAX_WIDTH = 183 / 25.4 # mm -> inches
figsize=(MAX_WIDTH, 2/3 * MAX_WIDTH)

file = 'sim_ablation.pkl'
with open(file, 'rb') as infile:
        random_performance_means, random_performance_stds, \
        oed_performance_means, oed_performance_stds = pickle.load(infile)

num_channels = [1, 8, 16, 24, 48]
num_rounds = [1, 2, 3, 4]

fig, ax = plt.subplots(2,3,figsize=figsize,sharey=True)

for k, channels in enumerate(num_channels):
    temp_ax = ax[int(k/3)][k%3]
    temp_ax.errorbar(np.array(num_rounds),
                 random_performance_means[k],
                 yerr=1.96*np.vstack((random_performance_stds[k], 
                                      random_performance_stds[k])),
                 marker='o',
                 linestyle=':',
                 color=u'#7A68A6',
                 label='CLO w/ random')
    
    temp_ax.errorbar(np.array(num_rounds),
                     oed_performance_means[k][1:],
                     yerr=1.96*np.vstack((oed_performance_stds[k][1:], 
                                          oed_performance_stds[k][1:])),
                     marker='o',
                     linestyle=':',
                     color=u'#467821',
                     label='CLO w/ MAB')
    
    temp_ax.set_title(chr(k+97), loc='left', weight='bold', fontsize=8)
    #temp_ax.set_xlim((0.5, 4.5))
    temp_ax.set_xticks(np.arange(1, 5))
    temp_ax.set_ylim((700,1200))
    annotation_text = ' channels'
    if k==0:
        annotation_text = ' channel'
    temp_ax.annotate(str(channels) + annotation_text, (4.1, 720),
                     horizontalalignment='right')
    if k==0:
        temp_ax.legend(loc='upper left')
        
    
    temp_ax.set_xlabel('Number of rounds of testing')
        
    if k%3 == 0:
        temp_ax.set_ylabel('True cycle life of current best protocol')
        
ax[-1, -1].axis('off')

## SAVE
plt.tight_layout()
plt.savefig('sim_ablation.png', dpi=300)
plt.savefig('sim_ablation.eps', format='eps')