#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import pickle

plt.close('all')

FS = 14

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

# IMPORT RESULTS
# Get pickle files of bounds
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
means = []
ubs = []
lbs = []
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        means.append(mean)
        ubs.append(ub)
        lbs.append(lb)

## Find number of batches and policies
n_batches  = len(means)
n_policies = len(param_space)

## plot
batches = np.arange(n_batches-1)+1
plt.subplots(3,2,figsize=(9,12))

batches = np.arange(n_batches-1)+1

## Bounds
for k, mean in enumerate(means):
    ub = ubs[k]
    lb = lbs[k]
    ye = [(mean-lb)/(5*0.5**5),(ub-mean)/(5*0.5**5)]
    
    ax = plt.subplot2grid((5, 2), (k, 0), colspan=2) 
    ax.errorbar(np.arange(224),mean,yerr=ye,fmt='o',color=[0.1,0.4,0.8],capsize=2)
    ax.set_xlim((-1,225))
    ax.set_ylim((0,2000))
    ax.set_xlabel('Protocol index')
    if k==0:
        ax.set_ylabel('Bounds on cycle life\nbefore round 1, $\mathit{β_{0}σ_{0,i}}$')
    else:
        mathstr = '{β_{'+str(k)+'}σ_{'+str(k)+',i}}'
        ax.set_ylabel('Bounds on cycle life\nafter round {}, $\mathit'.format(k)+mathstr+'$')
    ax.set_xticks([], [])
    ax.set_title(chr(97+k), loc='left', weight='bold')

plt.tight_layout()
plt.savefig('bounds_evolution.png', bbox_inches='tight')
plt.savefig('bounds_evolution.pdf', bbox_inches='tight', format='pdf')