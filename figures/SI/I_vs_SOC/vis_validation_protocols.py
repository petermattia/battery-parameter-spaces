#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:57:12 2018

@author: peter
"""

from IvSOC_function import plot_protocol
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle

MAX_WIDTH = 183 # mm
figsize=(MAX_WIDTH / 25.4, MAX_WIDTH / 25.4)

fig, ax = plt.subplots(3,3,figsize=figsize, sharex=True, sharey=True)

filename = 'validation_protocols.csv'
pols = np.genfromtxt(filename, delimiter=',',skip_header=1)

########## LOAD DATA ##########
# Load predictions
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',', skip_header=1)

validation_protocols = pred_data[:,0:3]
predicted_lifetimes = pred_data[:,3:]

# Load final results
filename = 'final_results.csv'
final_data = np.genfromtxt(filename, delimiter=',', skip_header=1)

final_lifetimes = final_data[:,3:]

# Load OED means
oed_bounds_file = glob.glob('4_bounds.pkl')[0]
with open(oed_bounds_file, 'rb') as infile:
        param_space, ub, lb, all_oed_means = pickle.load(infile)

intersect = [i for i, protocol in enumerate(param_space) if (protocol == validation_protocols).all(1).any()]
oed_protocol_subset = param_space[intersect]

oed_means = all_oed_means[intersect]
oed_bounds = (ub[intersect] - lb[intersect]) / 2 / (5 * 0.5 ** 5) # divide by beta_5

# reorder oed_means by comparing ordering of oed_protocol_subset with validation_protocols 
idx = np.argwhere(np.all(validation_protocols[:, None] == oed_protocol_subset, axis=-1))[:, 1]
oed_means = oed_means[idx]
oed_bounds = oed_bounds[idx]
    
########## CALCULATIONS ##########

# Summary statistics
pred_means = np.round(np.nanmean(predicted_lifetimes, axis=1))
pred_sterr = np.round(1.96*np.nanstd(predicted_lifetimes, axis=1)/np.sqrt(5))

final_means = np.round(np.nanmean(final_lifetimes, axis=1))
final_sterr = np.round(1.96*np.nanstd(final_lifetimes, axis=1)/np.sqrt(5))

########## PLOT ##########
for k, p in enumerate(pols):
    life_dict = {'oed': int(round(oed_means[k])),
                 'oed_bounds': int(round(oed_bounds[k])),
                 'pred': int(round(pred_means[k])),
                 'pred_sterr': int(round(pred_sterr[k])),
                 'final': int(round(final_means[k])),
                 'final_sterr': int(round(final_sterr[k]))}
    ax_temp = ax[int(k/3)][k%3]
    plot_protocol(p[0],p[1],p[2],ax_temp,life_dict)
    if int(k/3) == 2:
        ax_temp.set_xlabel('State of charge (%)')
    if k%3 == 0:
        ax_temp.set_ylabel('Current (C rate)')
    ax_temp.set_title(chr(k+97), loc='left', weight='bold')
    
plt.tight_layout()
plt.savefig('val_protocols.png', dpi=300)
plt.savefig('val_protocols.eps', format='eps')