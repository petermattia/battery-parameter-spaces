#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:03 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

## Load predictions
filename = 'predictions.csv'
data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = data[:,0:3]
predicted_lifetimes = data[:,3:]

## Load OED means
oed_bounds_file = glob.glob('../data/bounds/4_bounds.pkl')[0]
with open(oed_bounds_file, 'rb') as infile:
        param_space, ub, lb, all_oed_means = pickle.load(infile)

intersect = [i for i, policy in enumerate(param_space) if (policy == validation_policies).all(1).any()]

oed_means = all_oed_means[intersect]
oed_policy_subset = param_space[intersect]

# reorder oed_means by comparing ordering of oed_policy_subset with validation_policies 
idx = np.argwhere(np.all(validation_policies[:, None] == oed_policy_subset, axis=-1))[:, 1]
oed_means = oed_means[idx]

## Summary statistics
pred_means = np.round(np.nanmean(predicted_lifetimes,axis=1))
pred_stdevs = np.round(np.nanstd(predicted_lifetimes,axis=1))

## Lifetimes plot
fig = plt.errorbar(pred_means,oed_means,fmt='o',xerr=pred_stdevs)
ax = plt.gca()
plt.xlim([500,1200])
plt.ylim([500,1200])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Mean cycle life from early prediction')
plt.ylabel('Estimated cycle life from OED')
plt.savefig('plots/lifetimes_pred.png')


## Rankings calculations
oed_ranks = np.empty_like(oed_means.argsort())
oed_ranks[oed_means.argsort()] = np.arange(len(oed_means))
oed_ranks = np.max(oed_ranks) - oed_ranks + 1 # swap order and use 1-indexing

pred_ranks = np.empty_like(pred_means.argsort())
pred_ranks[pred_means.argsort()] = np.arange(len(pred_means))
pred_ranks = np.max(pred_ranks) - pred_ranks + 1 # swap order and use 1-indexing

## Rankings plot
plt.figure()
fig2 = plt.plot(pred_ranks,oed_ranks,'o')
ax = plt.gca()
plt.xlim([0,10])
plt.ylim([0,10])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Mean ranking from early prediction')
plt.ylabel('Estimated ranking from OED')
plt.savefig('plots/rankings_pred.png')