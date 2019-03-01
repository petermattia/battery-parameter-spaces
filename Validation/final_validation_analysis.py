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
from cycler import cycler

########## LOAD DATA ##########
# Load predictions
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = pred_data[:,0:3]
predicted_lifetimes = pred_data[:,3:]

# Load final results
filename = 'final_results.csv'
final_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

final_lifetimes = final_data[:,3:]

# Load OED means
oed_bounds_file = glob.glob('../data/bounds/4_bounds.pkl')[0]
with open(oed_bounds_file, 'rb') as infile:
        param_space, ub, lb, all_oed_means = pickle.load(infile)

intersect = [i for i, policy in enumerate(param_space) if (policy == validation_policies).all(1).any()]

oed_means = all_oed_means[intersect]
oed_policy_subset = param_space[intersect]

# reorder oed_means by comparing ordering of oed_policy_subset with validation_policies 
idx = np.argwhere(np.all(validation_policies[:, None] == oed_policy_subset, axis=-1))[:, 1]
oed_means = oed_means[idx]

########## CALCULATIONS ##########

# calculate early prediction error
rmse = np.sqrt(np.nanmean(((predicted_lifetimes - final_lifetimes) ** 2)))
print('RMSE = ' + str(rmse) + ' cycles')

mape = np.nanmean( np.abs(predicted_lifetimes - final_lifetimes) /final_lifetimes) * 100
print('MAPE = ' + str(mape) + '%')

# calculate bias-corrected early prediction error
predicted_lifetimes_bias_corrected = predicted_lifetimes \
  - np.nanmean(predicted_lifetimes) + np.nanmean(final_lifetimes)
rmse = np.sqrt(np.nanmean(((predicted_lifetimes_bias_corrected - final_lifetimes) ** 2)))
print('RMSE = ' + str(rmse) + ' cycles (bias-corrected)')

mape = np.nanmean( np.abs(predicted_lifetimes_bias_corrected - final_lifetimes) /final_lifetimes) * 100
print('MAPE = ' + str(mape) + '% (bias-corrected)')

# Summary statistics
pred_means = np.round(np.nanmean(predicted_lifetimes,axis=1))
pred_stdevs = np.round(np.nanstd(predicted_lifetimes,axis=1))

final_means = np.round(np.nanmean(final_lifetimes,axis=1))
final_stdevs = np.round(np.nanstd(final_lifetimes,axis=1))

# Rankings calculations
oed_ranks = np.empty_like(oed_means.argsort())
oed_ranks[oed_means.argsort()] = np.arange(len(oed_means))
oed_ranks = np.max(oed_ranks) - oed_ranks + 1 # swap order and use 1-indexing

pred_ranks = np.empty_like(pred_means.argsort())
pred_ranks[pred_means.argsort()] = np.arange(len(pred_means))
pred_ranks = np.max(pred_ranks) - pred_ranks + 1 # swap order and use 1-indexing

final_ranks = np.empty_like(final_means.argsort())
final_ranks[final_means.argsort()] = np.arange(len(final_means))
final_ranks = np.max(final_ranks) - final_ranks + 1 # swap order and use 1-indexing

########## PLOTS ##########

## Lifetimes plot
fig = plt.errorbar(pred_means,oed_means,fmt='o',xerr=pred_stdevs)
ax = plt.gca()
plt.xlim([0,1200])
plt.ylim([0,1200])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Mean cycle life from early prediction')
plt.ylabel('Estimated cycle life from OED')
plt.savefig('plots/lifetimes_oed_vs_pred.png')

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
plt.savefig('plots/rankings_oed_vs_pred.png')


## Lifetimes plot
plt.figure()
fig3 = plt.errorbar(pred_means,final_means,fmt='o',xerr=pred_stdevs,yerr=final_stdevs)
ax = plt.gca()
plt.xlim([0,1300])
plt.ylim([0,1300])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Mean cycle life from early prediction')
plt.ylabel('Mean cycle life')
plt.savefig('plots/lifetimes_pred_vs_final.png')

## Rankings plot
plt.figure()
fig4 = plt.plot(pred_ranks,final_ranks,'o')
ax = plt.gca()
plt.xlim([0,10])
plt.ylim([0,10])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Mean ranking from early prediction')
plt.ylabel('True ranking')
plt.savefig('plots/rankings_pred_vs_final.png')

## Lifetimes plot
plt.figure()
fig5 = plt.errorbar(oed_means,final_means,fmt='o',yerr=final_stdevs)
ax = plt.gca()
plt.xlim([0,1300])
plt.ylim([0,1300])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Estimated cycle life from OED')
plt.ylabel('Mean cycle life')
plt.savefig('plots/lifetimes_oed_vs_final.png')

## Rankings plot
plt.figure()
fig6 = plt.plot(oed_ranks,final_ranks,'o')
ax = plt.gca()
plt.xlim([0,10])
plt.ylim([0,10])
ax.set_aspect('equal', 'box')
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.xlabel('Estimated ranking from OED')
plt.ylabel('True ranking')
plt.savefig('plots/rankings_oed_vs_final.png')


## Lifetimes plot - raw
fig, ax0 = plt.subplots()
ax0.set_prop_cycle(cycler('color', ['b','r','r','r','k','b','b','b','k']))
fig7 = ax0.plot(np.transpose(predicted_lifetimes), np.transpose(final_lifetimes),'o')
plt.legend(validation_policies)
plt.xlim([0,1300])
plt.ylim([0,1300])
ax0.set_aspect('equal', 'box')
ax0.plot(ax0.get_xlim(), ax0.get_ylim(), ls='--', c='.3')
ax0.plot([100,100], ax0.get_ylim(), ls="--", c='r')
plt.xlabel('Mean cycle life from early prediction')
plt.ylabel('Mean cycle life')
plt.savefig('plots/lifetimes_pred_vs_final_raw.png')