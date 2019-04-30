#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:03 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import pickle
from cycler import cycler
from scipy.stats import kendalltau

plt.close('all')

FS = 14
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']
upper_lim = 1500

########## LOAD DATA ##########
# Load predictions
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = pred_data[:,0:3]
predicted_lifetimes = pred_data[:,3:]

validation_pol_leg = []
for p in validation_policies:
    c4 = 0.2/(1/6 - (0.2/p[0] + 0.2/p[1] + 0.2/p[2]))
    validation_pol_leg.append(str(p[0])+'C-'+str(p[1])+'C-'+str(p[2])+'C-'+'{0:.3f}C'.format(c4))

# Load final results
filename = 'final_results.csv'
final_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

final_lifetimes = final_data[:,3:]

# Load OED means
oed_bounds_file = glob.glob('4_bounds.pkl')[0]
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
print('Bias correction: ' + str(np.nanmean(final_lifetimes) - \
                                np.nanmean(predicted_lifetimes)) + ' cycles')
rmse = np.sqrt(np.nanmean(((predicted_lifetimes_bias_corrected - final_lifetimes) ** 2)))
print('RMSE = ' + str(rmse) + ' cycles (bias-corrected)')

mape = np.nanmean( np.abs(predicted_lifetimes_bias_corrected - final_lifetimes) /final_lifetimes) * 100
print('MAPE = ' + str(mape) + '% (bias-corrected)')

# Summary statistics
pred_means = np.round(np.nanmean(predicted_lifetimes,axis=1))
pred_sterr = np.round(1.96*np.nanstd(predicted_lifetimes,axis=1)/np.sqrt(5))

final_means = np.round(np.nanmean(final_lifetimes,axis=1))
final_sterr = np.round(1.96*np.nanstd(final_lifetimes,axis=1)/np.sqrt(5))

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

fig = plt.figure(figsize=(12,12))

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]
custom_cycler = (cycler(color=    [c1 , c2, c2, c2, c3, c1, c1, c1, c3]) +
                 cycler(marker=   ['o','o','s','v','o','s','v','^','s']) +
                 cycler(linestyle=['' , '', '', '', '', '', '', '', '']))

ax0 = plt.subplot(3,3,1)
#### PRED vs OED
## Lifetimes plot
ax0.set_prop_cycle(custom_cycler)
ax0.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3')
for k in range(len(pred_means)):
    ax0.errorbar(pred_means[k],oed_means[k],xerr=pred_sterr[k])
plt.xlim([0,upper_lim])
plt.ylim([0,upper_lim])
ax0.set_aspect('equal', 'box')
#plt.legend(validation_policies)
plt.xlabel('Mean predicted cycle life',fontsize=FS)
plt.ylabel('OED-estimated cycle life',fontsize=FS)
plt.title(chr(97),loc='left', weight='bold',fontsize=FS)
ax0.set_xticks([0,750,1500])
ax0.set_yticks([0,750,1500])

## Individual lifetimes plot
ax0 = plt.subplot(3,3,2)
ax0.set_prop_cycle(custom_cycler)
ax0.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3')
for k in range(len(predicted_lifetimes)):
    predicted_lifetimes_vector = predicted_lifetimes[k][~np.isnan(predicted_lifetimes[k])]
    oed_value_vector = oed_means[k]*np.ones(np.count_nonzero(~np.isnan(predicted_lifetimes[k])))
    ax0.plot(predicted_lifetimes_vector, oed_value_vector)
plt.xlim([0,upper_lim])
plt.ylim([0,upper_lim])
ax0.set_aspect('equal', 'box')
#plt.legend(validation_policies)
plt.xlabel('Early-predicted cycle life',fontsize=FS)
plt.ylabel('OED-estimated cycle life',fontsize=FS)
plt.title(chr(98),loc='left', weight='bold',fontsize=FS)
ax0.set_xticks([0,750,1500])
ax0.set_yticks([0,750,1500])

## Rankings plot
ax0 = plt.subplot(3,3,3)
ax0.plot((-1,11),(-1,11), ls="--", c=".3")
ax0.set_prop_cycle(custom_cycler)
for k in range(len(pred_ranks)):
    plt.plot(pred_ranks[k],oed_ranks[k])
plt.xlim([0,10])
plt.ylim([0,10])
ax0.set_aspect('equal', 'box')
plt.xlabel('Mean early-predicted ranking',fontsize=FS)
plt.ylabel('OED-estimated ranking',fontsize=FS)
plt.title(chr(99),loc='left', weight='bold',fontsize=FS)
ax0.set_yticks([0,5,10]) # consistent with x

#### PRED vs FINAL
## Lifetimes plot
ax0 = plt.subplot(3,3,4)
ax0.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3')
ax0.set_prop_cycle(custom_cycler)
for k in range(len(pred_means)):
    plt.errorbar(pred_means[k],final_means[k],xerr=pred_sterr[k],yerr=final_sterr[k])
plt.xlim([0,upper_lim])
plt.ylim([0,upper_lim])
ax0.set_aspect('equal', 'box')
#plt.legend(validation_policies)
plt.xlabel('Mean early-predicted cycle life',fontsize=FS)
plt.ylabel('Mean cycle life',fontsize=FS)
plt.title(chr(100),loc='left', weight='bold',fontsize=FS)
ax0.set_xticks([0,750,1500])
ax0.set_yticks([0,750,1500])

## Lifetimes plot - raw
ax0 = plt.subplot(3,3,5)
ax0.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
ax0.set_prop_cycle(custom_cycler)
ax0.plot(np.transpose(predicted_lifetimes), np.transpose(final_lifetimes))
#plt.legend(validation_policies,loc=9, bbox_to_anchor=(2.1, 0.5))
plt.xlim([0,upper_lim])
plt.ylim([0,upper_lim])
ax0.set_aspect('equal', 'box')
#ax0.plot([100,100], ax0.get_ylim(), ls="--", c='r')
plt.xlabel('Early-predicted cycle life',fontsize=FS)
plt.ylabel('Observed cycle life',fontsize=FS)
plt.title(chr(101),loc='left', weight='bold',fontsize=FS)
ax0.set_xticks([0,750,1500])
ax0.set_yticks([0,750,1500])

## Rankings plot
ax0 = plt.subplot(3,3,6)
ax0.plot((-1,11),(-1,11), ls="--", c=".3")
ax0.set_prop_cycle(custom_cycler)
for k in range(len(pred_ranks)):
    plt.plot(pred_ranks[k],final_ranks[k])
plt.xlim([0,10])
plt.ylim([0,10])
ax0.set_aspect('equal', 'box')
plt.xlabel('Mean early-predicted ranking',fontsize=FS)
plt.ylabel('True ranking',fontsize=FS)
plt.title(chr(102),loc='left', weight='bold',fontsize=FS)
ax0.set_yticks([0,5,10]) # consistent with x

### Lifetimes plot - raw, bias-corrected
#fig, ax0 = plt.subplots()
#ax0.set_prop_cycle(custom_cycler)
#ax0.plot(np.transpose(predicted_lifetimes_bias_corrected), \
#                np.transpose(final_lifetimes))
#plt.legend(validation_policies)
#plt.xlim([0,upper_lim])
#plt.ylim([0,upper_lim])
#ax0.set_aspect('equal', 'box')
#ax0.plot(ax0.get_xlim(), ax0.get_ylim(), ls='--', c='.3')
##ax0.plot([100,100], ax0.get_ylim(), ls="--", c='r')
#plt.xlabel('Predicted cycle life')
#plt.ylabel('Observed cycle life')

#### OED vs FINAL
## Lifetimes plot
ax0 = plt.subplot(3,3,7)
ax0.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3')
ax0.set_prop_cycle(custom_cycler)
for k in range(len(oed_means)):
    plt.errorbar(oed_means[k],final_means[k],yerr=final_sterr[k])
plt.xlim([0,upper_lim])
plt.ylim([0,upper_lim])
#plt.legend(validation_policies)
ax0.set_aspect('equal', 'box')
plt.xlabel('OED-estimated cycle life',fontsize=FS)
plt.ylabel('Mean cycle life',fontsize=FS)
plt.title(chr(103),loc='left', weight='bold',fontsize=FS)
ax0.set_xticks([0,750,1500])
ax0.set_yticks([0,750,1500])

## Individual lifetimes plot
ax0 = plt.subplot(3,3,8)
ax0.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3')
ax0.set_prop_cycle(custom_cycler)
for k in range(len(predicted_lifetimes)):
    final_lifetimes_vector = final_lifetimes[k][~np.isnan(final_lifetimes[k])]
    oed_value_vector = oed_means[k]*np.ones(np.count_nonzero(~np.isnan(final_lifetimes[k])))
    ax0.plot(oed_value_vector, final_lifetimes_vector)
plt.xlim([0,upper_lim])
plt.ylim([0,upper_lim])
ax0.set_aspect('equal', 'box')
plt.xlabel('OED-estimated cycle life',fontsize=FS)
plt.ylabel('Observed cycle life',fontsize=FS)
plt.title(chr(104),loc='left', weight='bold',fontsize=FS)
ax0.set_xticks([0,750,1500])
ax0.set_yticks([0,750,1500])

## Rankings plot
ax0 = plt.subplot(3,3,9)
ax0.plot((-1,11),(-1,11), ls="--", c=".3")
ax0.set_prop_cycle(custom_cycler)
for k in range(len(pred_ranks)):
    plt.plot(oed_ranks[k],final_ranks[k])
plt.xlim([0,10])
plt.ylim([0,10])
ax0.set_aspect('equal', 'box')
plt.xlabel('OED-estimated ranking',fontsize=FS)
plt.ylabel('True ranking',fontsize=FS)
plt.title(chr(105),loc='left', weight='bold',fontsize=FS)
ax0.set_yticks([0,5,10]) # consistent with x

#
plt.tight_layout()
plt.subplots_adjust(right=0.8)
ax0 = plt.subplot(3,3,5)
#plt.legend(validation_pol_leg,loc=9, bbox_to_anchor=(3.8, 1.2),frameon=True)
plt.legend(validation_pol_leg,loc=9, bbox_to_anchor=(0.5, -2),frameon=True,ncol=3)
plt.savefig('validation_ablation.png',bbox_inches='tight')
plt.savefig('validation_ablation.pdf',bbox_inches='tight',format='pdf')