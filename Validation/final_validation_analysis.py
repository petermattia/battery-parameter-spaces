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
from scipy.stats import pearsonr

plt.close('all')

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

validation_pol_leg = []
for p in validation_policies:
    c4 = 0.2/(1/6 - (0.2/p[0] + 0.2/p[1] + 0.2/p[2]))
    validation_pol_leg.append(str(p[0])+'C-'+str(p[1])+'C-'+str(p[2])+'C-'+'{0:.3f}C'.format(c4))

FS = 14
upper_lim = 1500

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]
custom_cycler = (cycler(color=    [c1 , c2, c2, c2, c3, c1, c1, c1, c3]) +
                 cycler(marker=   ['o','o','s','v','o','s','v','^','s']) +
                 cycler(linestyle=['' , '', '', '', '', '', '', '', '']))

def init_lifetimes_plot():
    fig, ax = plt.subplots()
    ax.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
    ax.set_prop_cycle(custom_cycler)
    return ax
    
def init_rankings_plot():
    fig, ax = plt.subplots()
    ax.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
    ax.set_prop_cycle(custom_cycler)
    return ax

def format_lifetimes_plot(filename, r):
    plt.xlim([0,upper_lim])
    plt.ylim([0,upper_lim])
    ax0.set_aspect('equal', 'box')
    ax0.set_xticks([0,500,1000,1500])
    ax0.set_yticks([0,500,1000,1500])
    plt.legend(validation_pol_leg, bbox_to_anchor=(1.01, 0.85))
    plt.annotate('r = {:.2}'.format(r),(75,1350))
    plt.savefig('plots/'+filename+'.png',bbox_inches='tight')
    plt.savefig('plots/'+filename+'.pdf',bbox_inches='tight',format='pdf')

def format_rankings_plot(filename, tau):
    plt.xlim([0,10])
    plt.ylim([0,10])
    ax0.set_aspect('equal', 'box')
    plt.legend(validation_pol_leg, bbox_to_anchor=(1.01, 0.85))
    plt.annotate('τ = {:.2}'.format(tau),(0.5,9.0))
    plt.savefig('plots/'+filename+'.png',bbox_inches='tight')
    plt.savefig('plots/'+filename+'.pdf',bbox_inches='tight',format='pdf')
    
#### PRED vs OED
## Lifetimes plot
ax0 = init_lifetimes_plot()
for k in range(len(pred_means)):
    ax0.errorbar(pred_means[k],oed_means[k],xerr=pred_sterr[k])
plt.xlabel('Mean predicted cycle life',fontsize=FS)
plt.ylabel('OED-estimated cycle life',fontsize=FS)
r = pearsonr(pred_means,oed_means)[0]
format_lifetimes_plot('pred_vs_oed',r)

## Individual lifetimes plot
ax0 = init_lifetimes_plot()
for k in range(len(predicted_lifetimes)):
    predicted_lifetimes_vector = predicted_lifetimes[k][~np.isnan(predicted_lifetimes[k])]
    oed_value_vector = oed_means[k]*np.ones(np.count_nonzero(~np.isnan(predicted_lifetimes[k])))
    ax0.plot(predicted_lifetimes_vector, oed_value_vector)
plt.xlabel('Early-predicted cycle life',fontsize=FS)
plt.ylabel('OED-estimated cycle life',fontsize=FS)
idx = ~np.isnan(predicted_lifetimes.ravel())
r = pearsonr(predicted_lifetimes.ravel()[idx],np.repeat(oed_means,5)[idx])[0]
format_lifetimes_plot('pred_vs_oed_ind',r)

## Rankings plot
ax0 = init_rankings_plot()
for k in range(len(pred_ranks)):
    plt.plot(pred_ranks[k],oed_ranks[k])
plt.xlabel('Mean early-predicted ranking',fontsize=FS)
plt.ylabel('OED-estimated ranking',fontsize=FS)
tau = kendalltau(pred_ranks,oed_ranks)[0]
format_rankings_plot('pred_vs_oed_rankings',tau)

#### PRED vs FINAL
## Lifetimes plot
ax0 = init_lifetimes_plot()
for k in range(len(pred_means)):
    plt.errorbar(pred_means[k],final_means[k],xerr=pred_sterr[k],yerr=final_sterr[k])
plt.xlabel('Mean early-predicted cycle life',fontsize=FS)
plt.ylabel('Mean cycle life',fontsize=FS)
r = pearsonr(pred_means,final_means)[0]
format_lifetimes_plot('pred_vs_final',r)

## Individual lifetimes plot
ax0 = init_lifetimes_plot()
ax0.plot(np.transpose(predicted_lifetimes), np.transpose(final_lifetimes))
#plt.legend(validation_policies,loc=9, bbox_to_anchor=(2.1, 0.5))
#ax0.plot([100,100], ax0.get_ylim(), ls="--", c='r')
plt.xlabel('Early-predicted cycle life',fontsize=FS)
plt.ylabel('Observed cycle life',fontsize=FS)
r = pearsonr(predicted_lifetimes.ravel()[idx],final_lifetimes.ravel()[idx])[0]
format_lifetimes_plot('pred_vs_final_ind',r)

"""
## Lifetimes plot - raw, bias-corrected
ax0 = init_lifetimes_plot()
ax0.plot(np.transpose(predicted_lifetimes_bias_corrected), \
                np.transpose(final_lifetimes))
#plt.legend(validation_policies,loc=9, bbox_to_anchor=(2.1, 0.5))
plt.xlabel('Early-predicted cycle life',fontsize=FS)
plt.ylabel('Observed cycle life',fontsize=FS)
format_lifetimes_plot('pred_vs_final_biascorrected')
"""

## Rankings plot
ax0 = init_rankings_plot()
for k in range(len(pred_ranks)):
    plt.plot(pred_ranks[k],final_ranks[k])
plt.xlabel('Mean early-predicted ranking',fontsize=FS)
plt.ylabel('True ranking',fontsize=FS)
tau = kendalltau(pred_ranks,final_ranks)[0]
format_rankings_plot('pred_vs_final_rankings',tau)

#### OED vs FINAL
## Lifetimes plot
ax0 = init_lifetimes_plot()
for k in range(len(oed_means)):
    plt.errorbar(oed_means[k],final_means[k],yerr=final_sterr[k])
plt.xlabel('OED-estimated cycle life',fontsize=FS)
plt.ylabel('Mean cycle life',fontsize=FS)
r = pearsonr(oed_means,final_means)[0]
format_lifetimes_plot('oed_vs_final',r)

## Individual lifetimes plot
ax0 = init_lifetimes_plot()
for k in range(len(predicted_lifetimes)):
    final_lifetimes_vector = final_lifetimes[k][~np.isnan(final_lifetimes[k])]
    oed_value_vector = oed_means[k]*np.ones(np.count_nonzero(~np.isnan(final_lifetimes[k])))
    ax0.plot(oed_value_vector, final_lifetimes_vector)
plt.xlabel('OED-estimated cycle life',fontsize=FS)
plt.ylabel('Observed cycle life',fontsize=FS)
r = pearsonr(np.repeat(oed_means,5),final_lifetimes.ravel())[0]
format_lifetimes_plot('oed_vs_final_ind',r)

## Rankings plot
ax0 = init_rankings_plot()
for k in range(len(pred_ranks)):
    plt.plot(oed_ranks[k],final_ranks[k])
plt.xlabel('OED-estimated ranking',fontsize=FS)
plt.ylabel('True ranking',fontsize=FS)
tau = kendalltau(oed_ranks,final_ranks)[0]
format_rankings_plot('oed_vs_final_rankings',tau)