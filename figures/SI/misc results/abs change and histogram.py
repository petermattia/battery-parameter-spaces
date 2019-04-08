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
data = []
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        data.append(mean)
min_lifetime = 1000

## Find number of batches and policies
n_batches  = len(data)
n_policies = len(param_space)

change = np.zeros((n_policies, n_batches-1))
for k, batch in enumerate(data[:-1]):
    change[:,k] = data[k+1] - data[k]

mean_change = np.mean(np.abs(change),axis=0)

## find mean change as a percent
per_change = np.zeros((n_policies, n_batches-1))
for k, batch in enumerate(data[:-1]):
    per_change[:,k] = 100*(data[k+1] - data[k])/data[k+1]

mean_per_change = np.mean(np.abs(per_change),axis=0)

## find mean change for top K policies
top_K_pols_list = [5,10,25,50,224]
mean_change_topk = np.zeros((len(top_K_pols_list),n_batches-1))
mean_per_change_topk = np.zeros((len(top_K_pols_list),n_batches-1))

for k, n in enumerate(top_K_pols_list):
    top_pol_idx = np.argsort(-mean)[0:n]
    mean_change_topk[k,:] = np.mean(np.abs(change[top_pol_idx]),axis=0)
    mean_per_change_topk[k,:] = np.mean(np.abs(per_change[top_pol_idx]),axis=0)

## plot
batches = np.arange(n_batches-1)+1
fig, ax = plt.subplots(1,2,figsize=(16,6))
ax0 = ax[0]
ax1 = ax[1]
ax0.set_title('a', loc='left', weight='bold')
ax1.set_title('b', loc='left', weight='bold')

cm = plt.get_cmap('winter')
ax0.set_color_cycle([cm(1.*i/len(top_K_pols_list)) for i in range(len(top_K_pols_list))])

legend = ['K = ' + str(k) for k in top_K_pols_list]

for i in range(len(top_K_pols_list)-1):
    ax0.plot(batches,mean_change_topk[i])
ax0.plot(batches,mean_change_topk[i+1],'k')
ax0.set_xlabel('Batch index (change from round n-1 to n)')
ax0.set_ylabel('Mean abs. change in estimated lifetime\nfor top K policies (cycles)')
ax0.set_ylim((0,140))
ax0.set_xticks(np.arange(1, 5, step=1))
ax0.legend(legend,frameon=False)

## Histogram
with plt.style.context(('classic')):
    ax1.hist(data[-1], bins=12, range=(600,1200),color=[0.1,0.4,0.8])
ax1.set_xlabel('OED-estimated lifetimes')
ax1.set_ylabel('Count')
ax1.set_xlim([600,1200])

plt.savefig('misc_results.png', bbox_inches='tight')
plt.savefig('misc_results.pdf', bbox_inches='tight', format='pdf')

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_color_cycle([cm(1.*i/len(top_K_pols_list)) for i in range(len(top_K_pols_list))])
#for i in range(len(top_K_pols_list)-1):
#    plt.plot(batches,mean_per_change_topk[i])
#plt.plot(batches,mean_per_change_topk[i+1],'k')
#plt.xlabel('Batch index (change from round n-1 to n)')
#plt.ylabel('Mean abs. change in estimated lifetime for top K policies (%)')
#plt.ylim((0,14))
#plt.xticks(np.arange(1, 5, step=1))
#plt.legend(legend)
#
#plt.savefig('plots/percent_change_vs_batch.png', bbox_inches='tight')
