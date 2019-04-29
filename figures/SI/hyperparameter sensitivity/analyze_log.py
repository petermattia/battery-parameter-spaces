#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

lifetime_key = 'l4'

#sims = pd.read_csv('log.csv', names=['beta','gamma','epsilon','seed',lifetime_key])
sims = pd.read_csv('log.csv', names=['beta','gamma','epsilon','seed',
                                     'l1','l2','l3','l4','l5','l6','l7','l8','l9','l10'])

# Add column for rank
lifetimes = pd.read_csv('policies_lifetimes_sim.csv', 
                        names=['C1','C2','C3','C4','Lifetime'])
lifetimes_sorted = lifetimes.sort_values('Lifetime',ascending=False)
lifetimes_sorted = lifetimes_sorted.reset_index(drop=True)

lifetime_idx_array = np.zeros((len(sims),1))

for k,val in enumerate(sims[lifetime_key]):
    lifetime_idx_array[k] = lifetimes_sorted.index[lifetimes_sorted['Lifetime']==val].tolist()[0]

sims['mean rank'] = lifetime_idx_array


# Preinitialize
beta_len = len(sims.beta.unique())
gamma_len = len(sims.gamma.unique())
epsilon_len = len(sims.epsilon.unique())

means = np.zeros((beta_len,gamma_len,epsilon_len))
std = np.zeros((beta_len,gamma_len,epsilon_len))
mean_rank = np.zeros((beta_len,gamma_len,epsilon_len))
min_rank = np.zeros((beta_len,gamma_len,epsilon_len))
max_rank = np.zeros((beta_len,gamma_len,epsilon_len))

beta_count = 0
gamma_count = 0
epsilon_count = 0

for beta, subgroup in sims.groupby('beta'):
    for gamma, subgroup2 in subgroup.groupby('gamma'):
        for epsilon, subgroup3 in subgroup2.groupby('epsilon'):
            print('beta = ', beta,', gamma = ', gamma,', epsilon = ', epsilon)
            #print('beta_count = ', beta_count,', gamma_count = ', gamma_count,
            #      ', epsilon_count = ', epsilon_count)

            means[beta_count,gamma_count,epsilon_count] = subgroup3[lifetime_key].mean()
            std[beta_count,gamma_count,epsilon_count] = subgroup3[lifetime_key].std()

            mean_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].mean()
            min_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].min()
            max_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].max()

            epsilon_count += 1

        epsilon_count = 0
        gamma_count += 1

    gamma_count = 0
    beta_count += 1
    
min_lifetime = np.amin(means)
max_lifetime = np.amax(means)

"""
## Lifetime figure
plt.figure()
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()

c1='#1f77b4' #blue
c2='#7C0A02' #red

def fadeColor(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    assert len(c1)==len(c2)
    assert mix>=0 and mix<=1, 'mix='+str(mix)
    rgb1=np.array([int(c1[ii:ii+2],16) for ii in range(1,len(c1),2)])
    rgb2=np.array([int(c2[ii:ii+2],16) for ii in range(1,len(c2),2)])
    rgb=((1-mix)*rgb1+mix*rgb2).astype(int)
    c='#'+''.join([hex(a)[2:] for a in rgb])
    return c

for k, beta in enumerate(sims.beta.unique()):
    plt.subplot(2,3,k+1)
    for k2, gamma in enumerate(sims.gamma.unique()):
        means_subset = means[k,k2,:]
        std_subset = std[k,k2,:]
        plt.errorbar(sims.epsilon.unique(), means_subset, yerr=std_subset,
                     fmt='o', color=fadeColor(c1,c2,k2/7))

    plt.title('beta = ' + str(beta))
    plt.xlabel('epsilon')
    plt.ylabel('True lifetime of best policy')
    plt.gca().legend(sims.gamma.unique(),fontsize=10)
    #plt.xscale('log')
    plt.xlim((0.4,1.0))
    plt.ylim((1100,1250))
    plt.hlines([1208,1192],0.4,1.0)
"""
FS = 14
CROPPED_BOUNDS = False
colormap = 'plasma_r'
lower_lifetime_lim = 1000

## PLOT
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

## INITIALIZE CONTOUR PLOT
# SETTINGS
fig, ax = plt.subplots(2,3,figsize=(16,12),sharex=True,sharey=True)
#plt.style.use('classic')
plt.set_cmap(colormap)
if CROPPED_BOUNDS:
    minn, maxx = lower_lifetime_lim, max_lifetime
else:
    minn, maxx = min_lifetime, max_lifetime

fig.subplots_adjust(right=0.8)
fig.subplots_adjust(top=0.93)

k2 = 0

# FUNCTION FOR LOOPING THROUGH BATCHES
for k, c3 in enumerate(sims.epsilon.unique()):
    temp_ax = ax[int(k/3)][k%3]
    plt.axis('square')
    
    """
    ## PLOT COMBINATIONS
    idx_subset = np.where(param_space[:,2]==c3)
    policy_subset = param_space[idx_subset]
    lifetime_subset = data[k2][idx_subset]
    temp_ax.scatter(policy_subset[:,0],policy_subset[:,1],vmin=minn,vmax=maxx,
                c=lifetime_subset.ravel(),zorder=2,s=100)

    temp_ax.set_title(chr(k+97),loc='left', weight='bold',fontsize=FS)
    temp_ax.annotate('CC3=' + str(c3) + '\n' + str(len(policy_subset)) + ' policies',\
              (3.52, 3.52), fontsize=FS)
    if int(k/3)==1:
        temp_ax.set_xlabel('\beta',fontsize=FS)
    if k%3 == 0:
        temp_ax.set_ylabel('\gamma',fontsize=FS)
    temp_ax.set_xlim((min_policy_bound-margin, max_policy_bound+margin))
    temp_ax.set_ylim((min_policy_bound-margin, max_policy_bound+margin))
    """

# ADD COLORBAR
cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.72]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])
cbar = fig.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('True cycle life\nof best policy',fontsize=FS)

## SAVE
plt.savefig('hyperparameter_sensitivity.png', bbox_inches = 'tight')
plt.savefig('hyperparameter_sensitivity.pdf', bbox_inches = 'tight',format='pdf')
