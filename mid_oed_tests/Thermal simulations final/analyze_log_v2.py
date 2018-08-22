#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

sims = pd.read_csv('log.csv', names=['beta','gamma','epsilon','seed','lifetime'])
sim_name = 'hi'

# Add column for rank
lifetimes = pd.read_csv('policies_lifetimes' + sim_name + '.csv')
lifetimes_sorted = lifetimes.sort_values('Lifetime',ascending=False)
lifetimes_sorted = lifetimes_sorted.reset_index(drop=True)
    
lifetime_idx_array = np.zeros((len(sims),1))
    
for k,val in enumerate(sims['lifetime']):
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
            
            means[beta_count,gamma_count,epsilon_count] = subgroup3['lifetime'].mean()
            std[beta_count,gamma_count,epsilon_count] = subgroup3['lifetime'].std()
                
            mean_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].mean()
            min_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].min()
            max_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].max()
            
            epsilon_count += 1

        epsilon_count = 0
        gamma_count += 1
        
    gamma_count = 0
    beta_count += 1

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

plt.tight_layout()
plt.savefig(sim_name + '_lifetimes.png')


## Rank figure
plt.figure()
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()

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
        median_subset = mean_rank[k,k2,:]
        min_subset = min_rank[k,k2,:]
        max_subset = max_rank[k,k2,:]
        plt.errorbar(sims.epsilon.unique(), median_subset, 
                     yerr=[median_subset-min_subset,max_subset-median_subset],
                     fmt='o', color=fadeColor(c1,c2,k2/7))
        
    plt.title('beta = ' + str(beta))
    plt.xlabel('epsilon')
    plt.ylabel('True lifetime rank of best policy')
    plt.gca().legend(sims.gamma.unique(),fontsize=10)
    #plt.xscale('log')
    plt.xlim((0.4,1.0))
    plt.ylim((0,30))
    #plt.hlines([1208,1192],0.4,1.0)

plt.tight_layout()
plt.savefig(sim_name + '_ranks.png')


## Rank figure v2
plt.figure()
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()

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
        median_subset = mean_rank[k,k2,:]
        min_subset = min_rank[k,k2,:]
        max_subset = max_rank[k,k2,:]
        plt.errorbar(sims.epsilon.unique(), median_subset, 
                     yerr=[median_subset-min_subset,max_subset-median_subset],
                     fmt='o', color=fadeColor(c1,c2,k2/7))
        
    plt.title('beta = ' + str(beta))
    plt.xlabel('epsilon')
    plt.ylabel('True lifetime rank of best policy')
    plt.gca().legend(sims.gamma.unique(),fontsize=10)
    #plt.xscale('log')
    plt.xlim((0.4,1.0))
    plt.ylim((0,5))
    #plt.hlines([1208,1192],0.4,1.0)

plt.tight_layout()
plt.savefig(sim_name + '_ranks_magnified.png')


#### CHOOSE HYPERPARAMETERS BASED ON MINIMUM MEAN RANKING
print(np.min(mean_rank))
vals = np.where(mean_rank == np.amin(mean_rank))
beta_opt = sims.beta.unique()[vals[0][0]]
gamma_opt = sims.gamma.unique()[vals[1][0]]
epsilon_opt = sims.epsilon.unique()[vals[2][0]]
print('beta_opt=',beta_opt,'; gamma_opt=',gamma_opt,'; epsilon_opt=',epsilon_opt)