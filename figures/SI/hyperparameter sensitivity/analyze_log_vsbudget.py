#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

budget = 10

sims = pd.read_csv('log.csv', names=['beta','gamma','epsilon','seed',
                                     'l1','l2','l3','l4','l5','l6','l7','l8','l9','l10'])
sim_name = 'sim'

# Add column for rank
lifetimes = pd.read_csv('policies_lifetimes_' + sim_name + '.csv', 
                        names=['C1','C2','C3','C4','Lifetime'])
lifetimes_sorted = lifetimes.sort_values('Lifetime',ascending=False)
lifetimes_sorted = lifetimes_sorted.reset_index(drop=True)

lifetime_idx_array = np.zeros((len(sims),1))

for round_idx in np.arange(budget):
    key = 'l'+str(round_idx+1)
    for k,val in enumerate(sims[key]):
        lifetime_idx_array[k] = lifetimes_sorted.index[lifetimes_sorted['Lifetime']==val].tolist()[0]

    sims['mr' + str(round_idx+1)] = lifetime_idx_array


# Preinitialize
beta_len = len(sims.beta.unique())
gamma_len = len(sims.gamma.unique())
epsilon_len = len(sims.epsilon.unique())

means = np.zeros((budget,beta_len,gamma_len,epsilon_len))
std = np.zeros((budget,beta_len,gamma_len,epsilon_len))
mean_rank = np.zeros((budget,beta_len,gamma_len,epsilon_len))
min_rank = np.zeros((budget,beta_len,gamma_len,epsilon_len))
max_rank = np.zeros((budget,beta_len,gamma_len,epsilon_len))

beta_count = 0
gamma_count = 0
epsilon_count = 0

for round_idx in np.arange(budget):
    for beta, subgroup in sims.groupby('beta'):
        for gamma, subgroup2 in subgroup.groupby('gamma'):
            for epsilon, subgroup3 in subgroup2.groupby('epsilon'):
                life_key = 'l'+str(round_idx+1)
                rank_key = 'mr'+str(round_idx+1)
                
                means[round_idx,beta_count,gamma_count,epsilon_count] = subgroup3[life_key].mean()
                std[round_idx,beta_count,gamma_count,epsilon_count] = subgroup3[life_key].std()
    
                mean_rank[round_idx,beta_count,gamma_count,epsilon_count] = subgroup3[rank_key].mean()
                min_rank[round_idx,beta_count,gamma_count,epsilon_count] = subgroup3[rank_key].min()
                max_rank[round_idx,beta_count,gamma_count,epsilon_count] = subgroup3[rank_key].max()
    
                epsilon_count += 1
    
            epsilon_count = 0
            gamma_count += 1
    
        gamma_count = 0
        beta_count += 1
    
    beta_count = 0

## Rank figure
plt.figure()

ranks = np.zeros((budget,1))
opts = np.zeros((budget,3))
mins = np.zeros((budget,1))
maxs = np.zeros((budget,1))

for k in np.arange(budget):
    ranks[k] = np.min(mean_rank[k,:])
    vals = np.where(mean_rank[k,:]==np.amin(mean_rank[k,:]))
    opts[k] = [vals[0][0],vals[1][0],vals[2][0]]
    #mins = min_rank[k,opts[k][0],opts[k][1],opts[k][2]]
    #maxs = max_rank[k,opts[k][0],opts[k][1],opts[k][2]]

plt.plot(np.arange(budget)+1,ranks)
plt.title('Best performance for a given round among all hyperparameters')
plt.xlabel('Round index')
plt.ylabel('Minimum mean ranking for given round number')
plt.ylim((0,10))
plt.tight_layout()
plt.savefig('rank_vs_round.png')


#### CHOOSE HYPERPARAMETERS BASED ON MINIMUM MEAN RANKING
print(np.min(mean_rank))
vals = np.where(mean_rank == np.amin(mean_rank))
beta_opt = sims.beta.unique()[vals[0][0]]
gamma_opt = sims.gamma.unique()[vals[1][0]]
epsilon_opt = sims.epsilon.unique()[vals[2][0]]
print('beta_opt=',beta_opt,'; gamma_opt=',gamma_opt,'; epsilon_opt=',epsilon_opt)
