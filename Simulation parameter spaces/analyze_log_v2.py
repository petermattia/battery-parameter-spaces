#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

sims = pd.read_csv('log_sigma_oed.csv')

for sim_name, sim_group in sims.groupby('sim mode'):
    print('Starting sim mode:', sim_name)
    
    # Add column for rank
    lifetimes = pd.read_csv('policies_lifetimes' + sim_name + '.csv')
    lifetimes_sorted = lifetimes.sort_values('Lifetime',ascending=False)
    lifetimes_sorted = lifetimes_sorted.reset_index(drop=True)
    
    lifetime_idx_array = np.zeros((len(sim_group),1))
    
    for k,val in enumerate(sim_group['mean lifetime']):
        lifetime_idx_array[k] = lifetimes_sorted.index[lifetimes_sorted['Lifetime']==val].tolist()[0]
        
    sim_group['mean rank'] = lifetime_idx_array
    
    # Preinitialize
    likelihood_std_group_len = len(sim_group.likelihood_std.unique())
    means = np.zeros((likelihood_std_group_len,1))
    std = np.zeros((likelihood_std_group_len,1))
    median_rank = np.zeros((likelihood_std_group_len,1))
    min_rank = np.zeros((likelihood_std_group_len,1))
    max_rank = np.zeros((likelihood_std_group_len,1))
    
    likelihood_std_count = 0
    
    for likelihood_std, subgroup in sim_group.groupby('likelihood_std'):
        print('likelihood_std = ', likelihood_std)
        means[likelihood_std_count] = subgroup['mean lifetime'].mean()
        std[likelihood_std_count] = subgroup['mean lifetime'].std()
            
        median_rank[likelihood_std_count] = subgroup['mean rank'].median()
        min_rank[likelihood_std_count] = subgroup['mean rank'].min()
        max_rank[likelihood_std_count] = subgroup['mean rank'].max()
        
        likelihood_std_count += 1
    
    ## Lifetime figure
    plt.figure()
    manager = plt.get_current_fig_manager() # Make full screen
    manager.window.showMaximized()
    
    plt.errorbar(sim_group.likelihood_std.unique(), means,yerr=std, fmt='o')
    plt.title('Sim mode = ' + sim_name)
    plt.xlabel('sigma_oed')
    plt.ylabel('True lifetime of best policy')
    plt.xscale('log')
    plt.savefig(sim_name + '.png')
    
    ## Rank figure
    plt.figure()
    manager = plt.get_current_fig_manager() # Make full screen
    manager.window.showMaximized()
    
    plt.errorbar(sim_group.likelihood_std.unique(), median_rank,
                     yerr=[median_rank-min_rank,max_rank-median_rank], fmt='o')
    plt.title('Sim mode = ' + sim_name)
    plt.xlabel('sigma_oed')
    plt.ylabel('True lifetime rank of best policy')
    plt.xscale('log')
    plt.savefig(sim_name + '_ranks.png')
    
    plt.ylim([-1,10])
    plt.savefig(sim_name + '_ranks_magnified.png')