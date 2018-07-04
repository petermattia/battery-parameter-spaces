#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

sims = pd.read_csv('log_hi.csv')

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
    gamma_group_len = len(sim_group.gamma.unique())
    epsilon_group_len = len(sim_group.epsilon.unique())
    means = np.zeros((gamma_group_len,epsilon_group_len))
    std = np.zeros((gamma_group_len,epsilon_group_len))
    median_rank = np.zeros((gamma_group_len,epsilon_group_len))
    min_rank = np.zeros((gamma_group_len,epsilon_group_len))
    max_rank = np.zeros((gamma_group_len,epsilon_group_len))
    
    gamma_count = 0
    
    for gamma, gamma_group in sim_group.groupby('gamma'):
        epsilon_count = 0
        
        for epsilon, subgroup in gamma_group.groupby('epsilon'):
            print('gamma = ', gamma, 'epsilon = ', epsilon)
            means[gamma_count, epsilon_count] = subgroup['mean lifetime'].mean()
            std[gamma_count, epsilon_count] = subgroup['mean lifetime'].std()
            
            median_rank[gamma_count, epsilon_count] = subgroup['mean rank'].median()
            min_rank[gamma_count, epsilon_count] = subgroup['mean rank'].min()
            max_rank[gamma_count, epsilon_count] = subgroup['mean rank'].max()
            
            epsilon_count += 1
        
        gamma_count += 1
    
    ## Lifetime figure
    plt.figure()
    manager = plt.get_current_fig_manager() # Make full screen
    manager.window.showMaximized()
    
    for k in range(epsilon_group_len):
        plt.errorbar(sim_group.gamma.unique(), means[:,k],yerr=std[:,k], fmt='o')
    leg = []
    for count, eps in enumerate(sim_group.epsilon.unique()):
        leg.append("epsilon = " + str(eps))
    plt.legend(leg)
    plt.title('Sim mode = ' + sim_name)
    plt.xlabel('Gamma')
    plt.ylabel('True lifetime of best policy')
    plt.xscale('log')
    plt.savefig(sim_name + '.png')
    
    
    ## Rank figure
    plt.figure()
    manager = plt.get_current_fig_manager() # Make full screen
    manager.window.showMaximized()
    
    for k in range(epsilon_group_len):
        plt.errorbar(sim_group.gamma.unique(), median_rank[:,k],
                     yerr=[median_rank[:,k]-min_rank[:,k],
                           max_rank[:,k]-median_rank[:,k]], fmt='o')
    leg = []
    for count, eps in enumerate(sim_group.epsilon.unique()):
        leg.append("epsilon = " + str(eps))
    plt.legend(leg)
    plt.title('Sim mode = ' + sim_name)
    plt.xlabel('Gamma')
    plt.ylabel('True lifetime rank of best policy')
    plt.xscale('log')
    plt.savefig(sim_name + '_ranks.png')
    
    plt.ylim([-1,10])
    plt.savefig(sim_name + '_ranks_magnified.png')