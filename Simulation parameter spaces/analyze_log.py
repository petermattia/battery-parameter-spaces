#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

sims = pd.read_csv('log.csv')

for sim_name, sim_group in sims.groupby('sim mode'):
    print('Starting sim mode:', sim_name)
    gamma_group_len = len(sim_group.gamma.unique())
    epsilon_group_len = len(sim_group.epsilon.unique())
    means = np.zeros((gamma_group_len,epsilon_group_len))
    std = np.zeros((gamma_group_len,epsilon_group_len))
    
    gamma_count = 0
    
    for gamma, gamma_group in sim_group.groupby('gamma'):
        epsilon_count = 0
        
        for epsilon, subgroup in gamma_group.groupby('epsilon'):
            print('gamma = ', gamma, 'epsilon = ', epsilon)
            means[gamma_count, epsilon_count] = subgroup['mean lifetime'].mean()
            std[gamma_count, epsilon_count] = subgroup['mean lifetime'].std()
            
            epsilon_count += 1
        
        gamma_count += 1
    
    plt.figure()
    for k in range(epsilon_group_len):
        plt.errorbar(sim_group.gamma.unique(), means[:,k],yerr=std[:,k], fmt='o')
    leg = []
    for count, eps in enumerate(sim_group.epsilon.unique()):
        leg.append("epsilon = " + str(eps))
    plt.legend(leg)
    plt.title('Sim mode = ' + sim_name)
    plt.xlabel('Gamma')
    plt.ylabel('True lifetime of best policy')
    plt.savefig(sim_name + '.png')