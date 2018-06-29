#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd 
import numpy as np 

sims = pd.read_csv('log.csv')

for name, sim_group in sims.groupby('sim mode'):
    print('Starting sim mode:', name)
    gamma_group_len = len(sim_group.gamma.unique())
    epsilon_group_len = len(sim_group.epsilon.unique())
    for var_tuple, subgroup in sim_group.groupby(['gamma','epsilon']).groups:
        print('yo')
        #print('gamma = ', gamma)
        #print('epsilon = ', epsilon)