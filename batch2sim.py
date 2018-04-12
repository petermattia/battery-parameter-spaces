#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 2018

@author: peter
"""

import random
import numpy as np

def batch2sim(C1, C2, variance = True):
    
    # STANDARD DEVIATION
    if variance:
        sigma = np.std([495, 461, 471]); # batch2 baseline policy
    else:
        sigma = 0;
        
    # LOAD BATCH2
    policies = np.genfromtxt('batch2.csv',
				delimiter=',', skip_header=1)
    
    # FIND LIFETIME FROM TABLE
    idxC1 = np.where(policies == C1)
    idxC2 = np.where(policies == C2)
    idx_policy = np.intersect1d(idxC1[0],idxC2[0])
    if len(idx_policy) == 0:
        print('No policy found')
        return np.nan
    elif len(idx_policy) == 2:
        if policies[idx_policy[0]][0] == C1:
            idx_policy = idx_policy[0]
        else:
            idx_policy = idx_policy[1]
        lifetime = policies[idx_policy][2];
    else:
        lifetime = policies[idx_policy[0]][2];
    
    # ADD NOISE
    lifetime_meas = int(random.gauss(lifetime, sigma))
    
    # PRINT
    print('True lifetime = ' + str(lifetime))
    print('Measured lifetime = ' + str(lifetime_meas))
    
    return lifetime_meas