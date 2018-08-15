#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4-step policy simulator. Values are returned based on measurements 
    from oed1-oed4.
Peter Attia
Last modified August 8, 2018

INPUTS:
    -C1, C2, C3: C rate parameters (C4 is calculated)
    -variance: Include cell-to-cell variation

OUTPUT: Lifetime
"""

import random
import os
import pickle

def data_sim(C1, C2, C3, variance = True, seed=0):
    
    random.seed(seed*1000+C1*10+C2*20+C3*30) # deterministic for the same seed

    # STANDARD DEVIATION
    if variance:
        sigma = 98 # Sampling variation + prediction error. Estimated from batch8
    else:
        sigma = 0
    
    cwd = os.getcwd()
    rbf = pickle.load(open(cwd + '/rbf.pkl', 'rb'))
    
    lifetime_true = rbf(C1, C2, C3)
    
    lifetime_meas = int(random.gauss(lifetime_true, sigma))
    if lifetime_meas < 0: lifetime_meas = 1
    #print("Lifetime = " + str(lifetime_meas))
    return lifetime_meas