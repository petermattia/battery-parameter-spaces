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
import pandas as pd

def data_sim(C1, C2, C3, variance = True):

    # STANDARD DEVIATION
    if variance:
        sigma = 98 # Sampling variation + prediction error. Estimated from batch8
    else:
        sigma = 0
    
    cwd = os.getcwd()
    data = pickle.load(open(cwd + '/mid_oed_tests/data_1to4.pkl', 'rb'))
    df = pd.DataFrame(data, columns=['C1', 'C2', 'C3', 'C4', 'Lifetime'])
    
    loc = df.loc[(df['C1'] == C1) & (df['C2'] == C2) & (df['C3'] == C3)]
    lifetime_true = loc['Lifetime'].values[0]
    
    lifetime_meas = int(random.gauss(lifetime_true, sigma))
    if lifetime_meas < 0: lifetime_meas = 1
    #print("Lifetime = " + str(lifetime_meas))
    return lifetime_meas