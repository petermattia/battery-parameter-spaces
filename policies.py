#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policies.py generates 2-step polices based 

Created on Tue Feb 20 16:04:53 2018

@author: peter
"""

import numpy as np
import contour_points

##############################################################################

# PARAMETERS TO CREATE POLICY SPACE
LOWER_CRATE_LIM = 1.6 # C rate, lower cutoff
UPPER_CRATE_LIM = 8   # C rate, upper cutoff
LOWER_SOC1_LIM  = 10  # [%], lower SOC1 cutoff
UPPER_SOC1_LIM  = 70  # [%], upper SOC1 cutoff
DENSITY         = 11  # Points per line cut
STEP            = 0.2 # initial distance from baseline policy
chargetime      = 10  # [=] minutes

##############################################################################

# Find 1-step C rate for given charge time
one_step = 60*0.8/chargetime;

# C1 > C2
C1grida = np.linspace(one_step + STEP,UPPER_CRATE_LIM,DENSITY)
C2grida = np.linspace(LOWER_CRATE_LIM,one_step - STEP,DENSITY)
X2a, Y2a = np.meshgrid(C1grida, C2grida)

# Remove bad policies
for i in np.arange(0,len(X2a)):
    for j in np.arange(0,len(Y2a)):
        C1 = X2a[i,j]
        C2 = Y2a[i,j]
        SOC1 = 100 * ( chargetime - (60*0.8/C2) ) / (60/C1 - 60/C2)
        # removes policies that are basically 1-step
        if SOC1 < LOWER_SOC1_LIM or SOC1 > UPPER_SOC1_LIM: 
            X2a[i,j] = float('NaN')
            Y2a[i,j] = float('NaN')

# C1 < C2
C1gridb = np.linspace(LOWER_CRATE_LIM,one_step - STEP,DENSITY)
C2gridb = np.linspace(one_step + STEP,UPPER_CRATE_LIM,DENSITY)
X2b, Y2b = np.meshgrid(C1gridb, C2gridb)

# Remove bad policies
for i in np.arange(0,len(X2b)):
    for j in np.arange(0,len(Y2b)):
        C1 = X2b[i,j]
        C2 = Y2b[i,j]
        SOC1 = 100 * ( chargetime - (60*0.8/C2) ) / (60/C1 - 60/C2)
        # removes policies that are basically 1-step
        if SOC1 < LOWER_SOC1_LIM or SOC1 > UPPER_SOC1_LIM:
            X2b[i,j] = float('NaN')
            Y2b[i,j] = float('NaN')

## Unravel, merge, clean, and add baseline policy
X2 = np.concatenate((X2a.ravel(), X2b.ravel()))
Y2 = np.concatenate((Y2a.ravel(), Y2b.ravel()))
X2 = X2[~np.isnan(X2)]
Y2 = Y2[~np.isnan(Y2)]
X2 = np.insert(X2, 0, one_step)
Y2 = np.insert(Y2, 0, one_step)

print(str(len(X2)) + " total policies")

# Save policies
np.savetxt('policies.csv',np.c_[X2,Y2],delimiter=',', fmt='%1.3f')

contour_points.plot_contour(LOWER_CRATE_LIM, UPPER_CRATE_LIM, chargetime, len(X2))