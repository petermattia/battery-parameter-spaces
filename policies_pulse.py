#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policies.py is a script that generates all policies within a
2-step policy space based on your specifications.
It saves a file called policies.csv to the current directory,
where each row is a charging policy

Created on Tue Feb 20 16:04:53 2018

@author: peter
"""

import numpy as np
import contour_points

##############################################################################

# PARAMETERS TO CREATE POLICY SPACE
LOWER_CRATE_LIM = 3  # C rate, lower cutoff
UPPER_CRATE_LIM = 6  # C rate, upper cutoff
LOWER_SOC1_LIM  = 10   # [%], lower SOC1 cutoff
UPPER_SOC1_LIM  = 69   # [%], upper SOC1 cutoff
OFFSET          = 0.2  # initial distance from baseline policy
PULSE           = 8    # Pulse current
PULSE_WIDTH     = 10   # Pulse width, in SOC
chargetime      = 10-60/PULSE*(PULSE_WIDTH/100)  # [=] minutes
FINAL_CUTOFF    = 70   # SOC cutoff

SET_STEP        = True  # Set either a step size or a density of points per line cut
STEP_SIZE       = 0.3;  # Step size, in units of C rate
DENSITY         = 5     # Points per line cut

##############################################################################

# Use standard conditions to keep C1 and C2 points consistent
FINAL_CUTOFF    = 80  # SOC cutoff
chargetime = 10
one_step = 4.5 

# C1 > C2
if SET_STEP:
    C1grida = np.arange(one_step + OFFSET,UPPER_CRATE_LIM,STEP_SIZE)
    C2grida = np.arange(one_step - OFFSET,LOWER_CRATE_LIM - 0.1,-STEP_SIZE)
else:
    C1grida = np.linspace(one_step + OFFSET,UPPER_CRATE_LIM,DENSITY)
    C2grida = np.linspace(LOWER_CRATE_LIM,one_step - OFFSET,DENSITY)
X2a, Y2a = np.meshgrid(C1grida, C2grida)

# C1 < C2
if SET_STEP:
    C1gridb = np.arange(one_step - OFFSET, LOWER_CRATE_LIM - 0.1, -STEP_SIZE)
    C2gridb = np.arange(one_step + OFFSET, UPPER_CRATE_LIM + 0.1, STEP_SIZE)
else:
    C1gridb = np.linspace(LOWER_CRATE_LIM, one_step - OFFSET, DENSITY)
    C2gridb = np.linspace(one_step + OFFSET, UPPER_CRATE_LIM, DENSITY)
X2b, Y2b = np.meshgrid(C1gridb, C2gridb)

FINAL_CUTOFF    = 70  # SOC cutoff
chargetime      = 10-60/PULSE*(PULSE_WIDTH/100)  # [=] minutes
one_step = 60*(FINAL_CUTOFF/100)/chargetime

# Remove bad policies: C1 > C2
for i in np.arange(0,X2a.shape[0]):
    for j in np.arange(0,X2a.shape[1]):
        C1 = X2a[i,j]
        C2 = Y2a[i,j]
        SOC1 = 100 * ( chargetime - (60*FINAL_CUTOFF/100/C2) ) / (60/C1 - 60/C2)
        print(str(C1) + ', ' + str(C2) + ': ' + str(SOC1))
        # removes policies that are basically 1-step
        if SOC1 < LOWER_SOC1_LIM or SOC1 > UPPER_SOC1_LIM:
            X2a[i,j] = float('NaN')
            Y2a[i,j] = float('NaN')

# Remove bad policies: C1 < C2
for i in np.arange(0,X2b.shape[0]):
    for j in np.arange(0,X2b.shape[1]):
        C1 = X2b[i,j]
        C2 = Y2b[i,j]
        SOC1 = 100 * ( chargetime - (60*FINAL_CUTOFF/100/C2) ) / (60/C1 - 60/C2)
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
np.savetxt('policies_withpulse.csv',np.c_[X2,Y2],delimiter=',', fmt='%1.3f')

contour_points.plot_contour(LOWER_CRATE_LIM, UPPER_CRATE_LIM, chargetime, FINAL_CUTOFF, len(X2), PULSE)
