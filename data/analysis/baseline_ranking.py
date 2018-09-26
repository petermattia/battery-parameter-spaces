#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naive baseline tests

Created on Wed Sep 26 08:31:51 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt

# Import all predictions
preds = np.genfromtxt('allpreds.csv', delimiter=',')
np.flipud(preds) # reverse order

# Import policies
policies = np.genfromtxt('policies_all.csv', delimiter=',')

# Preinitialize array: policies + extra column
first_lifetime = np.zeros((len(policies),5))
first_lifetime[:,0:4] = policies

# Find the first prediction for each policy
for k, pol in enumerate(first_lifetime):
    for row in preds:
        if (pol[0:4]==row[0:4]).all():
            first_lifetime[k,:] = row
            
# Sort by lifetime column
first_lifetime_sorted = np.argsort(-first_lifetime[:,4])

# Print top 10 policies
for k in np.arange(10):
    print(first_lifetime_sorted[k],first_lifetime[first_lifetime_sorted[k],:])