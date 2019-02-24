#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates map of 4-step policy space with associated lifetimes

Peter Attia
Last modified February 23, 2018
"""

from thermal_sim import sim
import glob
import pickle
import numpy as np

# Import policies
file = glob.glob('./data/bounds/4_bounds.pkl')[0]
with open(file, 'rb') as infile:
    policies_temp, ub, lb, mean = pickle.load(infile)

# add cc4 to policies
policies = []
for k, pol in enumerate(policies_temp):
    cc4 = 0.2/(1/6 - (0.2/pol[0] + 0.2/pol[1] + 0.2/pol[2])) # analytical expression for cc4
    policies.append([pol[0],pol[1],pol[2],cc4])
    
policies = np.asarray(policies) # cast to numpy array


## CALCULATE POLICY LIFETIMES
lifetime = np.zeros((len(policies),1))
for i in range(len(policies)):
    C1 = policies[i,0]
    C2 = policies[i,1]
    C3 = policies[i,2]
    C4 = 0.2/(1/6 - (0.2/C1 + 0.2/C2 + 0.2/C3))
    lifetime[i] = sim(C1,C2,C3)

# Save csv with policies and lifetimes
f=open('policies_deg.csv','a')
np.savetxt(f,np.c_[policies,lifetime],delimiter=',', fmt='%1.3f')
f.close()