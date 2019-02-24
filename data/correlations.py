#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from scipy.stats import pearsonr

plt.close('all')

# IMPORT RESULTS
# Get folder path containing text files
file = glob.glob('./bounds/4_bounds.pkl')[0]
with open(file, 'rb') as infile:
    policies_temp, ub, lb, mean = pickle.load(infile)

# add cc4 to policies
policies = []
for k, pol in enumerate(policies_temp):
    cc4 = 0.2/(1/6 - (0.2/pol[0] + 0.2/pol[1] + 0.2/pol[2])) # analytical expression for cc4
    policies.append([pol[0],pol[1],pol[2],cc4])
    
policies = np.asarray(policies) # cast to numpy array


# Sum
sum_model = np.sum(policies,axis=1)
plt.figure()
plt.plot(sum_model,mean,'o')
plt.xlabel('sum(I)')
plt.ylabel('OED-estimated lifetime (cycles)')
plt.title('p = {:.2}'.format(pearsonr(sum_model,mean)[0]))
plt.savefig('./plots/correlations/sum.png', bbox_inches = 'tight')

# Sum sq
sum_sq_model = np.sum(policies*policies,axis=1)
plt.figure()
plt.plot(sum_sq_model,mean,'o')
plt.xlabel('sum(I^2)')
plt.ylabel('OED-estimated lifetime (cycles)')
plt.title('p = {:.2}'.format(pearsonr(sum_sq_model,mean)[0]))
plt.savefig('./plots/correlations/sumsq.png', bbox_inches = 'tight')

# Range
range_model = np.ptp(policies,axis=1)
plt.figure()
plt.plot(range_model,mean,'o')
plt.xlabel('range(I)')
plt.ylabel('OED-estimated lifetime (cycles)')
plt.title('p = {:.2}'.format(pearsonr(range_model,mean)[0]))
plt.savefig('./plots/correlations/range.png', bbox_inches = 'tight')

# Max
max_model = np.max(policies,axis=1)
plt.figure()
plt.plot(max_model,mean,'o')
plt.xlabel('max(I)')
plt.ylabel('OED-estimated lifetime (cycles)')
plt.title('p = {:.2}'.format(pearsonr(max_model,mean)[0]))
plt.savefig('./plots/correlations/max.png', bbox_inches = 'tight')

# Variance
var_model = np.var(policies,axis=1)
plt.figure()
plt.plot(var_model,mean,'o')
plt.xlabel('var(I)')
plt.ylabel('OED-estimated lifetime (cycles)')
plt.title('p = {:.2}'.format(pearsonr(var_model,mean)[0]))
plt.savefig('./plots/correlations/var.png', bbox_inches = 'tight')

# Thermal sim
sim_model = np.genfromtxt('../policies_deg.csv',delimiter=',')
plt.figure()
plt.plot(sim_model[:,4],mean,'o')
plt.xlabel('Thermal sim degradation')
plt.ylabel('OED-estimated lifetime (cycles)')
plt.title('p = {:.2}'.format(pearsonr(sim_model[:,4],mean)[0]))
plt.savefig('./plots/correlations/sim.png', bbox_inches = 'tight')