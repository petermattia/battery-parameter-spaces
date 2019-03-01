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


## MODELS 
sim_model = np.genfromtxt('../policies_deg.csv',delimiter=',')[:,4]

# create dictionary where keys = string, value = [model, x axis label]
MODELS = {'sum': [np.sum(policies,axis=1), 'sum(I)'],
          'sum_sq': [np.sum(policies*policies,axis=1), 'sum(I^2)'],
          'range': [np.ptp(policies,axis=1), 'range(I)'],
          'max': [np.max(policies,axis=1), 'max(I)'],
          'var': [np.var(policies,axis=1), 'var(I)'],
          'CC1': [policies[:,0], 'CC1'],
          'CC2': [policies[:,1], 'CC2'],
          'CC3': [policies[:,2], 'CC3'],
          'CC4': [policies[:,3], 'CC4'],
          'sim': [sim_model, 'Thermal sim degradation']}

for model in MODELS:
    values = MODELS[model][0]
    xlabel = MODELS[model][1]
    
    plt.figure()
    plt.plot(values,mean,'o')
    plt.xlabel(xlabel)
    plt.ylabel('OED-estimated lifetime (cycles)')
    plt.title('ρ = {:.2}'.format(pearsonr(values,mean)[0]))
    plt.savefig('./plots/correlations/'+model+'.png', bbox_inches = 'tight')
