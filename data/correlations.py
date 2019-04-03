#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import pickle
from scipy.stats import pearsonr, linregress

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
slope, intercept, r_value, p_value, std_err = linregress(np.log(np.sum(policies,axis=1)),np.log(mean))

# create dictionary where keys = string, value = [model, x axis label]
MODELS = {'sum': [np.sum(policies,axis=1), 'sum(I)'],
          'mean': [np.mean(policies,axis=1), 'mean(I)'],
          'sum_sq': [np.sum(policies**2,axis=1), 'sum(I^2)'],
          'mean_sq': [np.mean(policies**2,axis=1), 'mean(I^2)'],
          'rms': [np.sqrt(np.mean(policies**2,axis=1)), 'sqrt(mean(I^2))'],
          'power': [np.sum(policies,axis=1)**slope, 'sum(I^{'+str(int(slope*100)/100)+'})'],
          'range': [np.ptp(policies,axis=1), 'range(I)'],
          'max': [np.max(policies,axis=1), 'max(I)'],
          'var': [np.var(policies,axis=1), 'var(I)'],
          'sum_abs_diff': [np.sum(np.abs(policies-4.8),axis=1), 'sum(abs(I - 4.8))'],
          'sum_diff_sq': [np.sum((policies-4.8)**2,axis=1), 'sum((I - 4.8)^2)'],
          'CC1': [policies[:,0], 'CC1'],
          'CC2': [policies[:,1], 'CC2'],
          'CC3': [policies[:,2], 'CC3'],
          'CC4': [policies[:,3], 'CC4'],
          'sim': [sim_model, 'Thermal sim degradation']}

for model in MODELS:
    values = MODELS[model][0]
    xlabel = MODELS[model][1]
    
    xlabel_mod = r'$\mathdefault{'+xlabel+'}$'
    if model=='sim': xlabel_mod = xlabel
    leglabel = 'ρ = {:.2}'.format(pearsonr(values,mean)[0])
    
    plt.figure()
    rcParams['font.sans-serif'] = ['Arial']
    rcParams['font.size'] = 14
    plt.plot(values,mean,'o',label=leglabel)
    
    plt.xlabel(xlabel_mod,fontsize=14)
    plt.ylabel('OED-estimated lifetime (cycles)',fontsize=14)
    #plt.title(leglabel)
    plt.legend(loc='best',markerscale=0,frameon=False)
    plt.savefig('./plots/correlations/'+model+'.png', bbox_inches = 'tight')
    plt.savefig('./plots/correlations/'+model+'.pdf', bbox_inches = 'tight',format='pdf')
