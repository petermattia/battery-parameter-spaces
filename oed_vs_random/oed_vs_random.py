#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:44:08 2019

@author: peter
"""

import numpy as np
import pickle
import glob

np.random.seed(0)

# Load data
file = glob.glob('../data/bounds/4_bounds.pkl')[0]
with open(file, 'rb') as infile:
    param_space, ub, lb, mean = pickle.load(infile)
        
# Create data frame
data = np.zeros((224,4))
data[:,0:3] = param_space
data[:,3] = mean

# Subselect policies
a = np.where(data[:,0]==data[:,1])
b = np.where(data[:,2]==4.8)
pol_ind = np.intersect1d(a,b)
data = data[pol_ind,:]

# Save policies
np.savetxt('subspace.csv',data,delimiter=',',fmt='%.1f')

# Randomly generate lifetimes
arr_size = (12,4)
means_array = np.tile(mean[pol_ind],arr_size)[:,0:4]
scales = [126, 164] # stdev without and with early prediction

# Add noise
noise1 = np.random.normal(loc=0.0, scale=scales[0], size=arr_size)
result1 = means_array + noise1

noise2 = np.random.normal(loc=0.0, scale=scales[1], size=arr_size)
result2 = means_array + noise2

# Save lifetimes
data1 = np.zeros((4,15))
data1[:,0:3] = param_space[pol_ind]
data1[:,3:15] = np.transpose(result1)
np.savetxt('subspace_lifetimes_std126.csv',data1,delimiter=',',fmt='%.1f')

data2 = np.zeros((4,15))
data2[:,0:3] = param_space[pol_ind]
data2[:,3:15] = np.transpose(result2)
np.savetxt('subspace_lifetimes_std164.csv',data2,delimiter=',',fmt='%.1f')
