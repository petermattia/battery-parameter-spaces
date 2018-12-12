#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:08:33 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

plt.close('all')

# import results
data = np.genfromtxt('./pred/old_good.csv', delimiter=',')  # good cells from first run of OED
file_list = sorted(glob.glob('./pred/[0-9].csv')) # final run of OED
for k,file_path in enumerate(file_list):
    data = np.append(data,np.genfromtxt(file_path, delimiter=','),axis=0)
    
data[0][0] = 4 # import issue

# find list of policies
policies = data[:,0:3]
unique_policies = np.unique(policies,axis=0)

# count number of times repeated
count = np.zeros(unique_policies.shape[0])

for k, p in enumerate(unique_policies):
    for p2 in policies:
        if np.array_equal(p,p2):
            count[k] += 1
            
# quick plot
#plt.hist(count,bins=np.arange(6)+0.5)
#plt.xlabel('n')
#plt.ylabel('Number of policies tested n times')

# find policies repeated at least 3 times
repeated3x_policies = unique_policies[np.where(count>=3)]

# get lifetimes for those policies
repeated_lifetimes = np.zeros((repeated3x_policies.shape[0],5))

for k, p in enumerate(repeated3x_policies):
    repeat_count = 0
    
    for row in data:
        p2 = row[0:3]
        if np.array_equal(p,p2):
            repeated_lifetimes[k,repeat_count] = row[-1]
            repeat_count += 1
            
repeated_data = np.append(repeated3x_policies,repeated_lifetimes,axis=1)
np.savetxt('./testing/repeated.csv',repeated_data,delimiter=',')