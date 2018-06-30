#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:32:53 2018

@author: peter
"""

from sim_with_seed import sim
import matplotlib.pyplot as plt
import numpy as np

"""
## SIMULATED STD DEV
vals = np.zeros((1000,1))

for k in range(1000):
    vals[k] = sim(4.8,4.8,4.8,'lo',variance=True, seed=k)

plt.hist(vals)
plt.xlabel('Lifetime'),plt.ylabel('Frequency')
plt.title('sim(4.8,4.8,4.8,\'lo\',variance=True)')
print('Simulated baseline policy: mean =',str(np.mean(vals)),'std =',str(np.std(vals)))
"""

## EXP STD DEV
# From 18-Jun-218_batch8_predictions.csv in report
# Look at policies repeated 8x

# 5.3C(54%)-4C-newstructure
policy1 = [1063, 1039, 1315, 989, 935, 1158, 1156, 940]
mean1 = np.mean(policy1)
std1 = np.std(policy1)
range1 = np.max(policy1) - np.min(policy1)
print('Policy 1: mean =',str(mean1),'std =',str(std1),'range =',str(range1))
plt.figure()
plt.hist(policy1)

# 5.6C(19%)-4.6C-newstructure
policy2 = [1267, 1048, 817, 816, 1146, 1028, 1093, 796]
mean2 = np.mean(policy2)
std2 = np.std(policy2)
range2 = np.max(policy2) - np.min(policy2)
print('Policy 2: mean =',str(mean2),'std =',str(std2),'range =',str(range2))
plt.hist(policy2)

# 5.6C(36%)-4.3C-newstructure
policy3 = [1115, 828, 932, 858, 1155, 850, 923, 786]
mean3 = np.mean(policy3)
std3 = np.std(policy3)
range3 = np.max(policy3) - np.min(policy3)
print('Policy 3: mean =',str(mean3),'std =',str(std3),'range =',str(range3))
plt.hist(policy3)

# 5C(67%)-4C-newstructure
policy4 = [1009, 828, 813, 825, 1284, 1935, 1046]
mean4 = np.mean(policy4)
std4 = np.std(policy4)
print('Policy 4: mean =',str(mean4),'std =',str(std4))

# Remove outlier
policy4b = [1009, 828, 813, 825, 1284, 1046]
mean4b = np.mean(policy4b)
std4b = np.std(policy4b)
range4b = np.max(policy4b) - np.min(policy4b)
print('Policy 4b: mean =',str(mean4b),'std =',str(std4b),'range =',str(range4b))
plt.hist(policy4b)

plt.legend(['Policy 1','Policy 2','Policy 3','Policy 4'])
plt.xlabel('Lifetime'),plt.ylabel('Frequency')

## EXP SUMMARY STATS
print('Mean std =',str(np.mean([std1,std2,std3,std4b])))
print('Mean range =',str(np.mean([range1,range2,range3,range4b])))
print('Median range =',str(np.median([range1,range2,range3,range4b])))

"""
### STEVE HARRIS DATA
## Table 1 of Harris, Harris, Li (2017)
steve = [255, 301, 326, 338, 340, 341, 379, 408, 409, 430, 449, 475, 497, 509, 
         515, 518, 537, 541, 541, 560]
plt.figure()
plt.hist(steve)
"""