#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:06:18 2018

@author: peter
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

num_batches=5

font_size = 20

data = []
file_list = sorted(glob.glob('./pred/[0-9].csv'))
for k,file_path in enumerate(file_list):
    data.append(np.genfromtxt(file_path, delimiter=','))
    
policies = np.genfromtxt('policies_all.csv', delimiter=',')

isTested = np.zeros(len(policies))

for k, pol in enumerate(policies):
    for batch in data:
        for row in batch:
            if (pol==row[0:4]).all():
                isTested[k] += 1

print('Total tests:', sum(isTested))

pol_reps = np.zeros(num_batches)

for k in np.arange(num_batches):
    pol_reps[k] = sum(isTested==k)
    print('Policies tested', k, 'time(s):', int(pol_reps[k]))
    
plt.bar(np.arange(num_batches),pol_reps,tick_label = ['0','1','2','3','4'],align='center')
plt.xlabel('n',fontsize=font_size+2)
plt.ylabel('Number of policies tested n times',fontsize=font_size+2)
plt.savefig('plots/uniqueness.png',bbox_inches='tight')
plt.savefig('plots/uniqueness.pdf',bbox_inches='tight')