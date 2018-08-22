#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:06:18 2018

@author: peter
"""

import glob
import numpy as np

data = []
file_list = sorted(glob.glob('./pred/[0-9].csv'))
for k,file_path in enumerate(file_list):
    data.append(np.genfromtxt(file_path, delimiter=','))
    
policies = np.genfromtxt('policies_all.csv', delimiter=',')

isTested = np.zeros((len(policies),1))

for k, pol in enumerate(policies):
    for batch in data:
        for row in batch:
            if (pol==row[0:4]).all():
                isTested[k] += 1

print('Total tests:', sum(isTested)[0])

for k in range(0,10):
    print('Policies tested', k, 'time(s):', sum(isTested==k))