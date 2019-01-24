#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:31:32 2019

@author: peter
"""

import numpy as np
import csv

policies = np.asarray([[4.8,5.2,5.2], # oed1
                       [5.2,5.2,4.8], # oed2
                       [4.4,5.6,5.2], # oed3
                       [7,4.8,4.8], # Samsung
                       [8,4.4,4.4], # Notten
                       [3.6,6,5.6], # Zhang
                       [8,6,4.8], # Tesla
                       [8,7,5.2], # Lowest policy
                       [6,5.6,4.4]]) # Graphite overpotential minimization

# Repeat 9x
policies_repeated = np.transpose(np.tile(np.transpose(policies), 5))

validation_batch_file = 'validation.csv'

with open(validation_batch_file, 'w') as outfile:
	writer = csv.writer(outfile)
	writer.writerows(policies_repeated)