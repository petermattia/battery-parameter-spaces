#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:01:38 2018

@author: peter
This script performs hyperparameter optimization with closed_loop_oed.py
Note: May take a few hours to run on a personal computer
"""

import numpy as np
import subprocess

################################################
beta_list = [5.0]
gamma_list = [1.0]
epsilon_list = [0.5]
seed_list = np.arange(100)
################################################

# Calculate total simulations
total_sims = len(beta_list) * len(gamma_list) * len(epsilon_list) * len(seed_list)
count = 1

# Load bash script template
f = open('closed_loop_shell_script_template.sh','r')
text = f.read()
f.close()

# Loop through all combinations of hyperparameters,
# then generate, save and run new bash script
for beta,gamma,epsilon,seed in [(beta,
                                 gamma,
                                 epsilon,
                                 seed) for beta in beta_list for gamma in gamma_list for epsilon in epsilon_list for seed in seed_list]:
    
    print('Starting simulation ' + str(count) + ' of ' + str(total_sims))
    print('  beta =', beta, ', gamma =', gamma, ', eps =', epsilon, 
          ', seed =',seed)
    count += 1
    
    # Generate new bash script
    new_text = text.replace('beta=1','beta=' + str(beta))
    new_text = new_text.replace('gamma=1','gamma=' + str(gamma))
    new_text = new_text.replace('epsilon=0.8','epsilon=' + str(epsilon))
    #new_text = new_text.replace('likelihood_std=98','likelihood_std=' + str(likelihood_std))
    new_text = new_text.replace('seed=0','seed=' + str(seed))
    
    # Save and run new bash script
    text_file = open('closed_loop_shell_script.sh', 'w')
    text_file.write(new_text)
    text_file.close()
    subprocess.call(['./closed_loop_shell_script.sh'])