#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:01:38 2018

@author: peter
This script performs hyperparamater optimization with closed_loop_oed.py
"""

import subprocess

################################################
sim_mode_list = ['hi']
gamma_list = [20.0]
epsilon_list = [0.8]
seed_list = [0,1,2,3,4,5,6,7,8,9]
likelihood_std_list = [10,20,30,50,80,100,150,200,250]
################################################

# Calculate total simulations
total_sims = len(sim_mode_list) * len(gamma_list) * len(epsilon_list) * len(likelihood_std_list) * len(seed_list)
count = 1

# Load bash script template
f = open('closed_loop_shell_script_template.sh','r')
text = f.read()
f.close()

# Loop through all combinations of hyperparameters,
# then generate, save and run new bash script
for sim_mode,gamma,epsilon,likelihood_std,seed in [(sim_mode,
                                       gamma, 
                                       epsilon,
                                       likelihood_std,
                                       seed) for sim_mode in sim_mode_list for gamma in gamma_list for epsilon in epsilon_list for likelihood_std in likelihood_std_list for seed in seed_list]:
    
    print('Starting simulation ' + str(count) + ' of ' + str(total_sims))
    print('  sim_mode =', sim_mode, ', gamma =', gamma, ', eps =', epsilon, 
          ', likelihood_std =', likelihood_std, ', seed =',seed)
    count += 1
    
    # Generate new bash script
    new_text = text.replace('sim_mode=lo','sim_mode=' + sim_mode)
    new_text = new_text.replace('gamma=1','gamma=' + str(gamma))
    new_text = new_text.replace('epsilon=0.8','epsilon=' + str(epsilon))
    new_text = new_text.replace('likelihood_std=98','likelihood_std=' + str(likelihood_std))
    new_text = new_text.replace('seed=0','seed=' + str(seed))
    
    # Save and run new bash script
    text_file = open('closed_loop_shell_script.sh', 'w')
    text_file.write(new_text)
    text_file.close()
    subprocess.call(['./closed_loop_shell_script.sh'])