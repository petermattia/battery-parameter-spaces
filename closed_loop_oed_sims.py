#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:01:38 2018

@author: peter
"""
import subprocess

################################################
"""
sim_mode_list = ['lo','med','hi']
gamma_list = [0.01,0.1,0.2,0.5,1.0]
epsilon_list = [0.7,0.75,0.8,0.85,0.9]
seed_list = [0,1,2,3,4,5,6,7,8,9]
"""
sim_mode_list = ['lo']
gamma_list = [0.01]
epsilon_list = [0.7,0.8]
seed_list = [1]
################################################

# Load bash script template
f = open('closed_loop_shell_script_template.sh','r')
text = f.read()
f.close()

# Loop through all combinations of hyperparameters,
# then generate, save and run new bash script
for sim_mode, gamma, epsilon,seed in [(sim_mode,
                                       gamma, 
                                       epsilon,
                                       seed) for sim_mode in sim_mode_list for gamma in gamma_list for epsilon in epsilon_list for seed in seed_list]:
    print('Now running: ', sim_mode, gamma, epsilon, seed)
    
    # Generate new bash script
    new_text = text.replace('sim_mode=lo','sim_mode=' + sim_mode)
    new_text = new_text.replace('gamma=1','gamma=' + str(gamma))
    new_text = new_text.replace('epsilon=0.8','epsilon=' + str(epsilon))
    new_text = new_text.replace('seed=0','seed=' + str(seed))
    
    # Save and run new bash script
    text_file = open('closed_loop_shell_script.sh', 'w')
    text_file.write(new_text)
    text_file.close()
    subprocess.call(['./closed_loop_shell_script.sh'])