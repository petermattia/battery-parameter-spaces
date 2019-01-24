#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:42:58 2019

@author: peter
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

do_plots = True

Crates = np.asarray([0.1,0.2,0.5,0.75,1.0,1.5,2,2.5,3])
currents = Crates * 1.1
SOCs = ['20% SOC','40% SOC','60% SOC','80% SOC']
SOCs2 = [20,40,60,80]

file_list = sorted(glob.glob('2019*.csv'))

colors = cm.viridis(np.linspace(0, 1, 4))
colors = colors[:,0:3]

overpotential = np.zeros((len(file_list),4,9))
R = np.zeros((2,4,2))
eta_graphite = np.zeros((4))

f, ax = plt.subplots(1, 2, figsize=(15,8))

for k, file in enumerate(file_list):
    # Extract data
    data = np.genfromtxt(file,skip_header=True,delimiter=',')
    test_time = data[:,1]/3600
    step_time = data[:,3]/3600
    step_idx = data[:,4]
    cycle_idx = data[:,5]
    I = data[:,6]/1.1 # C rate
    V = data[:,7]
    Qc = data[:,8]
    Qd = data[:,9]
    
    
    # Plot individual drops
    if do_plots:
        plt.figure()
        plt.plot(test_time,V,'.')
    
    for k2, idx_c in enumerate(np.arange(4)):
        if do_plots: plt.figure()
        for k3, idx_p in enumerate(np.linspace(4,36,9)):
            idx = np.intersect1d(np.where(step_idx==idx_p),
                                 np.where(cycle_idx==idx_c))
            if idx.size > 0:
                idx = np.insert(idx, 0, idx[0]-1)
                if do_plots: plt.plot(test_time[idx]-test_time[idx[0]],V[idx])
                # Calculate potential change during rest period
                overpotential[k][k2][k3] = V[idx[0]] - V[idx[-1]]
        
    
    # Plotting
    for k2 in np.arange(4):
        ax[k].plot(Crates,overpotential[k][k2],'.-',c=colors[k2,:])
    ax[k].set_xlabel('C rate')
    ax[k].set_ylabel('Overpotential (V)')
    ax[k].set_title('Cell ' + str(k+1))
    ax[k].legend(SOCs,loc='upper left',frameon=False)
    ax[k].set_xlim((0,3.5))
    ax[k].set_ylim((0,0.16))
    for k2 in np.arange(4):
        # V = I*R + V0
        if k2<2:
            R[k][k2][:] = np.polyfit(currents,overpotential[k][k2],1)
        else:
            R[k][k2][:] = np.polyfit(currents[1:],overpotential[k][k2][1:],1)
        ax[k].plot(Crates,R[k][k2][0]*currents + R[k][k2][1],'--',c=colors[k2,:])
    
f.savefig('eta_vs_I.png')

# Resistance vs soc
Rmean = np.mean(R, axis=0)
Rstd  =  np.std(R, axis=0)

# Overall average resistance
R_tot = np.mean(Rmean[:,0],axis=0)