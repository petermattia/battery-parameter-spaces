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
from matplotlib import rcParams


FS = 14
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

Crates = np.asarray([3.6,4,4.4,4.8,5.2,5.6,6,7,8])
currents = Crates * 1.1
SOCs = ['20% SOC','40% SOC','60% SOC','80% SOC']
SOCs2 = [20,40,60,80]

file_list = sorted(glob.glob('2019*.csv'))

colors = cm.viridis(np.linspace(0, 1, 4))
colors = colors[:,0:3]

overpotential = np.zeros((len(file_list),4,9))
R = np.zeros((2,4,2))

f, ax = plt.subplots(3, 2, figsize=(10,8))

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
    
    for k2, idx_c in enumerate(np.arange(4)):
        for k3, idx_p in enumerate(np.linspace(4,36,9)):
            idx = np.intersect1d(np.where(step_idx==idx_p),
                                 np.where(cycle_idx==idx_c))
            if idx.size > 0:
                idx = np.insert(idx, 0, idx[0]-1)
                # Calculate potential change during rest period
                overpotential[k][k2][k3] = V[idx[0]] - V[idx[-1]]
    
    # Plotting
    for k2 in np.arange(4):
        ax[2][k].plot(Crates,overpotential[k][k2],'.-',c=colors[k2,:])
    ax[2][k].set_xlabel('C rate')
    ax[2][k].set_ylabel('Overpotential (V)')
    ax[2][k].set_title(chr(101+k), loc='left', weight='bold')
    ax[2][k].legend(SOCs,loc='upper left',frameon=False)
    ax[2][k].set_xlim((3.5,8.1))
    ax[2][k].set_ylim((0.15,0.4))
    for k2 in np.arange(4):
        # V = I*R + V0
        if k2<2:
            R[k][k2][:] = np.polyfit(currents,overpotential[k][k2],1)
        else:
            R[k][k2][:] = np.polyfit(currents[1:],overpotential[k][k2][1:],1)
        ax[2][k].plot(Crates,R[k][k2][0]*currents + R[k][k2][1],'--',c=colors[k2,:])
        
        ## annotate
        SOC_str = '{}% SOC: η = {:0.3f}I+{:0.3f}'.format(20*(k2+1),R[k][k2][0], R[k][k2][1])
        ax[2][k].annotate(SOC_str,(8,0.22-0.02*k2),ha='right')


# Resistance vs soc
Rmean = np.mean(R, axis=0)
Rstd  =  np.std(R, axis=0)

# Overall average resistance
R_tot = np.mean(Rmean[:,0],axis=0)

plt.tight_layout()
plt.savefig('cell_char.png',bbox_inches='tight')
plt.savefig('cell_char.pdf',bbox_inches='tight',format='pdf')