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
from cycler import cycler

MAX_WIDTH = 183 / 25.4 # mm -> inches
figsize = (MAX_WIDTH, 3/4*3/2*MAX_WIDTH)

rcParams['lines.linewidth'] = 1

f, ax = plt.subplots(3, 2, figsize=figsize)

for k in range(6):
    k1 = int(k/2)
    k2 = k%2
    ax[k1][k2].set_title(chr(97+k1*2+k2), loc='left', fontweight='bold')

########## a,c ##########
file = sorted(glob.glob('2018*.csv'))[1]

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
T = data[:,14]

# Pre-initialize lists
step_idx_list = np.arange(2,40,4)
n_cycles = len(step_idx_list)
t_cycle = []
V_cycle = []
I_cycle = []
Qc_cycle = []
T_cycle = []
I_leg = []

# Extract charge cycles
for k in range(n_cycles):
    step_indices = np.where(step_idx == step_idx_list[k])[0]
    t_cycle.append(test_time[step_indices] - test_time[step_indices[0]])
    V_cycle.append(V[step_indices])
    I_cycle.append(I[step_indices])
    Qc_cycle.append(Qc[step_indices] - Qc[step_indices[0]])
    T_cycle.append(T[step_indices])

    I_leg.append(str(int(np.mean(I_cycle[k]))) + 'C') # find charge C rate

cmap = plt.get_cmap('Reds')
color_cycler = (cycler(color=[cmap(1.*i/(n_cycles+3)) for i in range(3, n_cycles+3)]))
ax[0][0].set_prop_cycle(color_cycler)
ax[1][0].set_prop_cycle(color_cycler)
    
for k in range(n_cycles):
    ax[0][0].plot(Qc_cycle[k], V_cycle[k],'-')

for k in range(n_cycles):
    ax[1][0].plot(Qc_cycle[k], T_cycle[k],'-')

ax[0][0].set_yticks(np.arange(2,3.51,0.5))
ax[1][0].set_ylim([30,40])
ax[0][0].legend(I_leg,ncol=2,frameon=False)
ax[1][0].legend(I_leg,ncol=2,frameon=False)
    
ax[0][0].set_xlabel('Capacity (Ah)')
ax[0][0].set_ylabel('Voltage (V)')
ax[1][0].set_xlabel('Capacity (Ah)')
ax[1][0].set_ylabel('Can temperature (°C)')

########## b,d ##########
file = sorted(glob.glob('2018*.csv'))[0]

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
T = data[:,14]

# Pre-initialize lists
step_idx_list = np.arange(3,40,4)
n_cycles = len(step_idx_list)
t_cycle = []
V_cycle = []
I_cycle = []
Qd_cycle = []
T_cycle = []
I_leg = []

# Extract charge cycles
for k in range(n_cycles):
    step_indices = np.where(step_idx == step_idx_list[k])[0]
    t_cycle.append(test_time[step_indices] - test_time[step_indices[0]])
    V_cycle.append(V[step_indices])
    I_cycle.append(I[step_indices])
    Qd_cycle.append(Qd[step_indices] - Qd[step_indices[0]])
    T_cycle.append(T[step_indices])

for k in range(n_cycles):
    I_leg.append(str(-int(np.round(np.mean(I_cycle[k])))) + 'C') # find discharge C rate
    
cmap = plt.get_cmap('Blues')
color_cycler = (cycler(color=[cmap(1.*i/(n_cycles+3)) for i in range(3, n_cycles+3)]))
ax[0][1].set_prop_cycle(color_cycler)
ax[1][1].set_prop_cycle(color_cycler)

for k in range(n_cycles):
    ax[0][1].plot(Qd_cycle[k], V_cycle[k],'-')

for k in range(n_cycles):
    ax[1][1].plot(Qd_cycle[k], T_cycle[k],'-')

ax[0][1].legend(I_leg,ncol=2,frameon=False)
ax[1][1].legend(I_leg,ncol=2,frameon=False)
ax[0][1].set_yticks(np.arange(2,3.51,0.5))
ax[1][1].set_ylim([30,80])

ax[0][1].set_xlabel('Capacity (Ah)')
ax[0][1].set_ylabel('Voltage (V)')
ax[1][1].set_xlabel('Capacity (Ah)')
ax[1][1].set_ylabel('Can temperature (°C)')

########## e-f ##########
colors = []
colors = cm.viridis_r(np.linspace(0.15, 1, 4))
colors = colors[:,0:3]

file_list = sorted(glob.glob('2019*.csv'))

Crates = np.asarray([3.6,4,4.4,4.8,5.2,5.6,6,7,8])
currents = Crates * 1.1

overpotential = np.zeros((len(file_list),4,9))
R = np.zeros((2,4,2))

SOCs = ['20% SOC: Data', '40% SOC: Data', '60% SOC: Data', '80% SOC: Data']

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
                
    lines = []
    for k2 in np.arange(4):
        # V = I*R + V0
        if k2<2:
            R[k][k2][:] = np.polyfit(currents,overpotential[k][k2],1)
        else:
            R[k][k2][:] = np.polyfit(currents[1:], overpotential[k][k2][1:], 1)
        
        # Plotting
        label = '{}% SOC: $\eta$ = {:0.3f}$I$ + {:0.3f}'.format(20*(k2+1),R[k][k2][0], R[k][k2][1])
        lines.append(ax[2][k].scatter(Crates, overpotential[k][k2], c=colors[k2,:]))
        ax[2][k].plot(Crates,R[k][k2][0]*currents + R[k][k2][1], '--', c=colors[k2,:],
                      label=label)
        
    ax[2][k].set_xlabel('Current (C rate)')
    ax[2][k].set_ylabel('Overpotential (V)')
    ax[2][k].set_xlim((3.5,8.1))
    ax[2][k].set_ylim((0.1,0.45))
    
    leg1 = ax[2][k].legend(loc='lower right', labelspacing=0.5)
    leg2 = ax[2][k].legend(lines, SOCs, loc='upper left', labelspacing=0.5)
    ax[2][k].add_artist(leg1)

# Resistance vs SOC
Rmean = np.mean(R, axis=0)
Rstd  =  np.std(R, axis=0)

# Overall average resistance
R_tot = np.mean(Rmean[:,0],axis=0)

plt.tight_layout()
plt.savefig('cell_char.png', dpi=300)
plt.savefig('cell_char.eps', format='eps')
