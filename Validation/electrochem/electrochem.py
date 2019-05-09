#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:03 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from cycler import cycler

plt.close('all')

FS = 14
LW = 3
upper_lim = 1400

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

########## LOAD DATA ##########
Qn         = np.genfromtxt('Qn.csv', delimiter=',',skip_header=0)
QV_100_10  = np.genfromtxt('QV_100_10.csv', delimiter=',',skip_header=0)
QV_EOL     = np.genfromtxt('QV_EOL.csv', delimiter=',',skip_header=0)
QV_EOL_1   = np.genfromtxt('QV_EOL_1.csv', delimiter=',',skip_header=0)

########## PLOTS ##########
fig = plt.subplots(2,2,figsize=(10,10))
ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

#ax1.set_title('a',loc='left', weight='bold')
#ax2.set_title('b',loc='left', weight='bold')

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]
#custom_cycler = (cycler(color=    [c1 , c2, c2, c2, c3, c1, c1, c1, c4]) +
#                 cycler(marker=   ['o','o','s','v','o','s','v','^','o']) +
"""
custom_cycler = (cycler(color=    [c1, c2, c2, c2, c3, c1, c1, c1, c3]) +
                 cycler(marker=   ['v','^','<','>','o','p','h','8','s']) +
                 cycler(linestyle=['' , '', '', '', '', '', '', '', '']))
"""
custom_cycler1 = (cycler(color=    [c1, c1, c1, c1, c1,
                                   c2, c2, c2, c2, c2,
                                   c2, c2, c2, c2, c2,
                                   c2, c2, c2, c2, c2,
                                   c3, c3, c3, c3, c3,
                                   c1, c1, c1, c1, c1,
                                   c1, c1, c1, c1, c1,
                                   c1, c1, c1, c1, c1,
                                   c3, c3, c3, c3, c3]) +
                 cycler(marker=   ['v', 'v', 'v', 'v', 'v',
                                   '^', '^', '^', '^', '^',
                                   '<', '<', '<', '<', '<',
                                   '>', '>', '>', '>', '>',
                                   'o', 'o', 'o', 'o', 'o',
                                   'p', 'p', 'p', 'p', 'p',
                                   'h', 'h', 'h', 'h', 'h',
                                   '8', '8', '8', '8', '8', 
                                   's', 's', 's', 's', 's']) +
                 cycler(linestyle=['', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '']))

custom_cycler2 = (cycler(color=    [c1, c1, c1, c1, c1,
                                   c2, c2, c2, c2, c2,
                                   c2, c2, c2, c2, c2,
                                   c2, c2, c2, c2, c2,
                                   c3, c3, c3, c3, c3,
                                   c1, c1, c1, c1, c1,
                                   c1, c1, c1, c1, c1,
                                   c1, c1, c1, c1, c1,
                                   c3, c3, c3, c3, c3]) +
                 cycler(marker=   ['v', 'v', 'v', 'v', 'v',
                                   '^', '^', '^', '^', '^',
                                   '<', '<', '<', '<', '<',
                                   '>', '>', '>', '>', '>',
                                   'o', 'o', 'o', 'o', 'o',
                                   'p', 'p', 'p', 'p', 'p',
                                   'h', 'h', 'h', 'h', 'h',
                                   '8', '8', '8', '8', '8', 
                                   's', 's', 's', 's', 's']))



## Q(n)
ax1.set_prop_cycle(custom_cycler1)
for row in Qn:
    idx = np.where(row>0.88)[0]
    ax1.plot(idx,row[idx],markersize=3)
#ax1.legend(validation_policies)
ax1.set_xlim([0,1500])
ax1.set_ylim([0.88,1.1])
ax1.set_xlabel('Cycle number')
ax1.set_ylabel('Discharge capacity (Ah)')

## Q(V) @ 100-10
ax2.set_prop_cycle(custom_cycler2)
V = np.linspace(3.5,2.0,num=1000)
for row in QV_100_10:
    ax2.plot(row,V,markersize=0,linewidth=2)
#ax2.legend(validation_policies)
#ax2.set_xlim([0,upper_lim])
ax2.set_ylim([2.0,3.5])
ax2.set_xlabel(r'$\mathdefault{Q_{100} - Q_{10} (Ah)}$')
ax2.set_ylabel('Voltage (V)')

## Q(V) @ EOL
ax3.set_prop_cycle(custom_cycler2)
for row in QV_EOL:
    ax3.plot(row,V,markersize=0,linewidth=2)
#ax2.legend(validation_policies)
#ax2.set_xlim([0,upper_lim])
ax3.set_ylim([2.0,3.5])
ax3.set_xlabel(r'$\mathdefault{Q_{EOL} (Ah)}$')
ax3.set_ylabel('Voltage (V)')

## Q(V) @ EOL-1
ax4.set_prop_cycle(custom_cycler2)
for row in QV_EOL_1:
    ax4.plot(row,V,markersize=0,linewidth=2)
#ax2.legend(validation_policies)
#ax2.set_xlim([0,upper_lim])
ax4.set_ylim([2.0,3.5])
ax4.set_xlabel(r'$\mathdefault{Q_{EOL} - Q_{1} (Ah)}$')
ax4.set_ylabel('Voltage (V)')

plt.tight_layout()
plt.savefig('electrochem.png',bbox_inches='tight')
plt.savefig('electrochem.pdf',bbox_inches='tight',format='pdf')
