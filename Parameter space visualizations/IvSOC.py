#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

#hfont = {'fontname':'Arial'}

# Initialize axis limits
plt.xlabel('State of charge (%)')
plt.ylabel('Current (C rate)')
plt.xlim([0,100])
plt.ylim([0,10])

# Add grey lines
C1list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0]
C2list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
C3list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]

for c1 in C1list:
    ax.plot([0,20], [c1,c1], linewidth=2, color='grey')
for c2 in C2list:
    ax.plot([20,40],[c2,c2], linewidth=2, color='grey')
for c3 in C3list:
    ax.plot([40,60],[c3,c3], linewidth=2, color='grey')
    
# Add example policy
c1, c2, c3, c4 = 7.0, 4.8, 5.2, 3.45
ax.plot([0,20], [c1,c1], linewidth=2, color='blue')
ax.plot([20,40],[c2,c2], linewidth=2, color='blue')
ax.plot([40,60],[c3,c3], linewidth=2, color='blue')
ax.plot([60,80],[c4,c4], linewidth=2, color='red')

# Add blue bands
ax.axvspan(0,  20, ymin=0.36, ymax=0.8,  color='blue', alpha=0.25)
ax.axvspan(20, 40, ymin=0.36, ymax=0.7,  color='blue', alpha=0.25)
ax.axvspan(40, 60, ymin=0.36, ymax=0.56, color='blue', alpha=0.25)
ax.axvspan(60, 80, ymin=0,    ymax=0.48, color='red', alpha=0.25)

# Dotted lines for SOC bands
for k in [2,4,6,8]:
    ax.plot([k*10,k*10],[0,10], linewidth=2, color='grey', linestyle=':')
    
for k in np.arange(4):
    plt.text(10+20*k,9,'C'+str(k+1), horizontalalignment='center')
plt.text(90,9,'CC-CV', horizontalalignment='center')

# Add 1C charging
ax.plot([80,89],[1,1], linewidth=2, color='black')
x = np.linspace(89,100,100)
y = np.exp(-0.5*(x-89))
ax.plot(x,y, linewidth=2, color='black')

plt.savefig('2D.png')