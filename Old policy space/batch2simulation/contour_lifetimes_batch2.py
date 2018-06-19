#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter
"""

from batch2sim import batch2sim
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

LOWER_CRATE_LIM = 3 # C rate
UPPER_CRATE_LIM = 6   # C rate
chargetime = 10       # [=] minutes
margin = 0.1 # plotting margin

# Import policies
policies = np.genfromtxt('batch2policies.csv', delimiter=',', skip_header=1)

# Calculate Q1(CC1, CC2) values for contour lines
CC1 = np.arange(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,0.01)
CC2 = np.arange(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,0.01)
[X,Y] = np.meshgrid(CC1,CC2)
Q1 = (100)*(chargetime - ((60*0.8)/Y))/((60/X)-(60/Y))
Q1[np.where(Q1<0)]  = float('NaN')
Q1[np.where(Q1>80)] = float('NaN')
Q1_values = np.arange(5,76,10)

## Create contour plot
## Initialize plot 1: color = SOC1
fig = plt.figure() # x = CC1, y = CC2, contours = Q1
ax = fig.add_subplot(111)
plt.set_cmap('viridis') # winter_r
C = plt.contour(X,Y,Q1,colors='k',zorder=1)
plt.clabel(C, fontsize=10,fmt='%1.0f')
plt.title('Time to 80% = ' + str(chargetime) + ' minutes',fontsize=16)
plt.xlabel('CC1',fontsize=16)
plt.ylabel('CC2',fontsize=16)
plt.axis('square')
plt.xlim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
plt.ylim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))

# Make full screen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()

## CALCULATE POLICY LIFETIMES
lifetime = np.zeros((len(policies),1))
for i in range(len(policies)):
    C1 = policies[i,0]
    C2 = policies[i,1]
    print('\nC1 = ' + '{:.2f}'.format(C1) + ', C2 = ' + '{:.2f}'.format(C2))
    lifetime[i] = batch2sim(C1,C2,variance=False)

## PLOT POLICIES
plt.scatter(policies[:,0],policies[:,1],c=lifetime.ravel(),zorder=2,s=100)
plt.scatter(policies[0,0],policies[0,1],marker='s',c=np.asscalar(lifetime[0]),zorder=3,s=100)

## ANNOTATIONS
bbox_props = dict(fc="white", ec="k")
for i, txt in enumerate(lifetime):
    ax.annotate(txt[0],(policies[i,0],policies[i,1]),xytext=(policies[i,0]+0.05,policies[i,1]),
                bbox = bbox_props)

## COLORBAR
cbar = plt.colorbar()
plt.clim(400,600)
cbar.set_label('Cycle life')

## SAVE FIGURE
plt.savefig('contour_lifetimes_batch2.png', bbox_inches='tight')
