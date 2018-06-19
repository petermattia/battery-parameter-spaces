#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter
"""

from thermalsim import thermalsim
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

LOWER_CRATE_LIM = 1.6 # C rate
UPPER_CRATE_LIM = 8   # C rate
chargetime = 10       # [=] minutes

# Import policies
policies = np.genfromtxt('policies.csv', delimiter=',')

margin = 0.8 # plotting margin

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
plt.figure() # x = CC1, y = CC2, contours = Q1
plt.set_cmap('jet_r')
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

"""
## CALCULATE POLICY LIFETIMES
lifetime = np.zeros((len(policies),1))
lifetimechange = np.zeros((len(policies),1))

for i in range(len(policies)):
    C1 = policies[i,0]
    C2 = policies[i,1]
    print('\nC1 = ' + '{:.2f}'.format(C1) + ', C2 = ' + '{:.2f}'.format(C2))
    lifetime[i] = thermalsim(C1,C2,variance=False)
    if i==0: baseline = np.asscalar(lifetime[0])
    lifetimechange[i] = 100*(lifetime[i]-baseline)/baseline
    print('Change rel. to 1-step = ' + '{:.1%}'.format(np.asscalar(lifetimechange[i])/100))
"""

## PLOT POLICIES
plt.scatter(policies[1:,0],policies[1:,1],c=lifetimechange.ravel()[1:],zorder=2)
#plt.scatter(policies[0,0],policies[0,1],marker='s',c=0,zorder=3)

# Colorbar
cbar = plt.colorbar()
plt.clim(-100,100)
cbar.set_label('% improvement over 1-step policy')

plt.savefig('contour_lifetimechanges.png', bbox_inches='tight')