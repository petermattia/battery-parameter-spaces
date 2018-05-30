#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

NOTE: REQUIRES IMAGEMAGICK. https://www.imagemagick.org/script/download.php
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.close('all')

LOWER_CRATE_LIM = 1.6 # C rate
UPPER_CRATE_LIM = 8   # C rate
chargetime = 10       # [=] minutes
margin = 0.1 # plotting margin

# Import policies
policies = np.genfromtxt('bayesgap_log.csv', delimiter=',', skip_header=0)
policies=np.array([np.array(xi) for xi in policies])

# Calculate Q1(CC1, CC2) values for contour lines
CC1 = np.arange(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,0.01)
CC2 = np.arange(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,0.01)
[X,Y] = np.meshgrid(CC1,CC2)
Q1 = (100)*(chargetime - ((60*0.8)/Y))/((60/X)-(60/Y))
Q1[np.where(Q1<0)]  = float('NaN')
Q1[np.where(Q1>80)] = float('NaN')
Q1_values = np.arange(5,76,10)

# Find number of batches
idxnewbatch = np.where(policies == 0)
batchnum = int(len(idxnewbatch[0]) / 3)
batchsize = idxnewbatch[0][0]

## Create contour plot
## Initialize plot 1: color = SOC1
fig, ax = plt.subplots() # x = CC1, y = CC2, contours = Q1
plt.set_cmap('viridis') # winter_r
plt.axis('square')
plt.xlim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
plt.ylim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
plt.xlabel('CC1',fontsize=16)
plt.ylabel('CC2',fontsize=16)
C = plt.contour(X,Y,Q1,colors='k',zorder=1)
plt.clabel(C, fontsize=10,fmt='%1.0f')
# Make full screen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
scatplot = ax.scatter([10,10],[10,10],c=[400,600],zorder=2,s=100) 


# animation function. This is called sequentially
def animate(i):
    idx1 = (batchsize+1)*(i)
    idx2 = (batchsize+1)*(i+1)-1
    
    batch = policies[idx1:idx2,:]
    scatplot.set_array(batch[:,2])
    scatplot.set_offsets(batch[:,0:2])
    plt.title('Batch ' + str(i+1))
    return (scatplot,)

## COLORBAR
cbar = plt.colorbar()
cbar.set_label('Cycle life')
cbar.set_clim(400,600)

anim = animation.FuncAnimation(fig, animate, frames=batchnum, 
                               interval=1000, blit=False)

anim.save('animation.gif', writer='imagemagick', fps=1)

#
### SAVE FIGURE
#plt.savefig('contour_lifetimes_batch2.png', bbox_inches='tight')
