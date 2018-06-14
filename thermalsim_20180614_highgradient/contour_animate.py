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
import operator

def animate(best_arm_params=[0,0]):
    plt.close('all')
    
    LOWER_CRATE_LIM = 1.6 # C rate
    UPPER_CRATE_LIM = 8   # C rate
    chargetime = 10       # [=] minutes
    margin = 0.1 # plotting margin
    
    # Import results
    policies = np.genfromtxt('bayesgap_log.csv', delimiter=',', skip_header=0)
    policies=np.array([np.array(xi) for xi in policies])
    
    # Calculate Q1(CC1, CC2) values for contour lines
    CC1 = np.arange(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,0.01)
    CC2 = np.arange(LOWER_CRATE_LIM-margin,UPPER_CRATE_LIM + margin,0.01)
    [X,Y] = np.meshgrid(CC1,CC2)
    Q1 = (100)*(chargetime - ((60*0.8)/Y))/((60/X)-(60/Y))
    Q1[np.where(Q1<0)]  = float('NaN')
    Q1[np.where(Q1>80)] = float('NaN')
    
    # Find number of batches
    idxnewbatch = np.where(policies == 0)
    batchnum = int(len(idxnewbatch[0]) / 3)
    batchsize = idxnewbatch[0][0]
    
    # Min and max lifetime
    min_lifetime = min(i for i in policies[:,2] if i > 0)
    max_lifetime = max(policies[:,2])
    
    ## CREATE CONTOUR PLOT
    # SETTINGS
    fig, ax = plt.subplots() # x = CC1, y = CC2, contours = Q1
    plt.set_cmap('viridis') # winter_r
    plt.axis('square') 
    plt.rcParams.update({'font.size': 16})
    manager = plt.get_current_fig_manager() # Make full screen
    manager.window.showMaximized()
    
    # LABELS AND AXES LIMITS
    plt.xlim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
    plt.ylim((LOWER_CRATE_LIM-margin, UPPER_CRATE_LIM+margin))
    plt.xlabel('C1',fontsize=16)
    plt.ylabel('C2',fontsize=16)
    
    # CONTOUR LINES
    C = plt.contour(X,Y,Q1,colors='k',zorder=1)
    plt.clabel(C, fontsize=12,fmt='%1.0f')
    
    # INITIALIZE SCATTER PLOT
    scatplot = plt.scatter([10,10],[10,10],c=[min_lifetime,max_lifetime], 
                          zorder=2,s=100,vmin=min_lifetime,vmax=max_lifetime) 
    
    # ANIMATION FUNCTION. This is called sequentially
    def animate(i):
        idx1 = (batchsize+1)*(i)
        idx2 = (batchsize+1)*(i+1)-1
        
        batch = policies[idx1:idx2,:]
        scatplot.set_array(batch[:,2])
        scatplot.set_offsets(batch[:,0:2])
        plt.title('Batch ' + str(i+1))
        return (scatplot,)
    
    # COLORBAR
    cbar = plt.colorbar()
    cbar.set_label('Cycle life')
    cbar.set_clim(min_lifetime,max_lifetime)
    
    # ADD TEXT LABELS
    # LABEL FOR TRUE BEST ARM
    try:
        policies_lifetimes = np.genfromtxt('highgradient.csv', delimiter=',', skip_header=0)
        index, max_lifetime = max(enumerate(policies_lifetimes[:,2]), key=operator.itemgetter(1))
        plt.gcf().text(0.05, 0.9, "True best arm: (" + str(policies_lifetimes[index,0]) + ", " + \
               str(policies_lifetimes[index,1]) + ")", fontsize=14)
        plt.gcf().text(0.05, 0.85, " Lifetime=" + str(policies_lifetimes[index,2]),
                fontsize=14)
    except:
        pass
    
    # LABEL FOR ESTIMATED BEST ARM
    if best_arm_params[0]:
        plt.gcf().text(0.05, 0.7, "Estimated best arm: (" + str(best_arm_params[0]) + ", " + \
               str(best_arm_params[1]) + ")", fontsize=14)
    
    
    ## SAVE ANIMATION
    anim = animation.FuncAnimation(fig, animate, frames=batchnum, 
                                   interval=1000, blit=False)
    
    anim.save('animation.gif', writer='imagemagick', fps=1)