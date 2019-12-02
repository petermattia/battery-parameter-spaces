#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter
"""
import numpy as np
from matplotlib import rcParams

def plot_policy(CC1, CC2, CC3, ax,life_dict):
    
    tol = 0
    
    LW = 0.5
    rcParams['lines.linewidth'] = LW
    
    # Initialize axis limits
    ax.set_xlim([0,100])
    ax.set_ylim([0,10])
    
    # Add grey lines
    C1list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0]
    C2list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
    C3list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]
    
    for c1 in C1list:
        ax.plot([0,20-tol], [c1,c1], color='grey')
    for c2 in C2list:
        ax.plot([20+tol,40-tol],[c2,c2], color='grey')
    for c3 in C3list:
        ax.plot([40+tol,60-tol],[c3,c3], color='grey')
        
    # Add policy
    CC4 = 0.2/(1/6 - (0.2/CC1 + 0.2/CC2 + 0.2/CC3))
    ax.plot([0,20-tol], [CC1,CC1], linewidth=LW+0.5, color='red')
    ax.plot([20+tol,40-tol],[CC2,CC2], linewidth=LW+0.5, color='red')
    ax.plot([40+tol,60-tol],[CC3,CC3], linewidth=LW+0.5, color='red')
    ax.plot([60+tol,80-tol],[CC4,CC4], linewidth=LW+0.5, color='blue')
    
    # Add bands
    
    r_band = np.array([255,191,191])/255
    b_band = np.array([207,191,255])/255
    
    ax.axvspan(0,        20-tol/2, ymin=0.36, ymax=0.8,  facecolor=r_band, lw=0)
    ax.axvspan(20+tol/2, 40-tol/2, ymin=0.36, ymax=0.7,  facecolor=r_band, lw=0)
    ax.axvspan(40+tol/2, 60-tol/2, ymin=0.36, ymax=0.56, facecolor=r_band, lw=0)
    ax.axvspan(60+tol/2, 80-tol/2, ymin=0,    ymax=0.48, facecolor=b_band, lw=0)
    
    # Add 1C charging
    ax.plot([80,89],[1,1], linewidth=LW+0.5, color='black')
    x = np.linspace(89,100,100)
    y = np.exp(-0.5*(x-89))
    ax.plot(x,y, linewidth=LW+0.5, color='black')
    
    ax.set_xticks(np.arange(0,101,20)) 
    
    # Label policies
    name = '{0}C-{1}C-{2}C-{3:.3f}C'.format(CC1, CC2, CC3, CC4)
    ax.text(50,9,name,horizontalalignment='center')
    
    oed_str = 'CLO: {:d}'.format(int(life_dict['oed']))
    ep_str  = 'Early pred: {:d} ± {:d}'.format(int(life_dict['pred']), int(life_dict['pred_sterr']))
    final_str  = 'Final: {:d} ± {:d}'.format(int(life_dict['final']), int(life_dict['final_sterr']))
    
    ax.text(2,2.5,oed_str,horizontalalignment='left')
    ax.text(2,1.4,ep_str,horizontalalignment='left')
    ax.text(2,0.3,final_str,horizontalalignment='left')
    
    
    