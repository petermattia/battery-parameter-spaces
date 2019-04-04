#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def plot_policy(CC1, CC2, CC3):
    fig, ax = plt.subplots()
    
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    #rcParams['font.sans-serif'] = ['Arial', 'Tahoma', 'DejaVu Sans',
    #                               'Lucida Grande', 'Verdana']
    
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
    CC4 = 0.2/(1/6 - (0.2/CC1 + 0.2/CC2 + 0.2/CC3))
    ax.plot([0,20], [CC1,CC1], linewidth=2, color='red')
    ax.plot([20,40],[CC2,CC2], linewidth=2, color='red')
    ax.plot([40,60],[CC3,CC3], linewidth=2, color='red')
    ax.plot([60,80],[CC4,CC4], linewidth=2, color='blue')
    
    # Add bands
    ax.axvspan(0,  20, ymin=0.36, ymax=0.8,  color='red', alpha=0.25)
    ax.axvspan(20, 40, ymin=0.36, ymax=0.7,  color='red', alpha=0.25)
    ax.axvspan(40, 60, ymin=0.36, ymax=0.56, color='red', alpha=0.25)
    ax.axvspan(60, 80, ymin=0,    ymax=0.48, color='blue', alpha=0.25)
    
    # Dotted lines for SOC bands
    for k in [2,4,6,8]:
        ax.plot([k*10,k*10],[0,10], linewidth=2, color='grey', linestyle=':')
        
    for k in np.arange(4):
        plt.text(10+20*k,9,'CC'+str(k+1), horizontalalignment='center')
    plt.text(90,9,'CC5-CV1', horizontalalignment='center')
    
    # Add 1C charging
    ax.plot([80,89],[1,1], linewidth=2, color='black')
    x = np.linspace(89,100,100)
    y = np.exp(-0.5*(x-89))
    ax.plot(x,y, linewidth=2, color='black')
    
    name = '{0}-{1}-{2}-{3:.3f}'.format(CC1, CC2, CC3, CC4)
    plt.title(name)
    
    plt.savefig(name+'.png')
    #plt.savefig(name+'pdf', format='pdf')