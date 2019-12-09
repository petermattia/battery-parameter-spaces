#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.gaussian_process import GaussianProcessRegressor

plt.close('all')

MAX_WIDTH = 183 / 25.4 # mm -> inches
figsize=(MAX_WIDTH, 5/6*MAX_WIDTH)

plt.figure(figsize=figsize)

ax0 = plt.subplot2grid((3, 4), (0, 0))
ax0b = plt.subplot2grid((3, 4), (0, 1))
ax0c = plt.subplot2grid((3, 4), (0, 2))
ax1 = plt.subplot2grid((3, 4), (1, 0))
ax2 = plt.subplot2grid((3, 4), (1, 1))
ax2b = plt.subplot2grid((3, 4), (1, 2))
ax4 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
ax5 = plt.subplot2grid((3, 4), (2, 2), colspan=2)

ax0.set_title('a', loc='left', weight='bold')
ax4.set_title('b', loc='left', weight='bold')

## plot
c1 = u'#348ABD'
c2 = u'#A60628'

## Early prediction

# load data
files = glob.glob('*.csv')[::-1] # cells from validation batch
data = []
for f in files:
    data.append(np.genfromtxt(f,delimiter=',',skip_header=1))

# Discharge capacity
ax0.set_prop_cycle(plt.style.library['bmh']['axes.prop_cycle'])
ax0.plot(np.arange(100)+1,data[0][:100,0], label='Cell A')
ax0.plot(np.arange(100)+1,data[1][:100,0], label='Cell B')
ax0.set_xlabel('Cycle number')
ax0.set_ylabel('Discharge capacity (Ah)')
ax0.set_xlim((0,100))
ax0.annotate("Cell A", xy=(70, 1.048), xytext=(70, 1.0494), color=c1)
ax0.annotate("Cell B", xy=(70, 1.054), xytext=(70, 1.0542),  color=c2)

ax0b.text(-0.15, 0.35, '• ' + 'max(' + r'$Q_{discharge}$'+ ')' + r'$ - Q_{discharge,2}$')
ax0b.text(-0.15, 0.55, '• ' + r'$Q_{discharge,2}$')
ax0b.text(-0.15, 0.70, 'Capacity features\n(from first 100 cycles)')
ax0b.axis('off')

ax0c.annotate("", xy=(0.6, 0.25), xytext=(-0.1, 0.4), arrowprops=dict(arrowstyle="->"))
ax0c.axis('off')

# Voltage
QV100a = data[0][:,1][~np.isnan(data[0][:,1])]
QV10a  = data[0][:,2][~np.isnan(data[0][:,2])]
QV100b = data[1][:,1][~np.isnan(data[1][:,1])]
QV10b  = data[1][:,2][~np.isnan(data[1][:,2])]
ax1.plot(QV10b ,np.linspace(3.5,2.0,1000),'k', label='Cycle 10')
ax1.plot(QV100b,np.linspace(3.5,2.0,1000),'--k',label='Cycle 100')
ax1.set_xlabel('Discharge capacity (Ah)')
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim((2.0,3.5))
ax1.legend()
ax1.set_xlim((0,1.1))

# Voltage difference
ax2.set_prop_cycle(plt.style.library['bmh']['axes.prop_cycle'])
ax2.plot(QV100a - QV10a,np.linspace(3.5,2.0,1000), label='Cell A')
ax2.plot(QV100b - QV10b,np.linspace(3.5,2.0,1000), label='Cell B')
ax2.axvline(0,color='k',ls=':')
ax2.set_xlabel(r'$Q_{100}-Q_{10}$'+' (Ah)')
ax2.set_ylabel('Voltage (V)')
ax2.set_ylim((2.0,3.5))
ax2.annotate("Cell A", xy=(0.002, 3.22), xytext=(0.002, 3.22), color=c1)
ax2.annotate("Cell B", xy=(-0.02, 3.22), xytext=(-0.02, 3.22), color=c2)
ax2.set_xlim((-0.045, 0.025))

ax2b.text(-0.15, 0.15, '• ' + 'log' + r'$_{10}(|$'+ 'skew' + r'$(\Delta Q_{100-10}(V))|)$')
ax2b.text(-0.15, 0.35, '• ' + 'log' + r'$_{10}(|$'+ 'var' + r'$(\Delta Q_{100-10}(V))|)$')
ax2b.text(-0.15, 0.55, '• ' + 'log' + r'$_{10}(|$'+ 'min' + r'$(\Delta Q_{100-10}(V))|)$')
ax2b.text(-0.15, 0.70, 'Voltage features\n(from first 100 cycles)')
ax2b.axis('off')
ax2b.annotate("", xy=(0.9, 1.0), xytext=(0.65, 0.8), arrowprops=dict(arrowstyle="->"))



### GP plots, adapted from:
# https://scikit-learn.org/0.17/auto_examples/gaussian_process/plot_gp_regression.html

# Observations and noise
def plot_GP(X, ax, label_align):
    def f(x):
        return 10 * x * np.sin(x) + 1200
    
    #X = np.concatenate((np.linspace(0.1, 4.9, 5),np.linspace(5.0, 9.9, 10)))
    X = np.atleast_2d(X).T
    y = f(X).ravel()
    dy = 20.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    
    # Instanciate a Gaussian Process model
    gp = GaussianProcessRegressor(normalize_y=True)
    
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    
    # Make the prediction on the meshed x-axis
    y_pred, std = gp.predict(x, return_std=True)
    std *= 100
    
    mse = np.mean((y_pred-f(X))**2)
    print(mse)
    
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    ax.plot(x, f(x), ':', color='tab:red', label='True cycle life')
    ax.errorbar(X.ravel(), y, dy, fmt='.', color='tab:purple', markersize=7,
                label=u'Observed data')
    ax.plot(x, y_pred, '-', color='tab:blue', label='Estimated\ncycle life')
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.9600 * std,
                           (y_pred + 1.9600 * std)[::-1]]),
            fc=np.array([151,199,226])/255, ec='None')
    
    ub = y_pred + 1.9600 * std
    ax.plot(x, ub, '--', label='Upper confidence bound\n(UCB)')
    max_acq_idx = np.argmax(ub)
    
    ax.plot(x[max_acq_idx], ub[max_acq_idx],'v', markersize=4)
    ax.annotate('max(UCB)', (x[max_acq_idx], ub[max_acq_idx]),
                xytext=(0, 5),textcoords="offset points",ha=label_align)
    
    ax.set_xlabel(r'Charging protocol parameter (e.g. CC1)')
    ax.set_ylabel('Cycle life (cycles)')
    ax.set_xlim((0,10))
    ax.set_ylim((1000,1801))
    leg = ax.legend(ncol = 2, frameon=False, loc='upper center')
    
    # Move legend up
    # https://stackoverflow.com/questions/23238041/move-and-resize-legends-box-in-matplotlib
    
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    
    # Change to location of the legend. 
    yOffset = 0.04
    bb.y0 += yOffset
    bb.y1 += yOffset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)
    
    return x[max_acq_idx][0] #return new point

# GP before
np.random.seed(3)
X = 10*np.random.random(3)
new_x = plot_GP(X,ax4, 'center')
# GP after
X = np.append(X, new_x)
new_x2 = plot_GP(X,ax5, 'left')
ax5.annotate('new\ndata', (new_x, 1400), 
             xytext=(0, 5),textcoords="offset points",ha='center')
ax5.plot(new_x, 1300,'kv', markersize=3) # arrow base
ax5.plot((new_x, new_x), (1310, 1420),'k-') # arrow stem

plt.tight_layout()

# Capacity vs cycles
ax3 = plt.axes([0.75, 0.65, 0.2, 0.25])
ax3.set_prop_cycle(plt.style.library['bmh']['axes.prop_cycle'])
idx1 = np.where(data[0][:,0]<0.88)[0][0]
idx2 = np.where(data[1][:,0]<0.88)[0][0]
ax3.plot(data[0][:idx1,0], ls=':', label='Cell A')
ax3.plot(data[1][:idx2,0], ls=':', label='Cell B')
ax3.plot(data[0][:100,0], ls='-', color=c1)
ax3.plot(data[1][:100,0], ls='-', color=c2)
ax3.set_xlim((0,1100))
ax3.set_ylim((0.88,1.06))
#ax3.axhline(0.88,color='k',ls=':')
ax3.set_xlabel('Cycle number')
ax3.set_ylabel('Discharge capacity (Ah)')
ax3.annotate("Cell A\n{} cycles".format(idx1), xy=(idx1-500, 0.885), xytext=(idx1-500, 0.885), color=c1)
ax3.annotate("Cell B\n{} cycles".format(idx2), xy=(idx2-500, 0.885), xytext=(idx2-450, 0.885), color=c2)

plt.savefig('ML_methods.png', dpi=300)
plt.savefig('ML_methods.eps', format='eps')