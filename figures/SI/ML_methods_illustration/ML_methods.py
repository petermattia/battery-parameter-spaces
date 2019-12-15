#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor

plt.close('all')

MAX_WIDTH = 183 / 25.4 # mm -> inches
figsize=(MAX_WIDTH, 5/6 * MAX_WIDTH)

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
    

# Maximum value for plot
max_x1 = np.argmax(data[0][:100,0])
max_y1 = np.max(data[0][:100,0])
max_x2 = np.argmax(data[1][:100,0])
max_y2 = np.max(data[1][:100,0])

# Discharge capacity
ax0.set_prop_cycle(plt.style.library['bmh']['axes.prop_cycle'])
ax0.plot(np.arange(100)+1,data[0][:100,0])
ax0.plot(np.arange(100)+1,data[1][:100,0])
ax0.scatter(2, data[0][2,0], c=c1)
ax0.scatter(2, data[1][2,0], c=c2)
ax0.scatter(max_x1, max_y1, marker='s', c=c1)
ax0.scatter(max_x2, max_y2, marker='s', c=c2)
ax0.set_xlabel('Cycle number')
ax0.set_ylabel('Discharge capacity, $Q_d$ (Ah)')
ax0.set_xlim((0,100))
ax0.annotate("Cell A", xy=(70, 1.048), xytext=(70, 1.0494), color=c1)
ax0.annotate("Cell B", xy=(70, 1.054), xytext=(70, 1.0542),  color=c2)
# Custom legend
legend_elements = [Line2D([0], [0], marker='o', color='w', label=r'$Q_{d,2}$',
                          markerfacecolor='k', markersize=6),
                   Line2D([0], [0], marker='s', color='w', label=r'max($Q_d$)',
                          markerfacecolor='k', markersize=6)]
ax0.legend(handles=legend_elements, loc='lower center')

ax0b.text(-0.15, 0.70, 'Capacity features\n(from first 100 cycles)')
ax0b.text(-0.15, 0.45, '• 2$\mathdefault{ ^{nd}}$-cycle discharge capacity\n  (' +
          r'$Q_{d,2}$)')
ax0b.text(-0.15, 0.15, '• Max $-$ 2$\mathdefault{ ^{nd}}$-cycle discharge capacity\n  (' +
          'max(' + r'$Q_d$'+ ')' + r'$ - Q_{d,2}$)')
ax0b.axis('off')

ax0c.annotate('', xy=(0.6, 0.25), xytext=(-0.1, 0.4),
              arrowprops=dict(arrowstyle="->"))
ax0c.annotate('To early prediction', xy=(0.58, 0.35), xytext=(-0.15, 0.42))
ax0c.axis('off')

# Voltage
QV100a = data[0][:,1][~np.isnan(data[0][:,1])]
QV10a  = data[0][:,2][~np.isnan(data[0][:,2])]
QV100b = data[1][:,1][~np.isnan(data[1][:,1])]
QV10b  = data[1][:,2][~np.isnan(data[1][:,2])]
ax1.plot(QV10b , np.linspace(3.5,2.0,1000), 'k', label='Cycle 10')
ax1.plot(QV100b, np.linspace(3.5,2.0,1000), '--k', label='Cycle 100')
ax1.set_xlabel('Discharge capacity (Ah)')
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim((2.0, 3.5))
ax1.legend()
ax1.set_xlim((0, 1.1))

# Voltage difference

# Minimum value for plot
y = np.linspace(3.5,2.0,1000)
DeltaQVa = QV100a - QV10a
DeltaQVb = QV100b - QV10b
min_x1 = np.min(DeltaQVa)
min_y1 = y[np.argmin(DeltaQVa)]
min_x2 = np.min(DeltaQVb)
min_y2 = y[np.argmin(DeltaQVb)]

ax2.set_prop_cycle(plt.style.library['bmh']['axes.prop_cycle'])
ax2.plot(DeltaQVa, y, label='Cell A')
ax2.plot(DeltaQVb, y, label='Cell B')
ax2.scatter(min_x1, min_y1, marker='^', c=c1)
ax2.scatter(min_x2, min_y2, marker='^', c=c2)
ax2.axvline(0,color='k', ls=':')
ax2.set_xlabel(r'$Q_{100}-Q_{10}$' + ' (Ah)')
ax2.set_ylabel('Voltage (V)')
ax2.set_ylim((2.0, 3.5))
ax2.annotate("Cell A", xy=(0.002, 3.22), xytext=(0.002, 3.22), color=c1)
ax2.annotate("Cell B", xy=(-0.02, 3.22), xytext=(-0.02, 3.22), color=c2)
ax2.set_xlim((-0.045, 0.025))

# Custom legend
legend_elements = [Line2D([0], [0], marker='^', color='w', label='min',
                          markerfacecolor='k', markersize=6)]
ax2.legend(handles=legend_elements, loc='lower left')

ax2b.text(-0.15, 0.15, '• Skewness (of entire curve)')
ax2b.text(-0.15, 0.35, '• Variance (of entire curve)')
ax2b.text(-0.15, 0.55, '• Minimum')
ax2b.text(-0.15, 0.70, 'Voltage features\n(from cycle 100 $-$ cycle 10)')
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
ax3.plot(data[0][:idx1,0], ls=':', lw=1)
ax3.plot(data[1][:idx2,0], ls=':', lw=1)
ax3.plot(data[0][:100,0], ls='-', color=c1, lw=1)
ax3.plot(data[1][:100,0], ls='-', color=c2, lw=1)
ax3.scatter(idx1, 0.88, marker='*')
ax3.scatter(idx2, 0.88, marker='*')
ax3.set_xlim((0,1100))
ax3.set_ylim((0.875,1.08))
ax3.set_xlabel('Cycle number')
ax3.set_ylabel('Discharge capacity (Ah)')
ax3.annotate("Cell A\n{} cycles".format(idx1), xy=(idx1-425, 0.88), xytext=(idx1-425, 0.88), color=c1)
ax3.annotate("Cell B\n{} cycles".format(idx2), xy=(idx2-400, 0.88), xytext=(idx2-400, 0.88), color=c2)

# Custom legend
legend_elements = [Line2D([0], [0], color='k', ls='-', label='Measured'),
                   Line2D([0], [0], color='k', ls=':', label='Predicted')]
ax3.legend(handles=legend_elements, loc='upper right')

plt.savefig('ML_methods.png', dpi=300)
plt.savefig('ML_methods.eps', format='eps')