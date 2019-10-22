#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:40:41 2019

@author: peter
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

comparison_plots = False

train_file = glob.glob('Qn_train.csv')[0]
train = np.genfromtxt(train_file,delimiter=',')

valid_file = glob.glob('Qn_valid.csv')[0]
valid = np.genfromtxt(valid_file,delimiter=',')

if comparison_plots:
    # Compare datasets
    plt.plot(train.T,color='tab:blue')
    plt.plot(valid.T,color='tab:red')
    plt.show()

# Filter data <0.88
train[np.where(train<0.88)] = np.NaN
valid[np.where(valid<0.88)] = np.NaN

if comparison_plots:
    # Compare datasets again
    plt.figure()
    plt.plot(train.T,color='tab:blue')
    plt.plot(valid.T,color='tab:red')
    plt.show()

# Test functional forms
# From Fig 1 of Domhan et al: 
# https://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf
# Equations inversed to match capacity fade curves
def pow3(x, a, b, c):
    return a/(b-x)**c

def logpow(x, a, b, c):
    return a*(b/x-1)**c

def pow4(x, a, b, c, d):
    return a/(b-x)**c + d

def MMF(x, a, b, c, d):
    return (((b-a)/(b-x)-1)/c) ** d

def exp3(x, a, b, c): # exp4 didn't converge
    return a*np.log(b-x) + c

def Janoschek(x, a, b, c, d): # same as Weibull
    return a*(np.log((b-c)/(b-x)))**d

def ilog2(x, a, b, c):
    return a*np.log(b-x)+c

def rat(x, a, b): ## not in Domhan
    return (b-a)/(b-x)

def exp2(x, a, b, c, d): ## not in Domhan
    return a*np.exp(b*x) + c*np.exp(d*x)

def pow2(x, a, b, c): ## not in Domhan
    return a * x**b + c

# Sample dataset
y_v = 1.1 - valid[2,:][~np.isnan(valid[2,:])]
x_v = np.arange(len(y_v))+1

y = 1.1 - train[2,:][~np.isnan(train[2,:])]
x = np.arange(len(y))+1
# import pdb
# pdb.set_trace()

plt.figure()
plt.plot(x, y, 'ok', label='data')
plt.ylim((0, 0.25))
plt.xlabel('Cycle number')
plt.ylabel('1.1 Ah - capacity (Ah)')

# Pre-set bounds for functions to help convergence. Primarily set lower bounds on 'b'
bounds2=([-np.inf, 1075], [np.inf, np.inf])
bounds3=([-np.inf, 1075, -np.inf], [np.inf, np.inf, np.inf])
bounds4=([-np.inf, 1075, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

# pow3
popt,pcov = curve_fit(pow3, x, y, bounds=bounds3)
plt.plot(x,pow3(x,popt[0],popt[1],popt[2]), label='pow3')

# logpow
popt,pcov = curve_fit(logpow, x, y, bounds=bounds3)
print(popt[0],popt[1],popt[2])
plt.plot(x,logpow(x,popt[0],popt[1],popt[2]), label='logpow')

# pow4
popt,pcov = curve_fit(pow4, x, y, bounds=bounds4)
plt.plot(x,pow4(x,popt[0],popt[1],popt[2],popt[3]), label='pow4')

# MMF
popt,pcov = curve_fit(MMF, x, y, bounds=bounds4)
plt.plot(x,MMF(x,popt[0],popt[1],popt[2],popt[3]), label='MMF')

# exp3
popt,pcov = curve_fit(exp3, x, y, bounds=bounds3)
plt.plot(x,exp3(x,popt[0],popt[1],popt[2]), label='exp3')

# Janoschek
popt,pcov = curve_fit(Janoschek, x, y, bounds=bounds4, maxfev=5000) # had trouble converging
plt.plot(x,Janoschek(x,popt[0],popt[1],popt[2],popt[3]), label='Janoschek')

# ilog2
popt,pcov = curve_fit(ilog2, x, y, bounds=bounds3)
plt.plot(x,ilog2(x,popt[0],popt[1],popt[2]), label='ilog2')

# rat
popt,pcov = curve_fit(rat, x, y, bounds=bounds2)
plt.plot(x,rat(x,popt[0],popt[1]), label='rat')

# exp2
popt,pcov = curve_fit(exp2, x, y)
plt.plot(x,exp2(x,popt[0],popt[1],popt[2],popt[3]), label='exp2')

# pow2
bounds_pow=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
popt,pcov = curve_fit(pow2, x, y, bounds=bounds_pow, maxfev=5000) # had trouble converging
plt.plot(x,pow2(x,popt[0],popt[1],popt[2]), label='pow2')

plt.legend(frameon=False)
plt.savefig('fits.png', bbox_inches='tight')