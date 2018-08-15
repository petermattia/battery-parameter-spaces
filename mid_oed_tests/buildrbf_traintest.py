#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 10:22:45 2018

@author: peter
"""

import numpy as np
from scipy.interpolate import Rbf
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

##############################################################################
hp_list = np.linspace(0.1,20.0,20) # Smoothing parameter

k_folds = 10 # folds for cross validation
##############################################################################

# IMPORT RESULTS
# Get folder path containing text files
cwd = os.getcwd()
data = pickle.load(open(cwd + '/data so far/good.pkl', 'rb'))

# PREINITIALIZE
RMSE_train_all = np.zeros((len(hp_list),k_folds))
RMSE_test_all  = np.zeros((len(hp_list),k_folds))

for k, hp in enumerate(hp_list):
    kf = KFold(n_splits=k_folds)
    
    for k2, (train_index, test_index) in enumerate(kf.split(data[:,0:3])):
        # SPLIT INTO TRAIN/TEST
        X_train, X_test = data[train_index,0:3], data[test_index,0:3]
        y_train, y_test = data[train_index,4],   data[test_index,4]

        # CREATE RBF
        rbf = Rbf(X_train[:,0], X_train[:,1], X_train[:,2],y_train,epsilon=hp)

        # EVALUATE RBF ON TRAIN DATA
        y_pred_train = rbf(X_train[:,0], X_train[:,1], X_train[:,2])
        RMSE_train_all[k,k2] = np.sqrt(mean_squared_error(y_train,y_pred_train))
        
        # EVALUATE RBF ON TEST DATA
        y_pred_test = rbf(X_test[:,0], X_test[:,1], X_test[:,2])
        RMSE_test_all[k,k2] = np.sqrt(mean_squared_error(y_test,y_pred_test))

# EVALUATE RMSE FOR A GIVEN HYPERPARAMETER FOR ALL FOLDS
RMSE_train = np.mean(RMSE_train_all,axis=1)
RMSE_test  = np.mean(RMSE_test_all,axis=1)


## PLOT
plt.subplot(2,1,1)
plt.plot(hp_list, RMSE_train)
plt.plot(hp_list, RMSE_test)
plt.xlabel('smoothing param'), plt.ylabel('RMSE')
plt.gca().legend(('train','test'))

plt.subplot(2,1,2)
plt.plot(hp_list, RMSE_train)
plt.plot(hp_list, RMSE_test)
plt.ylim((0,1000))
plt.xlabel('smoothing param'), plt.ylabel('RMSE')
plt.gca().legend(('train','test'))

# save fullscreen
manager = plt.get_current_fig_manager() # Make full screen
manager.window.showMaximized()
plt.savefig('rmse.png')