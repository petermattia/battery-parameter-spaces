#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:27:28 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FINAL_CUTOFF_10    = 70
FINAL_CUTOFF_20    = 60
FINAL_CUTOFF_30    = 50
FINAL_CUTOFF_40    = 40
PULSE_WIDTH_10     = 10   # Pulse width, in SOC
PULSE_WIDTH_20     = 20   # Pulse width, in SOC
PULSE_WIDTH_30     = 30   # Pulse width, in SOC
PULSE_WIDTH_40     = 40   # Pulse width, in SOC


PULSE           = np.linspace(0.5,20,100)    # Pulse current
chargetime_10      = 10-60/PULSE*(PULSE_WIDTH_10/100)  # [=] minutes
one_step_10 = 60*(FINAL_CUTOFF_10/100)/chargetime_10
one_step_10[one_step_10 < 0] = 100

chargetime_20      = 10-60/PULSE*(PULSE_WIDTH_20/100)  # [=] minutes
one_step_20 = 60*(FINAL_CUTOFF_20/100)/chargetime_20
one_step_20[one_step_20 < 0] = 100

chargetime_30      = 10-60/PULSE*(PULSE_WIDTH_30/100)  # [=] minutes
one_step_30 = 60*(FINAL_CUTOFF_30/100)/chargetime_30
one_step_30[one_step_30 < 0] = 100

chargetime_40      = 10-60/PULSE*(PULSE_WIDTH_40/100)  # [=] minutes
one_step_40 = 60*(FINAL_CUTOFF_40/100)/chargetime_40
one_step_40[one_step_40 < 0] = 100

sns.set()
plt.plot(PULSE, one_step_10)
plt.plot(PULSE, one_step_20)
plt.plot(PULSE, one_step_30)
plt.plot(PULSE, one_step_40)

plt.xlabel('Pulse C rate')
plt.ylabel('Average C rate')
plt.xlim((0,12))
plt.ylim((3,8))
plt.legend(('Pulse width = 10% SOC','Pulse width = 20% SOC','Pulse width = 30% SOC','Pulse width = 40% SOC'))