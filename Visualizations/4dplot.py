#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:34:46 2018

@author: peter
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# x-y limits
x = np.linspace(3.6,8,64) # C1
y = np.linspace(3.6,8,64) # C2

list_c3 = np.array([3.6, 4.0, 4.4, 4.8, 5.2, 5.6])

# convert to 2d matrices
C1, C2 = np.meshgrid(x, y)    # 50x50

# Cuts 
##list_cp = np.linspace(2,10,9)
C3 = []
for c3 in list_c3:
    C3.append(c3*np.ones((64,64)))
#C3.append(C1)

# Plot stuff
fig = plt.figure()
plt.rcParams.update({'font.size': 16})
ax = fig.gca(projection='3d')
minn, maxx = 2.5, 4.8
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
m.set_array([])

# fourth dimention - colormap
for k, c3 in enumerate(C3):
    # type this into wolfram alpha:
    # solve for x: 50/3 = x/C_1 + (10)/C_5 + (70-x)/C_2
    c4 = 0.2/(1/6 - (0.2/C1 + 0.2/C2 + 0.2/c3))
    c3[c4 < 0] = float('NaN')
    c3[c4 > 4.8] = float('NaN')
    fcolor = m.to_rgba(c4)
    ax.plot_surface(C1,C2,c3, rstride=1, cstride=1, facecolors=fcolor, vmin=minn, vmax=maxx, shade=False)

# PLOT FORMATTING
ax.scatter(4.8, 4.8, 4.8, c='k', marker='s', s=50)
ax.set_title('SOC width = 20%, C4 constrained')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
ax.set_zlim(2.5, 6.5)
cbar = plt.colorbar(m)
cbar.ax.set_title('C4')

# Make full screen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
fig.canvas.show()

el = 45
ax.view_init(elev=el, azim=0)

# animation function. This is called sequentially
def animate(i):
    azimuth = i*10
    ax.view_init(elev=el, azim=azimuth)
    return ax

anim = animation.FuncAnimation(fig, animate, frames=36,
                               interval=1000, blit=False)

anim.save('4danimation_layers_' + str(el) + '.gif', writer='imagemagick', fps=1)
