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
x = np.linspace(3,6,50) # [1,8,50]
y = np.linspace(3,6,50) # [6, 9]

# convert to 2d matrices
C1, C2 = np.meshgrid(x, y)    # 50x50

# Cuts (Cp=pulse)
list_cp = np.linspace(8,10,1)
Cp = []
for cp in list_cp:
    Cp.append(cp*np.ones((50,50)))
Cp.append(C1)

# Plot stuff
fig = plt.figure()
plt.rcParams.update({'font.size': 16})
ax = fig.gca(projection='3d')
minn, maxx = 0, 70
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='winter')
m.set_array([])

# fourth dimention - colormap
for k, cp in enumerate(Cp):
    # type this into wolfram alpha:
    # solve for x: 50/3 = x/C_1 + (10)/C_5 + (70-x)/C_2
    color = (10*C1*(C2*(3-5*cp) + 21*cp))/(3*(C1-C2)*cp)
    cp[color < 0] = float('NaN')
    cp[color > 70] = float('NaN')
    fcolor = m.to_rgba(color)
    ax.plot_surface(C1,C2,cp, rstride=1, cstride=1, facecolors=fcolor, vmin=minn, vmax=maxx, shade=False)

# PLOT FORMATTING
ax.scatter(4.8, 4.8, 4.8, c='k', s=50)
ax.set_title('Q2 = Q1 + 10%')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C_pulse')
ax.set_zlim(0, 10)
cbar = plt.colorbar(m)
cbar.ax.set_ylabel('Q1')

# Make full screen
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
fig.canvas.show()

ax.view_init(elev=0, azim=0)


# animation function. This is called sequentially
def animate(i):
    azimuth = i*10
    ax.view_init(elev=0, azim=azimuth)
    return ax

anim = animation.FuncAnimation(fig, animate, frames=36, 
                               interval=1000, blit=False)

anim.save('4danimation2.gif', writer='imagemagick', fps=1)