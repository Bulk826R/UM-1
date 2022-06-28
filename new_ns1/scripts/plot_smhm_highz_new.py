from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.color'] = 'black'
matplotlib.rcParams["grid.color"] = 'grey'
matplotlib.rcParams["grid.linestyle"] = '--'
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator, LogLocator
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
fig=plt.figure()

h = 0.7
omega_m = 0.286
wr = '/sdf/group/kipac/u/ycwang/MWmass_new/hmf/TNG'
baseDir = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

######################################

fi_lmc = np.loadtxt(wr+'/fi_lmc.txt')
fi_x = fi_lmc[:, 0]
fi_y = fi_lmc[:, 1]

fi_1s_x = np.log10(fi_x[:7])
fi_2s_x= np.log10(fi_x[14:21])
fi_1s_yhigh = np.log10(fi_y[:7])
fi_1s_ylow = np.log10(fi_y[7:14])
fi_2s_yhigh = np.log10(fi_y[14:21])
fi_2s_ylow = np.log10(fi_y[21:28])

UM_z0 = np.loadtxt(wr+'/UM_z0.txt')
UM_x0 = np.log10(UM_z0[:, 0])
UM_y0 = np.log10(UM_z0[:, 1])

xl = 7

xA = UM_x0[0]
xB = UM_x0[1]
yA = UM_y0[0]
yB = UM_y0[1]
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

UM_x0 = np.concatenate(([xl], UM_x0))
UM_y0 = np.concatenate(([yl], UM_y0))

######################################

x_0 = np.loadtxt(baseDir+'/smhm/z0/full_x.txt')
y_0 =np.loadtxt(baseDir+'/smhm/z0/full_y.txt')
y_std_0 = np.loadtxt(baseDir+'/smhm/z0/full_y_std.txt')
y_med_0 = np.loadtxt(baseDir+'/smhm/z0/full_y_med.txt')
y_low_0 = np.loadtxt(baseDir+'/smhm/z0/full_y_low.txt')
y_high_0 = np.loadtxt(baseDir+'/smhm/z0/full_y_high.txt')

######################################

x_0p2 = np.loadtxt(baseDir+'/smhm/z0p2/full_x.txt')
y_0p2 =np.loadtxt(baseDir+'/smhm/z0p2/full_y.txt')
y_std_0p2 = np.loadtxt(baseDir+'/smhm/z0p2/full_y_std.txt')
y_med_0p2 = np.loadtxt(baseDir+'/smhm/z0p2/full_y_med.txt')
y_low_0p2 = np.loadtxt(baseDir+'/smhm/z0p2/full_y_low.txt')
y_high_0p2 = np.loadtxt(baseDir+'/smhm/z0p2/full_y_high.txt')

######################################

x_0p5 = np.loadtxt(baseDir+'/smhm/z0p5/full_x.txt')
y_0p5 =np.loadtxt(baseDir+'/smhm/z0p5/full_y.txt')
y_std_0p5 = np.loadtxt(baseDir+'/smhm/z0p5/full_y_std.txt')
y_med_0p5 = np.loadtxt(baseDir+'/smhm/z0p5/full_y_med.txt')
y_low_0p5 = np.loadtxt(baseDir+'/smhm/z0p5/full_y_low.txt')
y_high_0p5 = np.loadtxt(baseDir+'/smhm/z0p5/full_y_high.txt')

######################################

x_1 = np.loadtxt(baseDir+'/smhm/z1/full_x.txt')
y_1 =np.loadtxt(baseDir+'/smhm/z1/full_y.txt')
y_std_1 = np.loadtxt(baseDir+'/smhm/z1/full_y_std.txt')
y_med_1 = np.loadtxt(baseDir+'/smhm/z1/full_y_med.txt')
y_low_1 = np.loadtxt(baseDir+'/smhm/z1/full_y_low.txt')
y_high_1 = np.loadtxt(baseDir+'/smhm/z1/full_y_high.txt')

######################################

x_2 = np.loadtxt(baseDir+'/smhm/z2/full_x.txt')
y_2 =np.loadtxt(baseDir+'/smhm/z2/full_y.txt')
y_std_2 = np.loadtxt(baseDir+'/smhm/z2/full_y_std.txt')
y_med_2 = np.loadtxt(baseDir+'/smhm/z2/full_y_med.txt')
y_low_2 = np.loadtxt(baseDir+'/smhm/z2/full_y_low.txt')
y_high_2 = np.loadtxt(baseDir+'/smhm/z2/full_y_high.txt')

######################################

x_4 = np.loadtxt(baseDir+'/smhm/z4/full_x.txt')
y_4 =np.loadtxt(baseDir+'/smhm/z4/full_y.txt')
y_std_4 = np.loadtxt(baseDir+'/smhm/z4/full_y_std.txt')
y_med_4 = np.loadtxt(baseDir+'/smhm/z4/full_y_med.txt')
y_low_4 = np.loadtxt(baseDir+'/smhm/z4/full_y_low.txt')
y_high_4 = np.loadtxt(baseDir+'/smhm/z4/full_y_high.txt')

######################################

x_6 = np.loadtxt(baseDir+'/smhm/z6/full_x.txt')
y_6 =np.loadtxt(baseDir+'/smhm/z6/full_y.txt')
y_std_6 = np.loadtxt(baseDir+'/smhm/z6/full_y_std.txt')
y_med_6 = np.loadtxt(baseDir+'/smhm/z6/full_y_med.txt')
y_low_6 = np.loadtxt(baseDir+'/smhm/z6/full_y_low.txt')
y_high_6 = np.loadtxt(baseDir+'/smhm/z6/full_y_high.txt')

######################################

x_8 = np.loadtxt(baseDir+'/smhm/z8/full_x.txt')
y_8 =np.loadtxt(baseDir+'/smhm/z8/full_y.txt')
y_std_8 = np.loadtxt(baseDir+'/smhm/z8/full_y_std.txt')
y_med_8 = np.loadtxt(baseDir+'/smhm/z8/full_y_med.txt')
y_low_8 = np.loadtxt(baseDir+'/smhm/z8/full_y_low.txt')
y_high_8 = np.loadtxt(baseDir+'/smhm/z8/full_y_high.txt')

######################################

x_10 = np.loadtxt(baseDir+'/smhm/z10/full_x.txt')
y_10 =np.loadtxt(baseDir+'/smhm/z10/full_y.txt')
y_std_10 = np.loadtxt(baseDir+'/smhm/z10/full_y_std.txt')
y_med_10 = np.loadtxt(baseDir+'/smhm/z10/full_y_med.txt')
y_low_10 = np.loadtxt(baseDir+'/smhm/z10/full_y_low.txt')
y_high_10 = np.loadtxt(baseDir+'/smhm/z10/full_y_high.txt')

######################################

fig.set_size_inches(8, 6)
ax1 = fig.add_subplot(1, 1, 1)

cmap = matplotlib.cm.get_cmap('seismic_r')
rgba0 = cmap(0.95) 
rgba1 = cmap(0.8)
rgba2 = cmap(0.68)
rgba4 = cmap(0.55)
rgba6 = cmap(0.42)
rgba8 = cmap(0.28)
rgba10 = cmap(0.12)

ax1.plot(10**x_0, 10**y_med_0, color = rgba0, linewidth = 2.4, label = r'$z=0$', zorder = 999)
ax1.plot(10**x_1, 10**y_med_1, color = rgba1, linewidth = 2.3, label = r'$z=1$')
ax1.plot(10**x_2, 10**y_med_2, color = rgba2, linewidth = 2.2, label = r'$z=2$')
ax1.plot(10**x_4, 10**y_med_4, color = rgba4, linewidth = 2.1, label = r'$z=4$')
ax1.plot(10**x_6, 10**y_med_6, color = rgba6, linewidth = 2.1, label = r'$z=6$')
ax1.plot(10**x_8, 10**y_med_8, color = rgba8, linewidth = 2., label = r'$z=8$')
ax1.plot(10**x_10, 10**y_med_10, color = rgba10, linewidth = 2., label = r'$z=10$')
ax1.plot(10**x_8, 10**y_med_8*2, color = rgba8, linewidth = 1.2, linestyle = ':')
ax1.plot(10**x_8, 10**y_med_8/2, color = rgba8, linewidth = 1.2, linestyle = ':')

ax1.fill_between(10**fi_1s_x[:-1], 10**fi_1s_ylow[:-1], 10**fi_1s_yhigh[:-1], facecolor='dimgray', alpha=0.32, label = 'Nadler+2020')    
ax1.fill_between(10**fi_2s_x[:-1], 10**fi_2s_ylow[:-1], 10**fi_2s_yhigh[:-1], facecolor='dimgray', alpha=0.16)

ax1.legend(loc = 'lower right', fontsize = 15, ncol = 2, handlelength = 2.8, frameon = False, borderpad = 1)
ax1.tick_params(labelsize = 18)
ax1.set_xlabel(r'$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$M_{\mathrm{\ast}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24, labelpad = 8)
ax1.set_xlim([10**7.5, 10**11])
ax1.set_ylim([10**-0.2, 10**8.8])

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(np.logspace(8, 11, 4))
ax1.set_yticks(np.logspace(0, 8, 9))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

fig.savefig('/sdf/home/y/ycwang/figures/smhm_highz.png', dpi = 400, bbox_inches = 'tight')
plt.show()