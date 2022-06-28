from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
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

########################################################

h = 0.7
omega_m = 0.286
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

data_2048 = np.loadtxt(wr+'/Data_join_MWmass.txt')
lg_Mvir = np.log10(data_2048[:, 11])
lg_Mpeak = np.log10(data_2048[:, 13])
Vmpeak = data_2048[:, 14]

########################################################

bins_x = np.linspace(np.min(lg_Mvir), np.max(lg_Mvir), 20)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_Mvir) - bins_h, np.max(lg_Mvir) + bins_h, 21)

bin_tot = np.digitize(lg_Mvir, bins)
bin_means_tot = np.asarray([lg_Mvir[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 50) for i in range(1, len(bins))])
bin_std = np.asarray([lg_Mpeak[bin_tot == i].std() for i in range(1, len(bins))])

bin_Mvir = bins_x
bin_Mpeak_mean = bin_means
bin_Mpeak_std = bin_std

bin_Mpeak_low_1s =  np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 15.85) for i in range(1, len(bins))])
bin_Mpeak_low_2s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 2.3) for i in range(1, len(bins))])
bin_Mpeak_high_1s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 84.15) for i in range(1, len(bins))])
bin_Mpeak_high_2s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 97.7) for i in range(1, len(bins))])

######

bins_x = np.linspace(np.min(lg_Mpeak), np.max(lg_Mpeak), 20)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_Mpeak) - bins_h, np.max(lg_Mpeak) + bins_h, 21)

bin_tot = np.digitize(lg_Mpeak, bins)
bin_means_tot = np.asarray([lg_Mpeak[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([np.percentile(Vmpeak[bin_tot == i], 50) for i in range(1, len(bins))])
bin_std = np.asarray([Vmpeak[bin_tot == i].std() for i in range(1, len(bins))])

bin_Mpeak = bins_x
bin_Vpeak_mean = bin_means
bin_Vpeak_std = bin_std

bin_Vpeak_low_1s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 15.85) for i in range(1, len(bins))])
bin_Vpeak_low_2s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 2.3) for i in range(1, len(bins))])
bin_Vpeak_high_1s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 84.15) for i in range(1, len(bins))])
bin_Vpeak_high_2s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 97.7) for i in range(1, len(bins))])

########################################################

fig.set_size_inches(24, 12)

ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(10**lg_Mvir, 10**lg_Mpeak, s = 0.5, alpha = 0.2, color = 'deepskyblue', marker = '.', rasterized = True)
ax1.axvline(x = 2.82e7, color = 'darkturquoise', dashes = (5, 2.4), linewidth = 2., alpha = 0.72)
ax1.fill_between(10**bin_Mvir, 10**bin_Mpeak_low_1s, 10**bin_Mpeak_high_1s, facecolor='navy', alpha=0.42)
ax1.fill_between(10**bin_Mvir, 10**bin_Mpeak_low_2s, 10**bin_Mpeak_high_2s, facecolor='navy', alpha=0.24)
#ax1.scatter([11.2], [12.2], s = 42, alpha = 0.8, color = 'deepskyblue', marker = '.', label = r'Zoom-in')

ax1.set_xlabel('$M_{\mathrm{vir}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax1.set_ylabel('$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax1.legend(loc = 'upper left', fontsize = 18, frameon = False, borderpad = 0.02)
ax1.set_xlim([10**6.2, 10**8.8])
ax1.set_ylim([10**6.2, 10**8.8])

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin * 1.2, ymax / 1.2, r'Zoom-in', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax1.text(2.82e7 * 1.2, ymin * 1.2, 
         r'$100\times m_{\mathrm{DM}}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'black', alpha = 0.8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(np.logspace(7, 8, 2))
ax1.set_yticks(np.logspace(7, 8, 2))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


ax2 = fig.add_subplot(2, 3, 4)
ax2.scatter(10**lg_Mpeak, Vmpeak, s = 0.5, alpha = 0.2, color = 'deepskyblue', marker = '.', rasterized = True)
ax2.axvline(x = 2.82e7, color = 'darkturquoise', dashes = (5, 2.4), linewidth = 2., alpha = 0.72)
ax2.axhline(y = 10, color = 'crimson', linestyle = '-.', linewidth = 2., alpha = 0.82)
ax2.fill_between(10**bin_Mpeak, bin_Vpeak_low_1s, bin_Vpeak_high_1s, facecolor='navy', alpha=0.42)
ax2.fill_between(10**bin_Mpeak, bin_Vpeak_low_2s, bin_Vpeak_high_2s, facecolor='navy', alpha=0.24)
#ax2.scatter([11.2], [100.2], s = 42, alpha = 0.8, color = 'deepskyblue', marker = '.', label = r'Zoom-in')

ax2.set_xlabel('$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax2.set_ylabel('$V_{\mathrm{Mpeak}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 24)
ax2.legend(loc = 'upper left', fontsize = 18, frameon = False, borderpad = 0.02)
ax2.set_xlim([10**6.2, 10**8.8])
ax2.set_ylim([0, 18])

xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax2.text(xmin * 1.2, ymax - height * 0.03, r'Zoom-in', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax2.text(2.82e7 * 1.2, ymin + height * 0.03, 
         r'$100\times m_{\mathrm{DM}}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'black', alpha = 0.8)

ax2.text(xmin * 1.2, 10 + height * 0.03, 
         r'$V_{\mathrm{Mpeak}}\ \mathrm{cut}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'crimson', alpha = 0.8)

ax2.set_xscale('log')
ax2.set_xticks(np.logspace(7, 8, 2))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.xaxis.set_minor_locator(loc1)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

########################################################

h = 0.7
omega_m = 0.286

data_2048 = np.loadtxt(src+'/hmf/Data_2048_z0.txt')
lg_Mvir = np.log10(data_2048[:, 5])
lg_Mpeak = np.log10(data_2048[:, 4])
Vmpeak = data_2048[:, 7]

########################################################

bins_x = np.linspace(np.min(lg_Mvir), np.max(lg_Mvir), 20)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_Mvir) - bins_h, np.max(lg_Mvir) + bins_h, 21)

bin_tot = np.digitize(lg_Mvir, bins)
bin_means_tot = np.asarray([lg_Mvir[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 50) for i in range(1, len(bins))])
bin_std = np.asarray([lg_Mpeak[bin_tot == i].std() for i in range(1, len(bins))])

bin_Mvir = bins_x
bin_Mpeak_mean = bin_means
bin_Mpeak_std = bin_std

bin_Mpeak_low_1s =  np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 15.85) for i in range(1, len(bins))])
bin_Mpeak_low_2s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 2.3) for i in range(1, len(bins))])
bin_Mpeak_high_1s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 84.15) for i in range(1, len(bins))])
bin_Mpeak_high_2s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 97.7) for i in range(1, len(bins))])

######

bins_x = np.linspace(np.min(lg_Mpeak), np.max(lg_Mpeak), 20)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_Mpeak) - bins_h, np.max(lg_Mpeak) + bins_h, 21)

bin_tot = np.digitize(lg_Mpeak, bins)
bin_means_tot = np.asarray([lg_Mpeak[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([np.percentile(Vmpeak[bin_tot == i], 50) for i in range(1, len(bins))])
bin_std = np.asarray([Vmpeak[bin_tot == i].std() for i in range(1, len(bins))])

bin_Mpeak = bins_x
bin_Vpeak_mean = bin_means
bin_Vpeak_std = bin_std

bin_Vpeak_low_1s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 15.85) for i in range(1, len(bins))])
bin_Vpeak_low_2s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 2.3) for i in range(1, len(bins))])
bin_Vpeak_high_1s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 84.15) for i in range(1, len(bins))])
bin_Vpeak_high_2s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 97.7) for i in range(1, len(bins))])

########################################################

ax1 = fig.add_subplot(2, 3, 2)
ax1.scatter(10**lg_Mvir, 10**lg_Mpeak, s = 0.5, alpha = 0.2, color = 'deepskyblue', marker = '.', rasterized = True)
ax1.axvline(x = 1.8e9, color = 'darkturquoise', dashes = (5, 2.4), linewidth = 2., alpha = 0.72)
ax1.fill_between(10**bin_Mvir, 10**bin_Mpeak_low_1s, 10**bin_Mpeak_high_1s, facecolor='navy', alpha=0.42)
ax1.fill_between(10**bin_Mvir, 10**bin_Mpeak_low_2s, 10**bin_Mpeak_high_2s, facecolor='navy', alpha=0.24)
#ax1.scatter([11.2], [12.2], s = 42, alpha = 0.8, color = 'deepskyblue', marker = '.', label = r'c125-2048')

ax1.set_xlabel('$M_{\mathrm{vir}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax1.set_ylabel('$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax1.legend(loc = 'upper left', fontsize = 18, frameon = False, borderpad = 0.02)
ax1.set_xlim([10**8, 10**10.5])
ax1.set_ylim([10**8, 10**10.5])

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin * 1.2, ymax / 1.2, r'c125-2048', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax1.text(1.8e9 * 1.2, ymin * 1.2, 
         r'$100\times m_{\mathrm{DM}}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'black', alpha = 0.8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(np.logspace(8, 10, 3))
ax1.set_yticks(np.logspace(8, 10, 3))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


ax2 = fig.add_subplot(2, 3, 5)
ax2.scatter(10**lg_Mpeak, Vmpeak, s = 0.5, alpha = 0.2, color = 'deepskyblue', marker = '.', rasterized = True)
ax2.axvline(x = 1.8e9, color = 'darkturquoise', dashes = (5, 2.4), linewidth = 2., alpha = 0.72)
ax2.axhline(y = 35, color = 'crimson', linestyle = '-.', linewidth = 2., alpha = 0.82)
ax2.fill_between(10**bin_Mpeak, bin_Vpeak_low_1s, bin_Vpeak_high_1s, facecolor='navy', alpha=0.42)
ax2.fill_between(10**bin_Mpeak, bin_Vpeak_low_2s, bin_Vpeak_high_2s, facecolor='navy', alpha=0.24)
#ax2.scatter([11.2], [100.2], s = 42, alpha = 0.8, color = 'deepskyblue', marker = '.', label = r'c125-2048')

ax2.set_xlabel('$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax2.set_ylabel('$V_{\mathrm{Mpeak}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 24)
ax2.legend(loc = 'upper left', fontsize = 18, frameon = False, borderpad = 0.02)
ax2.set_xlim([10**8, 10**10.5])
ax2.set_ylim([1, 60])

xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax2.text(xmin * 1.2, ymax - height * 0.03, r'c125-2048', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax2.text(1.8e9 * 1.2, ymin + height * 0.03, 
         r'$100\times m_{\mathrm{DM}}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax2.text(xmin * 1.2, 35 + height * 0.03, 
         r'$V_{\mathrm{Mpeak}}\ \mathrm{cut}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'crimson', alpha = 0.8)

ax2.set_xscale('log')
ax2.set_xticks(np.logspace(8, 10, 3))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.xaxis.set_minor_locator(loc1)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

########################################################

h = 0.7
omega_m = 0.286

data_2048 = np.loadtxt(src+'/hmf/Data_1024_z0.txt')
lg_Mvir = np.log10(data_2048[:, 5])
lg_Mpeak = np.log10(data_2048[:, 4])
Vmpeak = data_2048[:, 7]

########################################################

bins_x = np.linspace(np.min(lg_Mvir), np.max(lg_Mvir), 20)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_Mvir) - bins_h, np.max(lg_Mvir) + bins_h, 21)

bin_tot = np.digitize(lg_Mvir, bins)
bin_means_tot = np.asarray([lg_Mvir[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 50) for i in range(1, len(bins))])
bin_std = np.asarray([lg_Mpeak[bin_tot == i].std() for i in range(1, len(bins))])

bin_Mvir = bins_x
bin_Mpeak_mean = bin_means
bin_Mpeak_std = bin_std

bin_Mpeak_low_1s =  np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 15.85) for i in range(1, len(bins))])
bin_Mpeak_low_2s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 2.3) for i in range(1, len(bins))])
bin_Mpeak_high_1s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 84.15) for i in range(1, len(bins))])
bin_Mpeak_high_2s = np.asarray([np.percentile(lg_Mpeak[bin_tot == i], 97.7) for i in range(1, len(bins))])

######

bins_x = np.linspace(np.min(lg_Mpeak), np.max(lg_Mpeak), 20)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_Mpeak) - bins_h, np.max(lg_Mpeak) + bins_h, 21)

bin_tot = np.digitize(lg_Mpeak, bins)
bin_means_tot = np.asarray([lg_Mpeak[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([np.percentile(Vmpeak[bin_tot == i], 50) for i in range(1, len(bins))])
bin_std = np.asarray([Vmpeak[bin_tot == i].std() for i in range(1, len(bins))])

bin_Mpeak = bins_x
bin_Vpeak_mean = bin_means
bin_Vpeak_std = bin_std

bin_Vpeak_low_1s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 15.85) for i in range(1, len(bins))])
bin_Vpeak_low_2s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 2.3) for i in range(1, len(bins))])
bin_Vpeak_high_1s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 84.15) for i in range(1, len(bins))])
bin_Vpeak_high_2s = np.asarray([np.percentile(Vmpeak[bin_tot == i], 97.7) for i in range(1, len(bins))])

########################################################

ax1 = fig.add_subplot(2, 3, 3)
ax1.scatter(10**lg_Mvir, 10**lg_Mpeak, s = 0.5, alpha = 0.2, color = 'deepskyblue', marker = '.', rasterized = True)
ax1.axvline(x = 1.44e10, color = 'darkturquoise', dashes = (5, 2.4), linewidth = 2., alpha = 0.72)
ax1.fill_between(10**bin_Mvir, 10**bin_Mpeak_low_1s, 10**bin_Mpeak_high_1s, facecolor='navy', alpha=0.42)
ax1.fill_between(10**bin_Mvir, 10**bin_Mpeak_low_2s, 10**bin_Mpeak_high_2s, facecolor='navy', alpha=0.24)
#ax1.scatter([11.2], [12.2], s = 42, alpha = 0.8, color = 'deepskyblue', marker = '.', label = r'c125-1024')

ax1.set_xlabel('$M_{\mathrm{vir}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax1.set_ylabel('$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax1.legend(loc = 'upper left', fontsize = 18, frameon = False, borderpad = 0.02)
ax1.set_xlim([10**9, 10**11])
ax1.set_ylim([10**9, 10**12])

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin * 1.2, ymax / 1.2, r'c125-1024', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax1.text(1.44e10 * 1.2, ymin * 1.2, 
         r'$100\times m_{\mathrm{DM}}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'black', alpha = 0.8)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks(np.logspace(9, 11, 3))
ax1.set_yticks(np.logspace(9, 12, 4))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


ax2 = fig.add_subplot(2, 3, 6)
ax2.scatter(10**lg_Mpeak, Vmpeak, s = 0.5, alpha = 0.2, color = 'deepskyblue', marker = '.', rasterized = True)
ax2.axvline(x = 1.44e10, color = 'darkturquoise', dashes = (5, 2.4), linewidth = 2., alpha = 0.72)
ax2.axhline(y = 60, color = 'crimson', linestyle = '-.', linewidth = 2., alpha = 0.82)
ax2.fill_between(10**bin_Mpeak, bin_Vpeak_low_1s, bin_Vpeak_high_1s, facecolor='navy', alpha=0.42)
ax2.fill_between(10**bin_Mpeak, bin_Vpeak_low_2s, bin_Vpeak_high_2s, facecolor='navy', alpha=0.24)
#ax2.scatter([11.2], [100.2], s = 42, alpha = 0.8, color = 'deepskyblue', marker = '.', label = r'c125-1024')

ax2.set_xlabel('$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24)
ax2.set_ylabel('$V_{\mathrm{Mpeak}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 24)
ax2.legend(loc = 'upper left', fontsize = 18, frameon = False, borderpad = 0.02)
ax2.set_xlim([10**9, 10**11])
ax2.set_ylim([2, 100])

xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax2.text(xmin * 1.2, ymax - height * 0.03, r'c125-1024', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax2.text(1.44e10 * 1.2, ymin + height * 0.03,
         r'$100\times m_{\mathrm{DM}}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'black', alpha = 0.8)
ax2.text(xmin * 1.2, 60 + height * 0.03, 
         r'$V_{\mathrm{Mpeak}}\ \mathrm{cut}$', 
         horizontalalignment = 'left', verticalalignment = 'bottom', 
         fontsize = 20, color = 'crimson', alpha = 0.8)

ax2.set_xscale('log')
ax2.set_xticks(np.logspace(9, 11, 3))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.xaxis.set_minor_locator(loc1)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


########################################################

fig.savefig('/sdf/home/y/ycwang/figures/res_cut.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
