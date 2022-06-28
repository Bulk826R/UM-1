from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
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

src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'
baseDir = src+'/hmf'
HMF_2048 = np.load(baseDir+'/HMF_2048.npy')
HMF_1024 = np.load(baseDir+'/HMF_1024.npy')
mbins = np.load(baseDir+'/HMF_mbins.npy')
Mbins = 10**mbins

Halo_no = np.loadtxt(wr+'/halo2_list.txt')
Halo_no = Halo_no.astype(np.int64)
HMF_zoom = [np.load(baseDir+'/HMF_zoom_{0:03d}.npy'.format(Halo_no[0]))]
for i in range(1, len(Halo_no)):
    hmf_zoom = np.load(baseDir+'/HMF_zoom_{0:03d}.npy'.format(Halo_no[i]))
    HMF_zoom = np.concatenate((HMF_zoom, [hmf_zoom]), axis = 0)

#print(HMF_zoom)

zi_ext = np.loadtxt(baseDir+'/extents2.txt')
x_ext = zi_ext[:, 0]
y_ext = zi_ext[:, 1]
z_ext = zi_ext[:, 2]
Ext = x_ext * y_ext * z_ext
fV = np.sum(Ext)
print('Volume factor for zoom-in = ', fV)

def STD(A, W):
    me = np.average(A, axis = 0, weights = W)
    b = A - me * np.ones_like(A)
    #print(me)
    s2 = np.average(b**2, axis = 0, weights = W**2)
    #print(s2)
    return me, np.sqrt(s2)

a = STD(HMF_zoom, Ext/fV)
#HMF_mean = a[0]
HMF_mean =  np.average(HMF_zoom * np.outer(45*Ext/fV, np.ones_like(HMF_zoom[0])), axis = 0)
#HMF_std = a[1]
HMF_std = np.std(HMF_zoom * np.outer(45*Ext/fV, np.ones_like(HMF_zoom[0])), axis = 0)
#print(HMF_std)

HMF_med = HMF_mean
HMF_low = HMF_mean - HMF_std
HMF_high = HMF_mean + HMF_std


#Watkins 2019
MW_med = 1.54 * 1e12 
MW_high = (1.54+0.75) * 1e12
MW_low = (1.54-0.44) * 1e12

########################

fig.set_size_inches(8, 6)

ax1 = fig.add_subplot(1, 1, 1)

ax1.plot(Mbins, HMF_1024, color = 'deepskyblue', dashes = (5, 2.4), alpha = 0.8, lw = 1.6, label=r'c125-1024')
ax1.plot(Mbins, HMF_2048, color = 'royalblue', linestyle = '-.', alpha = 0.8, lw = 1.6, label=r'c125-2048')

mbins = np.load(baseDir+'/zoom_hmf_mbins.npy')
Mbins = 10**mbins

ax1.plot(Mbins, HMF_med, color = 'navy', alpha = 0.8, lw = 2, label=r'Zoom-in')
ax1.fill_between(Mbins, HMF_low, HMF_high, facecolor='deepskyblue', alpha=0.3)

ax1.axvline(x = MW_med, alpha = 0.8, color = 'crimson', lw = 1.6, linestyle = ':', label = r'Milky Way')
ax1.fill_between([MW_low, MW_high], [6e-7, 6e-7],
                 [2e3, 2e3], color = 'tomato', lw = 0., alpha = 0.32)

ax1.fill_between([10**10.30847029077118, 10**11.289506953223768], [6e-7, 6e-7],
                 [2e3, 2e3], color = 'black', lw = 0., alpha = 0.32)
ax1.fill_between([10**9.281921618204807, 10**10.30847029077118], [6e-7, 6e-7],
                 [2e3, 2e3], color = 'dimgray', lw = 0., alpha = 0.24)
ax1.fill_between([10**7.6991150442477885, 10**9.281921618204807], [6e-7, 6e-7],
                 [2e3, 2e3], color = 'dimgray', lw = 0., alpha = 0.1)
ax1.set_xlim([10**7, 10**15.2])


ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.set_xlim([5e5, 2e15])
ax1.set_ylim([7e-7, 2e2])
ax1.set_xlabel(r'$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24, labelpad = 8)
#ax1.set_ylabel(r'$d\,n/d\log_{10}\,M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}^{-1}Mpc^{-3}}$]', fontsize = 20)
ax1.set_ylabel(r'$d\,n/d\log_{10}\,M_{\mathrm{Peak}}\,$[$\mathrm{Mpc^{-3}\,dex^{-1}}$]', fontsize = 24, labelpad = 8)
ax1.tick_params(labelsize = 18)
ax1.legend(loc = 'upper right', fontsize = 14, frameon = False, borderpad = 1)

ax1.set_xticks(np.logspace(7, 15, 9))
ax1.set_yticks(np.logspace(-6, 2, 9))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


fig.savefig('/sdf/home/y/ycwang/figures/hmf.png', dpi = 400, bbox_inches = 'tight')
plt.show()
