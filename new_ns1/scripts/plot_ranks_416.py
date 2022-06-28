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
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator, LogLocator
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from scipy.ndimage.filters import gaussian_filter
fig=plt.figure()

h = 0.7
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'
halo_416 = np.loadtxt(src+'/Halo416/output/rockstar/groupcat/sfr_catalog_1.000000.txt')
Mpeak_416 = halo_416[:, 13]
Rank1_416 = halo_416[:, 16]
Mpart_zoom = 2.82 * 1e5 / h

Nbins = 26
y_bins = np.linspace(0.3, 6.8, Nbins+1)
z_bins = np.linspace(-4.33, 4.33, Nbins+1)
y = np.log10(Mpeak_416/Mpart_zoom)
z = Rank1_416
N = np.zeros((Nbins, Nbins))
for i in range(Nbins):
    for j in range(Nbins):
        id_bin = np.where((y >= y_bins[i]) & (y < y_bins[i+1]) &(z >= z_bins[j]) & (z < z_bins[j+1]))[0]
        N[j, i] += len(id_bin)

N = np.asarray(N)
N = np.reshape(N, (Nbins, Nbins))

######

fig.set_size_inches(8, 8)
ax1 = fig.add_subplot(1, 1, 1)
print('min x = ', np.min(y_bins[0]))
ax1.set_xlim([0.2, 7.2])
ax1.set_ylim([-4.8, 4.8])
ax1.set_xlabel(r'$N_{\mathrm{particle, Peak}}$', fontsize = 24)
ax1.set_ylabel(r'$\mathrm{Rank}\ \Delta\,v_{\mathrm{max}}$', fontsize = 24)
ax1.tick_params(labelsize = 20)
xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin

ax1.text(xmax - width * 0.03, ymax - height * 0.03, r'Halo 416',
         horizontalalignment = 'right', verticalalignment = 'top', fontsize = 20, color = 'black', alpha = 0.8)

ax1.tick_params(which='major', labelsize = 20, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax1.set_xticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$'])
ax1.set_yticklabels([r'$-6$', r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$'])

im1 = ax1.imshow(N, cmap = 'Greys', norm = LogNorm(1, 2*np.max(N)),
                 extent = [y_bins[0], y_bins[-1], z_bins[0], z_bins[-1]], 
                 interpolation = 'nearest', aspect='auto', origin = 'lower')

x1 = np.asarray([9950551.401308624, 53085154.94856838, 552326536.7533884, 6432741874.109157, 63535140338.27571, 638508739625.4298])
x2 = x1 / (Mpart_zoom)
x3 = np.linspace(0, 10, 6)


ax2= ax1.twiny()
ax2.set_xticks(np.log10(x2))
ax2.set_xticklabels([r'$10^{0}$', r'$10^{2}$', r'$10^{4}$', r'$10^{6}$', r'$10^{8}$', r'$10^{10}$'])
ax2.tick_params(which='major', labelsize = 20, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax2.set_xlim([0.2, 7.2])
ax2.tick_params(labelsize = 20)
ax2.set_xlabel(r'$\mathrm{Median}\ (M_{\mathrm{\ast, zoom\ in}}/\mathrm{M_{\odot}})$', fontsize = 24, labelpad = 12)

fig.savefig('/sdf/home/y/ycwang/figures/ranks_dot_416.png', dpi = 400, bbox_inches = 'tight')
plt.show()
