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
from matplotlib.ticker import AutoMinorLocator
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from scipy.ndimage.filters import gaussian_filter
fig=plt.figure()

h = 0.7
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

halos_2048 = np.loadtxt(src+'/hmf/Data_2048_z0.txt')
halos_1024 = np.loadtxt(src+'/hmf/Data_1024_z0.txt')
halos_zoom = np.loadtxt(wr+'/Data_join_MWmass.txt')
Mpeak_2048 = halos_2048[:, 4]
Rank1_2048 = halos_2048[:, 9]
Mpeak_1024 = halos_1024[:, 4]
Rank1_1024 = halos_1024[:, 9]
Mpeak_zoom = halos_zoom[:, 13]
Rank1_zoom = halos_zoom[:, 16]
Mpart_2048 = 1.8 * 1e7 / h
Mpart_1024 = 1.44 * 1e8 / h
Mpart_zoom = 2.82 * 1e5 / h

Nbins = 26
y_bins = np.linspace(0.3, 6.8, Nbins+1)
z_bins = np.linspace(-4.33, 4.33, Nbins+1)
y = np.log10(Mpeak_zoom/Mpart_zoom)
z = Rank1_zoom
N = np.zeros((Nbins, Nbins))
for i in range(Nbins):
    for j in range(Nbins):
        id_bin = np.where((y >= y_bins[i]) & (y < y_bins[i+1]) &(z >= z_bins[j]) & (z < z_bins[j+1]))[0]
        N[j, i] += len(id_bin)

N = np.asarray(N)
N = np.reshape(N, (Nbins, Nbins))

######

Nbins = 101
Z_2048, X_2048, Y_2048 = np.histogram2d(np.log10(Mpeak_2048/Mpart_2048), Rank1_2048, bins = Nbins)
Z_2048 = gaussian_filter(Z_2048.T, 0.8)

Z_1024, X_1024, Y_1024 = np.histogram2d(np.log10(Mpeak_1024/Mpart_1024), Rank1_1024, bins = Nbins)
Z_1024 = gaussian_filter(Z_1024.T, 0.8)

Z_zoom, X_zoom, Y_zoom = np.histogram2d(np.log10(Mpeak_zoom/Mpart_zoom), Rank1_zoom, bins = Nbins)
Z_zoom = gaussian_filter(Z_zoom.T, 0.8)

#####################################

fig.set_size_inches(8, 8)
ax1 = fig.add_subplot(1, 1, 1)

color_range = [np.min(Z_2048), np.max(Z_2048)]
c1 = ax1.contour(X_2048[:-1], Y_2048[:-1], Z_2048, cmap = 'Reds', levels = 8, 
                 norm=matplotlib.colors.Normalize(color_range[0], color_range[1]))
idm_2048 = np.argmax(Z_2048)
x2048, y2048 = np.meshgrid(X_2048[:-1], Y_2048[:-1])
x_2048 = x2048.flatten()
y_2048 = y2048.flatten()
#ax1.scatter(x_2048[idm_2048], y_2048[idm_2048], s = 82, alpha = 0.8, marker = 'o', lw = 0., color = 'crimson', label = r'c125-2048 ($z=0$)')

color_range = [np.min(Z_1024), np.max(Z_1024)]
c2 = ax1.contour(X_1024[:-1], Y_1024[:-1], Z_1024, cmap = 'Greens', levels = 8, 
                 norm=matplotlib.colors.Normalize(color_range[0], color_range[1]))
idm_1024 = np.argmax(Z_1024)
x1024, y1024 = np.meshgrid(X_1024[:-1], Y_1024[:-1])
x_1024 = x1024.flatten()
y_1024 = y1024.flatten()
#ax1.scatter(x_1024[idm_1024], y_1024[idm_1024], s = 82, alpha = 0.8, marker = '^', lw = 0., color = 'springgreen', label = r'c125-1024 ($z=0$)')

color_range = [np.min(Z_zoom), np.max(Z_zoom)]
c3 = ax1.contour(X_zoom[:-1], Y_zoom[:-1], Z_zoom, cmap = 'Blues', levels = 8, 
                 norm=matplotlib.colors.Normalize(color_range[0], color_range[1]))

im1 = ax1.imshow(N, cmap = 'Greys', norm=LogNorm(1, np.max(N)),
                 extent = [y_bins[0], y_bins[-1], z_bins[0], z_bins[-1]], 
                 interpolation = 'nearest', aspect='auto', origin = 'lower')
idm_zoom = np.argmax(Z_zoom)
xzoom, yzoom = np.meshgrid(X_zoom[:-1], Y_zoom[:-1])
x_zoom = xzoom.flatten()
y_zoom = yzoom.flatten()
#ax1.scatter(x_zoom[idm_zoom], y_zoom[idm_zoom], s = 82, alpha = 0.8, 
#            marker = 's', lw = 0., color = 'gold', label = r'Joined MW resims ($z=0$)')

#ax1.set_xlabel(r'$\log_{10}\,(M_{\mathrm{Peak}}/m_{\mathrm{DM}})\,$', fontsize = 20)
ax1.set_xlabel(r'$N_{\mathrm{particle, Peak}}$', fontsize = 24)
ax1.set_ylabel(r'$\mathrm{Rank}\ \Delta\,v_{\mathrm{max}}$', fontsize = 24)
ax1.tick_params(which='major', labelsize = 20, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
#ax2.set_xticklabels(x3)
ax1.set_xticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$', r'$10^{6}$', r'$10^{7}$'])
ax1.set_yticklabels([r'$-6$', r'$-4$', r'$-2$', r'$0$', r'$2$', r'$4$'])

h1,_ = c1.legend_elements()
h2,_ = c2.legend_elements()
h3,_ = c3.legend_elements()
ax1.legend([h1[4], h2[4], h3[4]], [r'c125-2048', r'c125-1024', r'Zoom-in'],
           loc = 'upper right', fontsize = 16, frameon = False, borderpad = 0.8)


#ax1.legend(loc = 'upper right', fontsize = 14, frameon = False, borderpad = 0.8)
ax1.set_xlim([0.2, 7.2])
ax1.set_ylim([-4.8, 4.8])

x1 = np.asarray([9950551.401308624, 53085154.94856838, 552326536.7533884, 6432741874.109157, 63535140338.27571, 638508739625.4298])
x2 = x1 / (Mpart_zoom)
x3 = np.linspace(0, 10, 6)

#Mstar ticks
ax2= ax1.twiny()
#ax2.set_xscale('log')
ax2.set_xticks(np.log10(x2))
#ax2.set_xticklabels(x3)
ax2.set_xticklabels([r'$10^{0}$', r'$10^{2}$', r'$10^{4}$', r'$10^{6}$', r'$10^{8}$', r'$10^{10}$'])
ax2.tick_params(which='major', labelsize = 20, width = 1., length = 6, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
#ax2.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = False, top = False, left = True, right = True)

ax2.set_xlim([0.2, 7.2])
ax2.tick_params(labelsize = 20)
#ax2.ticklabel_format(axis = 'x', style = 'scientific', useOffset = False, useMathText = True)
ax2.set_xlabel(r'$\mathrm{Median}\ (M_{\mathrm{\ast, zoom\ in}}/\mathrm{M_{\odot}})$', fontsize = 24, labelpad = 12)

fig.savefig('/sdf/home/y/ycwang/figures/ranks_cmap.png', dpi = 400, bbox_inches = 'tight')
plt.show()
