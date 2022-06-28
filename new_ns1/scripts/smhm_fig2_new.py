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
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'
baseDir = src+'/hmf'

####################################

data_1024 = np.loadtxt(baseDir+'/Data_1024_z0.txt')

UpID = data_1024[:, 1]
Mstar = data_1024[:, 10]
Vpeak = data_1024[:, 7]
x_1024 = data_1024[:, -6]
y_1024 = data_1024[:, -5]
z_1024 = data_1024[:, -4]
print('1024')
print(np.min(x_1024), np.max(x_1024))
print(np.min(y_1024), np.max(y_1024))
print(np.min(z_1024), np.max(z_1024))
ID_all = np.where(Mstar > 0.)[0]
ID_cent = np.where((UpID == -1) & (Mstar > 0.))[0]
ID_sat = np.where((UpID != -1) & (Mstar > 0.))[0]
ID_all_Vpeak = np.where((Mstar > 0.) & (Vpeak >= 70.))[0]
ID_cent_Vpeak = np.where((UpID == -1) & (Mstar > 0.) & (Vpeak >= 70.))[0]
ID_sat_Vpeak = np.where((UpID != -1) & (Mstar > 0.) & (Vpeak >= 70.))[0]

data_cent_1024 = data_1024[ID_cent]
data_sat_1024 = data_1024[ID_sat]
data_1024v = data_1024[ID_all_Vpeak]
data_cent_1024v = data_1024[ID_cent_Vpeak]
data_sat_1024v = data_1024[ID_sat_Vpeak]
data_1024 = data_1024[ID_all]

####################################
###Full, no cut

lg_mhalo = np.log10(data_1024[:, 4])
lg_mstar = np.log10(data_1024[:, 12])
print(lg_mhalo, lg_mstar)
nbins = 12
bins_x = np.linspace(np.min(lg_mhalo), np.max(lg_mhalo), nbins)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_mhalo) - bins_h, np.max(lg_mhalo) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_1024 = bins_x
data_bin_y_1024 = bin_means
data_bin_y_std_1024 = bin_std
data_bin_y_med_1024 = bin_med
data_bin_y_low_1024 = bin_low
data_bin_y_high_1024 = bin_high

######################################

data_2048 = np.loadtxt(baseDir+'/Data_2048_z0.txt')

UpID = data_2048[:, 1]
Mstar = data_2048[:, 12]
Vpeak = data_2048[:, 7]
x_2048 = data_2048[:, -6]
y_2048 = data_2048[:, -5]
z_2048 = data_2048[:, -4]
print('2048')
print(np.min(x_2048), np.max(x_2048))
print(np.min(y_2048), np.max(y_2048))
print(np.min(z_2048), np.max(z_2048))
ID_all = np.where(Mstar > 0.)[0]
ID_cent = np.where((UpID == -1) & (Mstar > 0.))[0]
ID_sat = np.where((UpID != -1) & (Mstar > 0.))[0]
ID_all_Vpeak = np.where((Mstar > 0.) & (Vpeak >= 35.))[0]
ID_cent_Vpeak = np.where((UpID == -1) & (Mstar > 0.) & (Vpeak >= 35.))[0]
ID_sat_Vpeak = np.where((UpID != -1) & (Mstar > 0.) & (Vpeak >= 35.))[0]

data_cent_2048 = data_2048[ID_cent]
data_sat_2048 = data_2048[ID_sat]
data_2048v = data_2048[ID_all_Vpeak]
data_cent_2048v = data_2048[ID_cent_Vpeak]
data_sat_2048v = data_2048[ID_sat_Vpeak]
data_2048 = data_2048[ID_all]

####################################
###Full, no cut

lg_mhalo = np.log10(data_2048[:, 4])
#lg_mstar = np.log10(data_2048[:, 10])
lg_mstar = np.log10(data_2048[:, 12])

nbins = 16
bins_x = np.linspace(np.min(lg_mhalo), np.max(lg_mhalo), nbins)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_mhalo) - bins_h, np.max(lg_mhalo) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_2048 = bins_x
data_bin_y_2048 = bin_means
data_bin_y_std_2048 = bin_std
data_bin_y_med_2048 = bin_med
data_bin_y_low_2048 = bin_low
data_bin_y_high_2048 = bin_high

######################################

data_zoom = np.loadtxt(wr+'/Data_join_MWmass.txt')

UpID = data_zoom[:, 1]
Mstar = data_zoom[:, 12]
Vpeak = data_zoom[:, 7]
x_zoom = data_zoom[:, -6]
y_zoom = data_zoom[:, -5]
z_zoom = data_zoom[:, -4]
print('zoom')
print(np.min(x_zoom), np.max(x_zoom))
print(np.min(y_zoom), np.max(y_zoom))
print(np.min(z_zoom), np.max(z_zoom))
ID_all = np.where(Mstar > 0.)[0]
ID_cent = np.where((UpID == -1) & (Mstar > 0.))[0]
ID_sat = np.where((UpID != -1) & (Mstar > 0.))[0]
ID_all_Vpeak = np.where((Mstar > 0.) & (Vpeak >= 10.))[0]
ID_cent_Vpeak = np.where((UpID == -1) & (Mstar > 0.) & (Vpeak >= 10.))[0]
ID_sat_Vpeak = np.where((UpID != -1) & (Mstar > 0.) & (Vpeak >= 10.))[0]

data_cent_zoom = data_zoom[ID_cent]
data_sat_zoom = data_zoom[ID_sat]
data_zoomv = data_zoom[ID_all_Vpeak]
data_cent_zoomv = data_zoom[ID_cent_Vpeak]
data_sat_zoomv = data_zoom[ID_sat_Vpeak]
data_zoom = data_zoom[ID_all]

####################################
###Full, no cut

lg_mhalo = np.log10(data_zoom[:, 13])
#lg_mstar = np.log10(data_zoom[:, 10])
lg_mstar = np.log10(data_zoom[:, 23])

nbins = 16
bins_x = np.linspace(np.min(lg_mhalo), np.max(lg_mhalo), nbins)
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(np.min(lg_mhalo) - bins_h, np.max(lg_mhalo) + bins_h, nbins+1)
bin_tot = np.digitize(lg_mhalo, bins)
bin_means_tot = np.asarray([lg_mhalo[bin_tot == i].mean() for i in range(1, len(bins))])

bin_means = np.asarray([lg_mstar[bin_tot == i].mean() for i in range(1, len(bins))])
bin_std = np.asarray([lg_mstar[bin_tot == i].std() for i in range(1, len(bins))])
bin_med = np.asarray([np.percentile(lg_mstar[bin_tot == i], 50) for i in range(1, len(bins))])
bin_low = np.asarray([np.percentile(lg_mstar[bin_tot == i], 16) for i in range(1, len(bins))])
bin_high = np.asarray([np.percentile(lg_mstar[bin_tot == i], 84) for i in range(1, len(bins))])

data_bin_x_zoom = bins_x
data_bin_y_zoom = bin_means
data_bin_y_std_zoom = bin_std
data_bin_y_med_zoom = bin_med
data_bin_y_low_zoom = bin_low
data_bin_y_high_zoom = bin_high

######################################

fi_lmc = np.loadtxt(baseDir+'/TNG/fi_lmc.txt')
fi_x = fi_lmc[:, 0]
fi_y = fi_lmc[:, 1]

fi_1s_x = np.log10(fi_x[:7])
fi_2s_x= np.log10(fi_x[14:21])
fi_1s_yhigh = np.log10(fi_y[:7])
fi_1s_ylow = np.log10(fi_y[7:14])
fi_2s_yhigh = np.log10(fi_y[14:21])
fi_2s_ylow = np.log10(fi_y[21:28])

UM_z0 = np.loadtxt(baseDir+'/TNG/UM_z0.txt')
UM_x0 = np.log10(UM_z0[:, 0])
UM_y0 = np.log10(UM_z0[:, 1])

UM_z2 = np.loadtxt(baseDir+'/TNG/UM_z2.txt')
UM_x2 = np.log10(UM_z2[:, 0])
UM_y2 = np.log10(UM_z2[:, 1])

AM_z0 = np.loadtxt(baseDir+'/TNG/AM_z0.txt')
AM_x0 = np.log10(AM_z0[:, 0])
AM_y0 = np.log10(AM_z0[:, 1])

AM_z2 = np.loadtxt(baseDir+'/TNG/AM_z2.txt')
AM_x2 = np.log10(AM_z2[:, 0])
AM_y2 = np.log10(AM_z2[:, 1])

#Watkins 2019
MW_med = 1.54 * 1e12 
MW_high = (1.54+0.75) * 1e12
MW_low = (1.54-0.44) * 1e12

#Licquia 2015
Mstar_med = 6.08 * 1e10
Mstar_low = (6.08 - 1.14) * 1e10
Mstar_high = (6.08 + 1.14) * 1e10

######

fig.set_size_inches(8, 6)
ax1 = fig.add_subplot(1, 1, 1)

#plot for c125-1024
ax1.plot(10**data_bin_x_1024, 10**data_bin_y_med_1024, color = 'deepskyblue', 
         dashes = (5, 2.4), alpha = 0.8, linewidth = 1.2, label = r'c125-1024')
#ax1.fill_between(data_bin_x_1024, data_bin_y_low_1024, data_bin_y_high_1024, facecolor='springgreen', alpha=0.3)   
#ax1.plot(data_bin_x_1024, data_bin_y_low_1024, color='seagreen', alpha=0.5, linewidth = 0.5, linestyle = ':')   
#ax1.plot(data_bin_x_1024, data_bin_y_high_1024, color='seagreen', alpha=0.5, linewidth = 0.5, linestyle = ':')   

#plot for c125-2048
ax1.plot(10**data_bin_x_2048, 10**data_bin_y_med_2048, color = 'royalblue', 
         linestyle = '-.', alpha = 0.8, linewidth = 1.2, label = r'c125-2048')
#ax1.fill_between(data_bin_x_2048, data_bin_y_low_2048, data_bin_y_high_2048, facecolor='tomato', alpha=0.3)   
#ax1.plot(data_bin_x_2048, data_bin_y_low_2048, color='crimson', alpha=0.5, linewidth = 0.5, linestyle = ':')   
#ax1.plot(data_bin_x_2048, data_bin_y_high_2048, color='crimson', alpha=0.5, linewidth = 0.5, linestyle = ':')   

#plot for zoom-in
ax1.plot(10**data_bin_x_zoom, 10**data_bin_y_med_zoom, color = 'navy', 
         alpha = 0.96, linewidth = 1.6, label = r'Zoom-in')
ax1.fill_between(10**data_bin_x_zoom, 10**data_bin_y_low_zoom, 10**data_bin_y_high_zoom, facecolor='deepskyblue', alpha=0.3)   
#ax1.plot(data_bin_x_zoom, data_bin_y_low_zoom, color='royalblue', alpha=0.5, linewidth = 0.5, linestyle = ':')   
#ax1.plot(data_bin_x_zoom, data_bin_y_high_zoom, color='royalblue', alpha=0.5, linewidth = 0.5, linestyle = ':')   

ax1.errorbar([MW_med], [Mstar_med], xerr = [[MW_med - MW_low], [MW_high - MW_med]], 
             yerr = [[Mstar_med - Mstar_low], [Mstar_high - Mstar_med]], fmt = '*', color = 'crimson', 
             alpha = 0.6, markersize = 8, capsize = 4.2, label = r'Milky Way')

ax1.legend(loc = 'upper left', fontsize = 14, frameon = False, borderpad = 1)
ax1.set_xlabel(r'$M_{\mathrm{Peak}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24, labelpad = 8)
ax1.set_ylabel(r'$M_{\mathrm{\ast}}\,$[$\mathrm{M_{\odot}}$]', fontsize = 24, labelpad = 8)
ax1.plot([10**11.289506953223768, 10**15.3], [10**9.002881844380404, 10**9.002881844380404], 
         alpha = 1., color = 'black', lw = 0.8, linestyle = ':')

ax1.plot([10**10.30847029077118, 10**15.3], [10**7.00864553314121, 10**7.00864553314121], 
         alpha = 0.6, color = 'black', lw = 0.8, linestyle = ':')

ax1.plot([10**9.281921618204807, 10**15.3], [10**5.002881844380404, 10**5.002881844380404], 
         alpha = 0.5, color = 'dimgray', lw = 0.8, linestyle = ':')

ax1.plot([10**7.6991150442477885, 10**15.3], [10**2.005763688760803, 10**2.005763688760803], 
         alpha = 0.4, color = 'dimgray', lw = 0.8, linestyle = ':')


ax1.fill_between([10**10.30847029077118, 10**11.289506953223768], [10**(-1.3), 10**(-1.3)],
                 [10**7.00864553314121, 10**9.002881844380404], color = 'black', lw = 0., alpha = 0.32)
ax1.fill_between([10**9.281921618204807, 10**10.30847029077118], [10**(-1.3), 10**(-1.3)],
                 [10**5.002881844380404, 10**7.00864553314121], color = 'dimgray', lw = 0., alpha = 0.24)
ax1.fill_between([10**7.6991150442477885, 10**9.281921618204807], [10**(-1.3), 10**(-1.3)],
                 [10**2.005763688760803, 10**5.002881844380404], color = 'dimgray', lw = 0., alpha = 0.1)
ax1.set_xlim([10**7, 10**15.2])

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim([10**7, 10**15.2])
ax1.set_ylim([10**(-0.8), 10**12.2])
ax1.set_xticks(np.logspace(7, 15, 9))
ax1.set_yticks(np.logspace(0, 12, 7))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.yaxis.set_minor_locator(loc2)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 18, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


fig.savefig('/sdf/home/y/ycwang/figures/smhm_fig2.png', dpi = 400, bbox_inches = 'tight')
plt.show()