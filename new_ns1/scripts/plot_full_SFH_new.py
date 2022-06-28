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
from matplotlib.ticker import AutoMinorLocator
#from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from astropy.stats import bootstrap
fig=plt.figure()

from astropy.cosmology import FlatLambdaCDM
#from astropy import units as u
cosmo = FlatLambdaCDM(H0=70.00, Om0=0.286)

h = 0.7
omega_m = 0.286
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

##########################
#1024

SFH5_1024 = np.loadtxt(src+'/1024_saga/Vmax5_all.txt')
SFH6_1024 = np.loadtxt(src+'/1024_saga/Vmax6_all.txt')
SFH7_1024 = np.loadtxt(src+'/1024_saga/Vmax7_all.txt')
SFH8_1024 = np.loadtxt(src+'/1024_saga/Vmax8_all.txt')

##########################
#2048

SFH3_2048 = np.loadtxt(src+'/2048_saga/Vmax3_all.txt')
SFH4_2048 = np.loadtxt(src+'/2048_saga/Vmax4_all.txt')
SFH5_2048 = np.loadtxt(src+'/2048_saga/Vmax5_all.txt')
SFH6_2048 = np.loadtxt(src+'/2048_saga/Vmax6_all.txt')
SFH7_2048 = np.loadtxt(src+'/2048_saga/Vmax7_all.txt')
SFH8_2048 = np.loadtxt(src+'/2048_saga/Vmax8_all.txt')

##########################
#zoom

SFH1_zoom = np.loadtxt(wr+'/Vmax1_all.txt')
SFH2_zoom = np.loadtxt(wr+'/Vmax2_all.txt')
SFH3_zoom = np.loadtxt(wr+'/Vmax3_all.txt')
SFH4_zoom = np.loadtxt(wr+'/Vmax4_all.txt')
SFH5_zoom = np.loadtxt(wr+'/Vmax5_all.txt')
SFH6_zoom = np.loadtxt(wr+'/Vmax6_all.txt')
SFH7_zoom = np.loadtxt(wr+'/Vmax7_all.txt')

SFH1_zoom_sat = np.loadtxt(wr+'/Vmax1_sat.txt')
SFH2_zoom_sat = np.loadtxt(wr+'/Vmax2_sat.txt')
SFH1_zoom_cent = np.loadtxt(wr+'/Vmax1_cent.txt')
SFH2_zoom_cent = np.loadtxt(wr+'/Vmax2_cent.txt')

##########################

fig.set_size_inches(16, 22)
plt.subplots_adjust(wspace = 0.04, hspace = 0.06)

cosmo = FlatLambdaCDM(H0=70, Om0=0.286)
z_lbt = np.asarray([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 4, 8])
lbt = cosmo.lookback_time(z_lbt)
LBT = np.linspace(0, 12, 7)

ax1 = fig.add_subplot(4, 2, 1)
ax1.set_ylabel(r'$(M_{\ast}/M_{\ast, 0})_{\mathrm{raw}}$', fontsize = 24, labelpad = 8)
ax1.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax1.set_xticks(lbt.value)
ax1.set_xticklabels(z_lbt)
plt.setp(ax1.get_xticklabels(), visible = False)
ax1.set_xlim([13.8, -0.5])
ax1.set_xticklabels([])

ax12= ax1.twiny()
ax12.set_xticks(LBT[::-1])
ax12.set_xlim([13.8, -0.5])
ax12.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax12.set_xlabel('Lookback time [Gyr]', fontsize = 24, labelpad = 8)

ax5 = fig.add_subplot(4, 2, 2)
ax5.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax5.set_xticks(lbt.value)
ax5.set_xticklabels(z_lbt)
plt.setp(ax5.get_xticklabels(), visible = False)
plt.setp(ax5.get_yticklabels(), visible = False)
ax5.set_xlim([13.8, -0.5])
ax5.set_xticklabels([])

ax52= ax5.twiny()
ax52.set_xticks(LBT[::-1])
ax52.set_xlim([13.8, -0.5])
ax52.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax52.set_xlabel('Lookback time [Gyr]', fontsize = 24, labelpad = 8)

ax2 = fig.add_subplot(4, 2, 3)
ax2.set_ylabel(r'$(M_{\ast}/M_{\ast, 0})_{\mathrm{raw}}$', fontsize = 24, labelpad = 8)
ax2.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax2.set_xticks(lbt.value)
ax2.set_xticklabels(z_lbt)
plt.setp(ax2.get_xticklabels(), visible = False)
ax2.set_xlim([13.8, -0.5])
ax2.set_xticklabels([])

ax22= ax2.twiny()
ax22.set_xticks(LBT[::-1])
ax22.set_xlim([13.8, -0.5])
ax22.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax22.set_xticklabels([])

ax6 = fig.add_subplot(4, 2, 4)
ax6.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax6.set_xticks(lbt.value)
ax6.set_xticklabels(z_lbt)
plt.setp(ax6.get_xticklabels(), visible = False)
plt.setp(ax6.get_yticklabels(), visible = False)
ax6.set_xlim([13.8, -0.5])
ax6.set_xticklabels([])

ax62= ax6.twiny()
ax62.set_xticks(LBT[::-1])
ax62.set_xlim([13.8, -0.5])
ax62.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax62.set_xticklabels([])

ax3 = fig.add_subplot(4, 2, 5)
ax3.set_ylabel(r'$(M_{\ast}/M_{\ast, 0})_{\mathrm{raw}}$', fontsize = 24, labelpad = 8)
ax3.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax3.set_xticks(lbt.value)
ax3.set_xticklabels(z_lbt)
plt.setp(ax3.get_xticklabels(), visible = False)
ax3.set_xlim([13.8, -0.5])
ax3.set_xticklabels([])

ax32= ax3.twiny()
ax32.set_xticks(LBT[::-1])
ax32.set_xlim([13.8, -0.5])
ax32.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax32.set_xticklabels([])

ax7 = fig.add_subplot(4, 2, 6)
ax7.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax7.set_xticks(lbt.value)
ax7.set_xticklabels(z_lbt)
plt.setp(ax7.get_xticklabels(), visible = False)
plt.setp(ax7.get_yticklabels(), visible = False)
ax7.set_xlim([13.8, -0.5])
ax7.set_xticklabels([])

ax72= ax7.twiny()
ax72.set_xticks(LBT[::-1])
ax72.set_xlim([13.8, -0.5])
ax72.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax72.set_xticklabels([])

ax4 = fig.add_subplot(4, 2, 7)
ax4.set_xlabel(r'$z$', fontsize = 24, labelpad = 8)
ax4.set_ylabel(r'$(M_{\ast}/M_{\ast, 0})_{\mathrm{raw}}$', fontsize = 24, labelpad = 8)
ax4.tick_params(which='major', labelsize = 16, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax4.set_xticks(lbt.value)
ax4.set_xticklabels(z_lbt)
ax4.set_xlim([13.8, -0.5])

ax42= ax4.twiny()
ax42.set_xticks(LBT[::-1])
ax42.set_xlim([13.8, -0.5])
ax42.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax42.set_xticklabels([])

ax8 = fig.add_subplot(4, 2, 8)
ax8.tick_params(which='major', labelsize = 16, width = 1., length = 4, direction='in', pad = 4, bottom = True, top = False, left = True, right = True)
ax8.set_xlabel(r'$z$', fontsize = 24, labelpad = 8)
plt.setp(ax8.get_yticklabels(), visible = False)
ax8.set_xticks(lbt.value)
ax8.set_xticklabels(z_lbt)
ax8.set_xlim([13.8, -0.5])

ax82= ax8.twiny()
ax82.set_xticks(LBT[::-1])
ax82.set_xlim([13.8, -0.5])
ax82.tick_params(which='major', labelsize = 18, width = 1., length = 4, direction='in', pad = 4, bottom = False, top = True, left = True, right = True)
ax82.set_xticklabels([])

#1024
scales = np.loadtxt(src+'/1024_saga/scales.txt')
scales = scales[:, 1]
z = 1/scales - 1
t = cosmo.lookback_time(z).value
ax5.plot(SFH5_1024[:, 0], SFH5_1024[:, 13], color = 'darkturquoise', linewidth = 1.2, alpha = 0.8, dashes = (5, 2.4), label = r'c125-1024 (all)')
ax6.plot(SFH6_1024[:, 0], SFH6_1024[:, 13], color = 'darkturquoise', linewidth = 1.2, alpha = 0.8, dashes = (5, 2.4), label = r'c125-1024 (all)')
ax7.plot(SFH7_1024[:, 0], SFH7_1024[:, 13], color = 'darkturquoise', linewidth = 1.2, alpha = 0.8, dashes = (5, 2.4), label = r'c125-1024 (all)')
ax8.plot(SFH8_1024[:, 0], SFH8_1024[:, 13], color = 'darkturquoise', linewidth = 1.2, alpha = 0.8, dashes = (5, 2.4), label = r'c125-1024 (all)')
ax5.fill_between(SFH5_1024[:, 0], SFH5_1024[:, 14], SFH5_1024[:, 15], color = 'springgreen', alpha = 0.2)
ax6.fill_between(SFH6_1024[:, 0], SFH6_1024[:, 14], SFH6_1024[:, 15], color = 'springgreen', alpha = 0.2)
ax7.fill_between(SFH7_1024[:, 0], SFH7_1024[:, 14], SFH7_1024[:, 15], color = 'springgreen', alpha = 0.2)
ax8.fill_between(SFH8_1024[:, 0], SFH8_1024[:, 14], SFH8_1024[:, 15], color = 'springgreen', alpha = 0.2)
ax5.scatter(t[39], -0.04,  color = 'darkturquoise', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax6.scatter(t[27], -0.04,  color = 'darkturquoise', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax7.scatter(t[19], -0.04,  color = 'darkturquoise', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax8.scatter(t[11], -0.06,  color = 'darkturquoise', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)

#2048
scales = np.loadtxt(src+'/2048_saga/scales.txt')
scales = scales[:, 1]
z = 1/scales - 1
t = cosmo.lookback_time(z).value
ax3.plot(SFH3_2048[:, 0], SFH3_2048[:, 13], color = 'crimson', linewidth = 1.2, linestyle = '-.', alpha = 0.8, label = r'c125-2048 (all)')
ax4.plot(SFH4_2048[:, 0], SFH4_2048[:, 13], color = 'crimson', linewidth = 1.2, linestyle = '-.', alpha = 0.8, label = r'c125-2048 (all)')
ax5.plot(SFH5_2048[:, 0], SFH5_2048[:, 13], color = 'crimson', linewidth = 1.2, linestyle = '-.', alpha = 0.8, label = r'c125-2048 (all)')
ax6.plot(SFH6_2048[:, 0], SFH6_2048[:, 13], color = 'crimson', linewidth = 1.2, linestyle = '-.', alpha = 0.8, label = r'c125-2048 (all)')
ax7.plot(SFH7_2048[:, 0], SFH7_2048[:, 13], color = 'crimson', linewidth = 1.2, linestyle = '-.', alpha = 0.8, label = r'c125-2048 (all)')
ax8.plot(SFH8_2048[:, 0], SFH8_2048[:, 13], color = 'crimson', linewidth = 1.2, linestyle = '-.', alpha = 0.8, label = r'c125-2048 (all)')
ax3.fill_between(SFH3_2048[:, 0], SFH3_2048[:, 14], SFH3_2048[:, 15], color = 'tomato', alpha = 0.2)
ax4.fill_between(SFH4_2048[:, 0], SFH4_2048[:, 14], SFH4_2048[:, 15], color = 'tomato', alpha = 0.2)
ax5.fill_between(SFH5_2048[:, 0], SFH5_2048[:, 14], SFH5_2048[:, 15], color = 'tomato', alpha = 0.2)
ax6.fill_between(SFH6_2048[:, 0], SFH6_2048[:, 14], SFH6_2048[:, 15], color = 'tomato', alpha = 0.2)
ax7.fill_between(SFH7_2048[:, 0], SFH7_2048[:, 14], SFH7_2048[:, 15], color = 'tomato', alpha = 0.2)
ax8.fill_between(SFH8_2048[:, 0], SFH8_2048[:, 14], SFH8_2048[:, 15], color = 'tomato', alpha = 0.2)
ax3.scatter(t[47], -0.04,  color = 'crimson', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax4.scatter(t[23], -0.04,  color = 'crimson', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax5.scatter(t[14], -0.04,  color = 'crimson', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax6.scatter(t[8], -0.04,  color = 'crimson', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax7.scatter(t[3], -0.06,  color = 'crimson', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax8.scatter(t[0], -0.04,  color = 'crimson', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)

#zoom
sz_zoom = np.loadtxt(src+'/Halo416/output/rockstar/outputs/scales.txt')
scales = sz_zoom[:, 1]
z = 1/scales - 1
t = cosmo.lookback_time(z).value
ax1.plot(SFH1_zoom[:, 0], SFH1_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in (all)')
ax2.plot(SFH2_zoom[:, 0], SFH2_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in (all)')
ax3.plot(SFH3_zoom[:, 0], SFH3_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in')
ax4.plot(SFH4_zoom[:, 0], SFH4_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in')
ax5.plot(SFH5_zoom[:, 0], SFH5_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in')
ax6.plot(SFH6_zoom[:, 0], SFH6_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in')
ax7.plot(SFH7_zoom[:, 0], SFH7_zoom[:, 13], color = 'navy', linewidth = 1.2, alpha = 0.8, label = r'Zoom-in')
ax1.fill_between(SFH1_zoom[:, 0], SFH1_zoom[:, 14], SFH1_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)
ax2.fill_between(SFH2_zoom[:, 0], SFH2_zoom[:, 14], SFH2_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)
ax3.fill_between(SFH3_zoom[:, 0], SFH3_zoom[:, 14], SFH3_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)
ax4.fill_between(SFH4_zoom[:, 0], SFH4_zoom[:, 14], SFH4_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)
ax5.fill_between(SFH5_zoom[:, 0], SFH5_zoom[:, 14], SFH5_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)
ax6.fill_between(SFH6_zoom[:, 0], SFH6_zoom[:, 14], SFH6_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)
ax7.fill_between(SFH7_zoom[:, 0], SFH7_zoom[:, 14], SFH7_zoom[:, 15], color = 'deepskyblue', alpha = 0.2)

ax1.plot(SFH1_zoom_sat[:, 0], SFH1_zoom_sat[:, 13], color = 'navy', dashes = (5, 2.4), linewidth = 1.2, alpha = 0.8, label = r'Satellites')
ax2.plot(SFH2_zoom_sat[:, 0], SFH2_zoom_sat[:, 13], color = 'navy', dashes = (5, 2.4), linewidth = 1.2, alpha = 0.8, label = r'Satellites')
ax1.plot(SFH1_zoom_cent[:, 0], SFH1_zoom_cent[:, 13], color = 'navy', linestyle = '-.', linewidth = 1.2, alpha = 0.8, label = r'Centrals')
ax2.plot(SFH2_zoom_cent[:, 0], SFH2_zoom_cent[:, 13], color = 'navy', linestyle = '-.', linewidth = 1.2, alpha = 0.8, label = r'Centrals')
ax1.scatter(t[40], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax2.scatter(t[24], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax3.scatter(t[14], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax4.scatter(t[6], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax5.scatter(t[1], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax6.scatter(t[0], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)
ax7.scatter(t[0], -0.04,  color = 'navy', alpha = 0.8, s = 64, marker = '^', lw = 0., rasterized = True)

#######################
#Weisz 2014 data

W = np.loadtxt(src+'/hmf/TNG/Weisz_14.txt')

Mstar = W[:, -1] * 1e6
ID1 = np.where((Mstar >= 1e4) & (Mstar <1e5))[0]
#ID1 = np.where((Mstar <1e5))[0]
ID2 = np.where((Mstar >= 1e5) & (Mstar <1e6))[0]
ID3 = np.where((Mstar >= 1e6) & (Mstar <1e7))[0]
ID4 = np.where((Mstar >= 1e7) & (Mstar <1e8))[0]
#ID4 = np.where((Mstar >= 1e7))[0]
#print(len(ID1), len(ID2), len(ID3), len(ID4))

sfh = W[:, 5::5]
SFH = sfh[:, :-1]
p1 = np.linspace(10.1, 8.7, 29)
p2 = np.linspace(8.6, 6.6, 21)
p = np.concatenate((p1, p2))
tw = 10**p / 1e9
#print(tw)
#print(len(tw), len(SFH[0]))

SFH1 = SFH[ID1]
SFH2 = SFH[ID2]
SFH3 = SFH[ID3]
SFH4 = SFH[ID4]

Wei_sfh1_med = np.percentile(SFH1, 50, axis = 0)
Wei_sfh1_low = np.percentile(SFH1, 16, axis = 0)
Wei_sfh1_high = np.percentile(SFH1, 84, axis = 0)

Wei_sfh2_med = np.percentile(SFH2, 50, axis = 0)
Wei_sfh2_low = np.percentile(SFH2, 16, axis = 0)
Wei_sfh2_high = np.percentile(SFH2, 84, axis = 0)

Wei_sfh3_med = np.percentile(SFH3, 50, axis = 0)
Wei_sfh3_low = np.percentile(SFH3, 16, axis = 0)
Wei_sfh3_high = np.percentile(SFH3, 84, axis = 0)

Wei_sfh4_med = np.percentile(SFH4, 50, axis = 0)
Wei_sfh4_low = np.percentile(SFH4, 16, axis = 0)
Wei_sfh4_high = np.percentile(SFH4, 84, axis = 0)

ax1.plot(tw, Wei_sfh1_med, color = 'black', alpha = 0.6, lw = 2., linestyle = ':', label = r'Weisz+2014')
ax2.plot(tw, Wei_sfh2_med, color = 'black', alpha = 0.6, lw = 2., linestyle = ':', label = r'Weisz+2014')
ax3.plot(tw, Wei_sfh3_med, color = 'black', alpha = 0.6, lw = 2., linestyle = ':', label = r'Weisz+2014')
ax4.plot(tw, Wei_sfh4_med, color = 'black', alpha = 0.6, lw = 2., linestyle = ':', label = r'Weisz+2014')
ax1.fill_between(tw, Wei_sfh1_low, Wei_sfh1_high, color = 'black', alpha = 0.1)
ax2.fill_between(tw, Wei_sfh2_low, Wei_sfh2_high, color = 'black', alpha = 0.1)
ax3.fill_between(tw, Wei_sfh3_low, Wei_sfh3_high, color = 'black', alpha = 0.1)
ax4.fill_between(tw, Wei_sfh4_low, Wei_sfh4_high, color = 'black', alpha = 0.1)

#######################

SFH1_zoom = np.loadtxt(wr+'/Vmax1_np.txt')
SFH2_zoom = np.loadtxt(wr+'/Vmax2_np.txt')
SFH1_zoom_sat = np.loadtxt(wr+'/Vmax1_np_sat.txt')
SFH2_zoom_sat = np.loadtxt(wr+'/Vmax2_np_sat.txt')
SFH1_zoom_cent = np.loadtxt(wr+'/Vmax1_np_cent.txt')
SFH2_zoom_cent = np.loadtxt(wr+'/Vmax2_np_cent.txt')

z_lbti = np.asarray([0,  0.5, 1,  2, 8])
lbti = cosmo.lookback_time(z_lbti)

axins1 = ax1.inset_axes([0.64, 0.12, 0.32, 0.28])
axins1.plot(SFH1_zoom[:, 0], SFH1_zoom[:, 1][::-1], color = 'navy', linewidth = 0.5, alpha = 0.8, label = r'Zoom-in (all)')
axins1.fill_between(SFH1_zoom[:, 0], SFH1_zoom[:, 2][::-1], SFH1_zoom[:, 3][::-1], color = 'deepskyblue', alpha = 0.2, lw = 0.)
axins1.plot(SFH1_zoom_sat[:, 0], SFH1_zoom_sat[:, 1][::-1], color = 'navy', dashes = (5, 2.4), linewidth = 0.5, alpha = 0.8, label = r'Satellites')
axins1.plot(SFH1_zoom_cent[:, 0], SFH1_zoom_cent[:, 1][::-1], color = 'navy', linestyle = '-.', linewidth = 0.5, alpha = 0.8, label = r'Centrals')
axins1.tick_params(which='major', labelsize = 12, width = 0.8, length = 3, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
axins1.set_xlabel(r'$z$', fontsize = 16, labelpad = 1)
axins1.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 16)
axins1.set_xticks(lbti.value)
axins1.set_xticklabels(z_lbti)
axins1.set_xlim([13.8, -0.5])
xmin, xmax = axins1.get_xlim()
ymin, ymax = axins1.get_ylim()
width = xmax - xmin
height = ymax - ymin
#axins1.text(xmax - width * 0.03, ymin + height * 0.08, r'Non-orphans only', 
#         horizontalalignment = 'right', verticalalignment = 'bottom', fontsize = 12, color = 'black', alpha = 0.8)


axins2 = ax2.inset_axes([0.64, 0.12, 0.32, 0.28])
axins2.plot(SFH2_zoom[:, 0], SFH2_zoom[:, 1][::-1], color = 'navy', linewidth = 0.5, alpha = 0.8, label = r'Zoom-in (all)')
axins2.fill_between(SFH2_zoom[:, 0], SFH2_zoom[:, 2][::-1], SFH2_zoom[:, 3][::-1], color = 'deepskyblue', alpha = 0.2, lw = 0.)
axins2.plot(SFH2_zoom_cent[:, 0], SFH2_zoom_cent[:, 1][::-1], color = 'navy', linestyle = '-.', linewidth = 0.5, alpha = 0.8, label = r'Centrals')
axins2.plot(SFH2_zoom_sat[:, 0], SFH2_zoom_sat[:, 1][::-1], color = 'navy', dashes = (5, 2.4), linewidth = 0.5, alpha = 0.8, label = r'Satellites')
axins2.tick_params(which='major', labelsize = 12, width = 0.8, length = 3, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
axins2.set_xlabel(r'$z$', fontsize = 16, labelpad = 1)
axins2.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 16)
axins2.set_xticks(lbti.value)
axins2.set_xticklabels(z_lbti)
axins2.set_xlim([13.8, -0.5])
xmin, xmax = axins2.get_xlim()
ymin, ymax = axins2.get_ylim()
width = xmax - xmin
height = ymax - ymin
#axins2.text(xmax - width * 0.03, ymin + height * 0.08, r'Non-orphans only', 
#         horizontalalignment = 'right', verticalalignment = 'bottom', fontsize = 12, color = 'black', alpha = 0.8)

#######################

xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^4, 10^5]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax2.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^5, 10^6]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax3.get_xlim()
ymin, ymax = ax3.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax3.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^6, 10^7]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax4.get_xlim()
ymin, ymax = ax4.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax4.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^7, 10^8]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax5.get_xlim()
ymin, ymax = ax5.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax5.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^8, 10^9]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax6.get_xlim()
ymin, ymax = ax6.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax6.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^9, 10^{10}]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax7.get_xlim()
ymin, ymax = ax7.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax7.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^{10}, 10^{11}]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

xmin, xmax = ax8.get_xlim()
ymin, ymax = ax8.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax8.text(xmin + width * 0.02, ymax - height * 0.02, r'$M_{\ast}/\mathrm{M_{\odot}}\in[10^{11}, 10^{12}]$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 18, color = 'black', alpha = 0.8)

ax1.legend(bbox_to_anchor=(0.16, 0.01, 0.2, 0.2), loc='lower left', ncol=1, fontsize = 14, frameon=False)
ax2.legend(bbox_to_anchor=(0.16, 0.01, 0.2, 0.2), loc='lower left', ncol=1, fontsize = 14, frameon=False)
ax3.legend(loc = 'lower right', fontsize = 16, frameon=False, borderpad = 0.8)
ax4.legend(loc = 'lower right', fontsize = 16, frameon=False, borderpad = 0.8)
ax5.legend(loc = 'lower right', fontsize = 16, frameon=False, borderpad = 0.8)
ax6.legend(loc = 'lower right', fontsize = 16, frameon=False, borderpad = 0.8)
ax7.legend(loc = 'lower right', fontsize = 16, frameon=False, borderpad = 0.8)
ax8.legend(loc = 'lower right', fontsize = 16, frameon=False, borderpad = 0.8)

fig.savefig('/sdf/home/y/ycwang/figures/full_SFH.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()
