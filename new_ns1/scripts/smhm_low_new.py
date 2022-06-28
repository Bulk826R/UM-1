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
baseDir = '/sdf/group/kipac/u/ycwang/MWmass_new'
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'
######################################

fi_lmc = np.loadtxt(baseDir+'/hmf/TNG/fi_lmc.txt')
fi_x = fi_lmc[:, 0]
fi_y = fi_lmc[:, 1]

fi_1s_x = np.log10(fi_x[:7])
fi_2s_x= np.log10(fi_x[14:21])
fi_1s_yhigh = np.log10(fi_y[:7])
fi_1s_ylow = np.log10(fi_y[7:14])
fi_2s_yhigh = np.log10(fi_y[14:21])
fi_2s_ylow = np.log10(fi_y[21:28])

UM_z0 = np.loadtxt(baseDir+'/hmf/TNG/UM_z0.txt')
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

x_zoom = np.loadtxt(wr+'/smhm/z0/full_x.txt')
y_zoom =np.loadtxt(wr+'/smhm/z0/full_y.txt')
y_std_zoom = np.loadtxt(wr+'/smhm/z0/full_y_std.txt')
y_med_zoom = np.loadtxt(wr+'/smhm/z0/full_y_med.txt')
y_low_zoom = np.loadtxt(wr+'/smhm/z0/full_y_low.txt')
y_high_zoom = np.loadtxt(wr+'/smhm/z0/full_y_high.txt')

######################################

Beh_13 = np.loadtxt(baseDir+'/hmf/TNG/Beh_13.txt')
Beh_13_mvir = Beh_13[:, 0]
Beh_13_mstar = Beh_13[:, 1]

xA = np.log10(Beh_13_mvir[0])
xB = np.log10(Beh_13_mvir[1])
yA = np.log10(Beh_13_mstar[0])
yB = np.log10(Beh_13_mstar[1])
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

Beh_13_mvir = np.concatenate(([10**xl], Beh_13_mvir))
Beh_13_mstar = np.concatenate(([10**yl], Beh_13_mstar))

######################################

Mos_13 = np.loadtxt(baseDir+'/hmf/TNG/Mos_13.txt')
Mos_13_mvir = Mos_13[:, 0]
Mos_13_mstar = Mos_13[:, 1]

xA = Mos_13_mvir[0]
xB = Mos_13_mvir[1]
yA = Mos_13_mstar[0]
yB = Mos_13_mstar[1]
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

Mos_13_mvir = np.concatenate(([xl], Mos_13_mvir))
Mos_13_mstar = np.concatenate(([yl], Mos_13_mstar))

######################################

GK_14 = np.loadtxt(baseDir+'/hmf/TNG/GK_14.txt')
GK_14_mvir = GK_14[:, 0]
GK_14_mstar = GK_14[:, 1]

xA = np.log10(GK_14_mvir[0])
xB = np.log10(GK_14_mvir[1])
yA = np.log10(GK_14_mstar[0])
yB = np.log10(GK_14_mstar[1])
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

GK_14_mvir = np.concatenate(([10**xl], GK_14_mvir))
GK_14_mstar = np.concatenate(([10**yl], GK_14_mstar))

######################################

GK_17 = np.loadtxt(baseDir+'/hmf/TNG/GK_17.txt')
GK_17_mvir = GK_17[:, 0]
GK_17_mstar = GK_17[:, 1]

xA = GK_17_mvir[0]
xB = GK_17_mvir[1]
yA = GK_17_mstar[0]
yB = GK_17_mstar[1]
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

GK_17_mvir = np.concatenate(([xl], GK_17_mvir))
GK_17_mstar = np.concatenate(([yl], GK_17_mstar))

#print(np.log10(Mos_18_mvir))
print(GK_17_mvir)

######################################

RP_17 = np.loadtxt(baseDir+'/hmf/TNG/RP_17.txt')
RP_17_mvir = RP_17[:, 0]
RP_17_mstar = RP_17[:, 1]

xA = RP_17_mvir[0]
xB = RP_17_mvir[1]
yA = RP_17_mstar[0]
yB = RP_17_mstar[1]
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

RP_17_mvir = np.concatenate(([xl], RP_17_mvir))
RP_17_mstar = np.concatenate(([yl], RP_17_mstar))

######################################

Mos_18 = np.loadtxt(baseDir+'/hmf/TNG/Mos_18.txt')
Mos_18_lg_mvir = Mos_18[:, 0]
Mos_18_ms_to_mb = Mos_18[:, 1]
Mos_18_mvir = 10**Mos_18_lg_mvir
fb = 0.156
Mos_18_mstar = Mos_18_mvir * fb * Mos_18_ms_to_mb

xA = np.log10(Mos_18_mvir[0])
xB = np.log10(Mos_18_mvir[1])
yA = np.log10(Mos_18_mstar[0])
yB = np.log10(Mos_18_mstar[1])
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

Mos_18_mvir = np.concatenate(([10**xl], Mos_18_mvir))
Mos_18_mstar = np.concatenate(([10**yl], Mos_18_mstar))
#print(np.log10(Mos_18_mvir))
#print(RP_17_mvir)

######################################

Kra_18 = np.loadtxt(baseDir+'/hmf/TNG/Kra_18.txt')
Kra_18_mvir = Kra_18[:, 0]
Kra_18_mstar = Kra_18[:, 1]

xA = np.log10(Kra_18_mvir[0])
xB = np.log10(Kra_18_mvir[1])
yA = np.log10(Kra_18_mstar[0])
yB = np.log10(Kra_18_mstar[1])
m = (yA*xB - yB*xA) / (xB - xA)
k =(yB - yA) / (xB - xA)
yl = k * xl + m

Kra_18_mvir = np.concatenate(([10**xl], Kra_18_mvir))
Kra_18_mstar = np.concatenate(([10**yl], Kra_18_mstar))

print(Kra_18_mvir)
print(RP_17_mvir)

######################################

Jet_18 = np.loadtxt(baseDir+'/hmf/TNG/Jet_18.txt')
Jet_18_x = Jet_18[:, 0]
Jet_18_y = Jet_18[:, 1]
Jet_18_mvir = Jet_18_x[:11]
Jet_18_mstar_low = Jet_18_y[:11]
Jet_18_mstar_high = Jet_18_y[11:]

######################################

fig.set_size_inches(8, 6)
ax1 = fig.add_subplot(1, 1, 1)

cmap = matplotlib.cm.get_cmap('seismic_r')
rgba0 = cmap(0.9) 


#3 + 1
ax1.plot(Beh_13_mvir[1:], Beh_13_mstar[1:], 
         linewidth = 1.6, alpha = 0.6, color = 'crimson')
#4
ax1.plot(Beh_13_mvir[0:2], Beh_13_mstar[0:2], linestyle = '-.', 
         linewidth = 1.6, alpha = 0.6, color = 'crimson', label = r'Behroozi+2013')

#5
ax1.plot(10**Mos_13_mvir[2:], 10**Mos_13_mstar[2:], linewidth = 1.6, alpha = 0.8, color = 'maroon')
#6
ax1.plot(10**Mos_13_mvir[0:3], 10**Mos_13_mstar[0:3], dashes = (5, 3.2), 
         linewidth = 1.6, alpha = 0.6, color = 'maroon', label = r'Moster+2013')

#7
ax1.plot(GK_14_mvir[3:], GK_14_mstar[3:],
         linewidth = 1.6, alpha = 0.6, color = 'orange')
#8
ax1.plot(GK_14_mvir[0:4], GK_14_mstar[0:4], linestyle = '--', 
         linewidth = 1.6, alpha = 0.6, color = 'orange', label = r'GK+2014')

#9
ax1.plot(10**GK_17_mvir[5:], 10**GK_17_mstar[5:], linewidth = 1.6, alpha = 0.8, color = 'orangered')
#10
ax1.plot(10**GK_17_mvir[0:6], 10**GK_17_mstar[0:6], dashes = (1.2, 4.2, 1.2, 4.2),
         linewidth = 1.6, alpha = 0.6, color = 'orangered', label = r'GK+2017')

#11
ax1.plot(10**RP_17_mvir[2:], 10**RP_17_mstar[2:],
         linewidth = 1.6, alpha = 0.6, color = 'seagreen')
#12
ax1.plot(10**RP_17_mvir[0:3], 10**RP_17_mstar[0:3], dashes = (4.2, 3.2, 1.2, 3.2), 
         linewidth = 1.6, alpha = 0.6, color = 'seagreen', label = r'RP+2017')
#r'Rodrı́guez-Puebla+2017 ($z=0.1$)'
#13
ax1.plot(Mos_18_mvir[3:], Mos_18_mstar[3:], 
         linewidth = 1.6, alpha = 0.6, color = 'springgreen')
#14
ax1.plot(Mos_18_mvir[0:4], Mos_18_mstar[0:4], dashes = (0.6, 1.2), 
         linewidth = 1.6, alpha = 0.6, color = 'springgreen', label = r'Moster+2018')
'''
ax1.plot(np.log10(Kra_18_mvir[1:]), np.log10(Kra_18_mstar[1:]),
         linewidth = 1.6, alpha = 0.8, color = 'darkslategrey')
ax1.plot(np.log10(Kra_18_mvir[0:2]), np.log10(Kra_18_mstar[0:2]), dashes = (3.2, 0.5, 1.2, 0.5), 
         linewidth = 1.6, alpha = 0.8, color = 'darkslategrey', label = r'Kravtsov+2018 ($z\,\lesssim\,0.1$)')
'''
#16
ax1.plot(10**UM_x0[1:], 10**UM_y0[1:], color = 'black', alpha = 0.6, linewidth = 1.6)
#17+1
ax1.plot(10**UM_x0[0:2], 10**UM_y0[0:2], color = 'black', alpha = 0.6, linewidth = 1.6, 
         label = r'Behroozi+2019', linestyle = ':')

'''
ax1.fill_between([7., 8.5], [-2., -2.], [4., 4.], facecolor = 'none', hatch = 'x', 
                 edgecolor= 'orange', linewidth = 1., label = r'Applebaum+2020')  
'''

l3 = ax1.fill_between(Jet_18_mvir, Jet_18_mstar_low, Jet_18_mstar_high, 
                 facecolor='yellow', alpha=0.82, label = r'Jethwa+2018')   

#0
ax1.plot(10**x_zoom, 10**y_med_zoom, color = rgba0, linewidth = 2.4, zorder = 999)
         #label = r'Joined MW resims median (UM, $z=0$)')
#1
l1 = ax1.fill_between(10**x_zoom, 10**y_low_zoom, 10**y_high_zoom, facecolor='royalblue', alpha = 0.32, zorder = 999,
                      label = r'Zoom-in (this work)')   
#2
l2 = ax1.fill_between(10**fi_1s_x[:-1], 10**fi_1s_ylow[:-1], 10**fi_1s_yhigh[:-1], facecolor='dimgray', alpha=0.32, label = r'Nadler+2020 (MW)')    
ax1.fill_between(10**fi_2s_x[:-1], 10**fi_2s_ylow[:-1], 10**fi_2s_yhigh[:-1], facecolor='dimgray', alpha=0.16)

 


Set1 = [l1, l2, l3]
Lab1 = [r'Zoom-in (this work)', 
        r'Nadler+2020 (Milky-Way)',
        r'Jethwa+2018 ']


lines = plt.gca().get_lines()
l2 = np.arange(15)
legend1 = plt.legend([l for l in Set1], [a for a in Lab1], 
            loc = 'upper left', fontsize = 14, handlelength = 2.8, frameon = False, borderpad = 0.8)
plt.legend([lines[i] for i in l2],[lines[i].get_label() for i in l2], 
           loc = 'lower right', fontsize = 14, handlelength = 2.8, frameon = False, borderpad = 0.8)
plt.gca().add_artist(legend1)


#ax1.legend(loc = 'lower right', fontsize = 9.6, handlelength = 2.8)


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



fig.savefig('/sdf/home/y/ycwang/figures/smhm_low.png', dpi = 400, bbox_inches = 'tight')
plt.show()