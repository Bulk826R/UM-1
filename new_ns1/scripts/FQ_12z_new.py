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
from matplotlib.ticker import AutoMinorLocator ,LogLocator
#from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from astropy.stats import bootstrap
fig=plt.figure()

from astropy.cosmology import FlatLambdaCDM
#from astropy import units as u
cosmo = FlatLambdaCDM(H0=70.00, Om0=0.286)

h = 0.7
omega_m = 0.286
Mv_sun = 4.80 #AB Mag, source: http://mips.as.arizona.edu/~cnaw/sun.html

def trans(z, x, y, Az, Ax, Ay): #Transforms Cartesian vectors to spherical, used for anisotropy parameter calculation
    r = np.sqrt(x**2 + y**2 + z**2)
    Ar = x/r*Ax + y/r*Ay + z/r*Az #r component
    At = z/r*x/np.sqrt(x**2+y**2)*Ax +  z/r*y/np.sqrt(x**2+y**2)*Ay - np.sqrt(x**2+y**2)/r*Az #theta component
    Ap = -y/np.sqrt(x**2+y**2)*Ax + x/np.sqrt(x**2+y**2)*Ay #phi component
    print(x[0], y[0], z[0], Ax[0], Ay[0], Az[0])
    return Ar, At, Ap

def get_Mv(mstar_obs):
    MtoL = 2 #Due to change
    L_obs = mstar_obs / MtoL
    M_obs = Mv_sun - 2.5*np.log10(L_obs)
    return M_obs #Observed AB magnitude 

src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

####################################

baseDir = src+'/hmf'
halos = np.loadtxt(baseDir+'/Data_1024_z0.txt')

ID = halos[:, 0]
UpID = halos[:, 1]
Mpeak = halos[:, 4]
Vpeak = halos[:, 7]
Mstar = halos[:, 10]
SFR = halos[:, 11]
Mstar_obs = halos[:, 12]
SFR_obs = halos[:, 13]

ID_Vpeak = np.where((Mstar > 0.))[0]
ID_sat = np.where((Mstar > 0.) & (UpID != -1))[0]
ID_cent = np.where((Mstar > 0.) & (UpID == -1))[0]

print('1024')
print(np.min(np.log10(Mstar)), np.max(np.log10(Mstar)))
print(np.min(np.log10(Mstar[ID_Vpeak])), np.max(np.log10(Mstar[ID_Vpeak])))
print(np.min(np.log10(Mstar[ID_sat])), np.max(np.log10(Mstar[ID_sat])))
print(np.min(np.log10(Mstar[ID_cent])), np.max(np.log10(Mstar[ID_cent])))

######

#bins_x = np.linspace(8, 12, 17)
bins_x = np.linspace(7, 12, 21)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_Vpeak])
sfr_cent = SFR_obs[ID_Vpeak]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - z)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - z)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
FQ_1024 = np.asarray(FQ)
FQ_1024_std = np.asarray(FQ_std)


#bins_x = np.linspace(8, 12, 17)
bins_x = np.linspace(7, 12, 21)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_cent])
sfr_cent = SFR_obs[ID_cent]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - z)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - z)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
FQ_1024_cent = np.asarray(FQ)
FQ_1024_std_cent = np.asarray(FQ_std)

#bins_x = np.linspace(8, 11.5, 15)
bins_x = np.linspace(7, 11.5, 19)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_sat])
sfr_cent = SFR_obs[ID_sat]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - 0.1)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - 0.1)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
FQ_1024_sat = np.asarray(FQ)
FQ_1024_std_sat = np.asarray(FQ_std)

####################################

baseDir = src+'/hmf'
halos = np.loadtxt(baseDir+'/Data_2048_z0.txt')

ID = halos[:, 0]
UpID = halos[:, 1]
Mpeak = halos[:, 4]
Vpeak = halos[:, 7]
Mstar = halos[:, 10]
SFR = halos[:, 11]
Mstar_obs = halos[:, 12]
SFR_obs = halos[:, 13]

ID_Vpeak = np.where((Mstar > 0.))[0]
ID_sat = np.where((Mstar > 0.) & (UpID != -1))[0]
ID_cent = np.where((Mstar > 0.) & (UpID == -1))[0]
'''
ID_Vpeak = np.where((Vpeak >= 35.) & (Mstar > 0.))[0]
ID_sat = np.where((Vpeak >= 35.) & (Mstar > 0.) & (UpID != -1))[0]
ID_cent = np.where((Vpeak >= 35.) & (Mstar > 0.) & (UpID == -1))[0]
'''
print('2048')
print(np.min(np.log10(Mstar)), np.max(np.log10(Mstar)))
print(np.min(np.log10(Mstar[ID_Vpeak])), np.max(np.log10(Mstar[ID_Vpeak])))
print(np.min(np.log10(Mstar[ID_sat])), np.max(np.log10(Mstar[ID_sat])))
print(np.min(np.log10(Mstar[ID_cent])), np.max(np.log10(Mstar[ID_cent])))

######

#bins_x = np.linspace(6, 12, 25)
bins_x = np.linspace(5, 12, 29)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_Vpeak])
sfr_cent = SFR_obs[ID_Vpeak]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - z)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - z)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
FQ_2048 = np.asarray(FQ)
FQ_2048_std = np.asarray(FQ_std)

#bins_x = np.linspace(6, 12, 25)
bins_x = np.linspace(5, 12, 29)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_cent])
sfr_cent = SFR_obs[ID_cent]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - 0.1)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - 0.1)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
FQ_2048_cent = np.asarray(FQ)
FQ_2048_std_cent = np.asarray(FQ_std)


#bins_x = np.linspace(6, 11.5, 23)
bins_x = np.linspace(5, 11.5, 27)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_sat])
sfr_cent = SFR_obs[ID_sat]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - 0.1)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - 0.1)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
FQ_2048_sat = np.asarray(FQ)
FQ_2048_std_sat = np.asarray(FQ_std)

###############################

halos = np.loadtxt(wr+'/Data_join_MWmass.txt')
ID = halos[:, 0]
UpID = halos[:, 2]
Mpeak = halos[:, 13]
Vpeak = halos[:, 14]
Mstar = halos[:, 20]
SFR = halos[:, 22]
Mstar_obs = halos[:, 23]
SFR_obs = halos[:, 24]

ID_Vpeak = np.where((Mstar > 0.))[0]
ID_sat = np.where((Mstar > 0.) & (UpID != -1))[0]
ID_cent = np.where((Mstar > 0.) & (UpID == -1))[0]

print('Zoom')
print(np.min(np.log10(Mstar)), np.max(np.log10(Mstar)))
print(np.min(np.log10(Mstar[ID_Vpeak])), np.max(np.log10(Mstar[ID_Vpeak])))
print(np.min(np.log10(Mstar[ID_sat])), np.max(np.log10(Mstar[ID_sat])))
print(np.min(np.log10(Mstar[ID_cent])), np.max(np.log10(Mstar[ID_cent])))

zp = 1.96 # probit of 95% interval for mu=0 sigma = 1 gaussian

######

#bins_x = np.linspace(4, 11, 29)
bins_x = np.linspace(2, 10.5, 35)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_Vpeak])
sfr_cent = SFR_obs[ID_Vpeak]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
Bi_mean = []
Bi_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - 0.1)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - 0.1)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
    pb = fq
    n = len(sfr_bin)
    bi_mean = (pb + zp**2/(2*n)) / (1+zp**2/n)
    bi_std = zp/(1+zp**2/n) * np.sqrt(pb/n*(1-pb) + zp**2/(4*n**2))
    Bi_mean.append(bi_mean)
    Bi_std.append(bi_std)
    
FQ_zoom = np.asarray(FQ)
FQ_zoom_std = np.asarray(FQ_std)
Bi_mean_all = np.asarray(Bi_mean)
Bi_std_all = np.asarray(Bi_std)

#bins_x = np.linspace(4, 11, 29)
bins_x = np.linspace(2, 10.5, 35)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_cent])
sfr_cent = SFR_obs[ID_cent]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
Bi_mean = []
Bi_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - 0.1)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
        
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - 0.1)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
    pb = fq
    n = len(sfr_bin)
    bi_mean = (pb + zp**2/(2*n)) / (1+zp**2/n)
    bi_std = zp/(1+zp**2/n) * np.sqrt(pb/n*(1-pb) + zp**2/(4*n**2))
    Bi_mean.append(bi_mean)
    Bi_std.append(bi_std)
    
FQ_zoom_cent = np.asarray(FQ)
FQ_zoom_std_cent = np.asarray(FQ_std)
Bi_mean_cent = np.asarray(Bi_mean)
Bi_std_cent = np.asarray(Bi_std)

#bins_x = np.linspace(4, 10, 25)
bins_x = np.linspace(2, 10, 33)
dh = (bins_x[1] - bins_x[0])/2
mstar_cent = np.log10(Mstar_obs[ID_sat])
sfr_cent = SFR_obs[ID_sat]
bins_h = (bins_x[1] - bins_x[0])/2
bins = np.linspace(bins_x[0] - bins_h, bins_x[-1] + bins_h, len(bins_x)+1)
bin_tot = np.digitize(mstar_cent, bins)
bin_means_tot = np.asarray([mstar_cent[bin_tot == i].mean() for i in range(1, len(bins))])

FQ = []
FQ_std = []
Bi_mean = []
Bi_std = []
for i in range(1, len(bins)):
    sfr_bin = sfr_cent[bin_tot == i]
    mstar_bin = (10**mstar_cent)[bin_tot == i]
    print(len(mstar_bin), i)
    bootarr = np.arange(len(sfr_bin))
    bootresult = bootstrap(bootarr, 500)
    FQ_res = []
    for j in range(len(bootresult)):
        sfr_res = sfr_bin[bootresult[j].astype(int)]
        mstar_res = mstar_bin[bootresult[j].astype(int)]
        ssfr_res = sfr_res / mstar_res
        #log_ssfr_th = -10.49 -0.35*np.log10(mstar_res/1e10) + 1.07 * (0 - 0.1)
        #idq = np.where(np.log10(ssfr_res) < log_ssfr_th)[0]
        idq = np.where(ssfr_res < 1e-11)[0]
        fq_res = len(idq)/len(sfr_res)
        FQ_res.append(fq_res)
    
    FQ_res = np.asarray(FQ_res)
    fq_std = np.std(FQ_res)
    FQ_std.append(fq_std)
    ssfr_bin = sfr_bin / mstar_bin
    
    #log_ssfr_th = -10.49 -0.35*np.log10(mstar_bin/1e10) + 1.07 * (0 - 0.1)
    #idq = np.where(np.log10(ssfr_bin) < log_ssfr_th)[0]
    idq = np.where(ssfr_bin < 1e-11)[0]
    fq = len(idq) / len(sfr_bin)
    FQ.append(fq)
    
    pb = fq
    n = len(sfr_bin)
    bi_mean = (pb + zp**2/(2*n)) / (1+zp**2/n)
    bi_std = zp/(1+zp**2/n) * np.sqrt(pb/n*(1-pb) + zp**2/(4*n**2))
    Bi_mean.append(bi_mean)
    Bi_std.append(bi_std)
    
FQ_zoom_sat = np.asarray(FQ)
FQ_zoom_std_sat = np.asarray(FQ_std)
Bi_mean_sat = np.asarray(Bi_mean)
Bi_std_sat = np.asarray(Bi_std)

###############################

fig.set_size_inches(8, 16)
plt.subplots_adjust(wspace = 0.2, hspace = 0.06)

ax1 = fig.add_subplot(3, 1, 1)
bins_x = np.linspace(2, 10.5, 35)
ax1.plot(10**bins_x, FQ_zoom, lw = 1.6, alpha = 0.8, color = 'navy', label = r'Zoom-in (high-res region)')
ax1.fill_between(10**bins_x, FQ_zoom-2*FQ_zoom_std, FQ_zoom+2*FQ_zoom_std, color = 'deepskyblue', 
                 alpha = 0.3, label = r'Zoom-in ($2\sigma$)')
ax1.errorbar(10**bins_x, Bi_mean_all, yerr = Bi_std_all, fmt = 's', color = 'black', 
             alpha = 0.5, markersize = 4, capsize = 4.2, label = r'Zoom-in (Wilson score)')
bins_x = np.linspace(5, 12, 29)
ax1.plot(10**bins_x, FQ_2048, lw = 1.2, alpha = 0.8, color = 'royalblue', 
         linestyle = '-.', label = r'c125-2048')
#ax1.fill_between(bins_x, FQ_2048-FQ_2048_std, FQ_2048+FQ_2048_std, color = 'black', alpha = 0.3)
bins_x = np.linspace(7, 12, 21)
ax1.plot(10**bins_x, FQ_1024, lw = 1.2, alpha = 0.8, color = 'deepskyblue', dashes = (5, 2.4),
         label = r'c125-1024')
#ax1.fill_between(bins_x, FQ_1024-FQ_1024_std, FQ_1024+FQ_1024_std, color = 'tomato', alpha = 0.3)

ax1.set_xlim([10**3.6, 10**12.2])
ax1.set_ylim([0, 1])
ax1.legend(loc = 'center left', fontsize = 14, frameon = False, borderpad = 0.8)
#ax1.set_xlabel(r'$M_{\ast}\,$[$\mathrm{M_{\odot}}$]', fontsize = 20, labelpad = 8)
#plt.setp(ax1.get_xticklabels(), visible = False)
ax1.set_ylabel(r'$f_{\mathrm{Quench}}$', fontsize = 20, labelpad = 8)
xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin * 2, ymax - height * 0.04, 
         r'All galaxies', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 16, color = 'black', alpha = 0.8)

ax1.set_xscale('log')
ax1.set_xticks(np.logspace(4, 12, 9))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax1.xaxis.set_minor_locator(loc1)
ax1.tick_params(which='major', labelsize = 16, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax1.tick_params(which='minor', labelsize = 16, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)
ax1.set_xticklabels([])

ax2 = fig.add_subplot(3, 1, 2)
bins_x = np.linspace(2, 10.5, 35)
ax2.plot(10**bins_x, FQ_zoom_cent, lw = 1.6, alpha = 0.8, color = 'navy')
ax2.fill_between(10**bins_x, FQ_zoom_cent-2*FQ_zoom_std_cent, FQ_zoom_cent+2*FQ_zoom_std_cent, 
                 color = 'deepskyblue', alpha = 0.3)
ax2.errorbar(10**bins_x, Bi_mean_cent, yerr = Bi_std_cent, fmt = 's', color = 'black', 
             alpha = 0.5, markersize = 4, capsize = 4.2)
bins_x = np.linspace(5, 12, 29)
ax2.plot(10**bins_x, FQ_2048_cent, lw = 1.2, alpha = 0.8, color = 'royalblue', 
         linestyle = '-.')
#ax2.fill_between(bins_x, FQ_2048_cent-FQ_2048_std_cent, FQ_2048_cent+FQ_2048_std_cent, color = 'black', alpha = 0.3)
bins_x = np.linspace(7, 12, 21)
ax2.plot(10**bins_x, FQ_1024_cent, lw = 1.2, alpha = 0.8, color = 'deepskyblue', dashes = (5, 2.4))
#ax2.fill_between(bins_x, FQ_1024_cent-FQ_1024_std_cent, FQ_1024_cent+FQ_1024_std_cent, color = 'tomato', alpha = 0.3)

ax2.set_xlim([10**3.6, 10**12.2])
ax2.set_ylim([0, 1])
#ax2.set_xlabel(r'$M_{\ast}\,$[$\mathrm{M_{\odot}}$]', fontsize = 20, labelpad = 8)
ax2.set_ylabel(r'$f_{\mathrm{Quench}}$', fontsize = 20, labelpad = 8)
xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax2.text(xmin * 2, ymax - height * 0.04, 
         r'Centrals', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 16, color = 'black', alpha = 0.8)

Gmin = 21232650.905858614
Gmax = 170634761804.7826
ax2.plot([Gmin, Gmax], [0.8, 0.8], color = 'dimgray', alpha = 0.8, lw = 1.2)
ax2.plot([Gmin, Gmin], [0.8, 0.84], color = 'dimgray', alpha = 0.8, lw = 1.2)
ax2.plot([Gmax, Gmax], [0.8, 0.84], color = 'dimgray', alpha = 0.8, lw = 1.2)

'''
from matplotlib.patches import Rectangle
ax2.add_patch(Rectangle((Gmin, 0.8), 
                        Gmax - Gmin, 0.1, 
                        color = 'dimgray',  
                        ec ='black', 
                        lw = 1.2, alpha = 0.6,
                        label = r'Geha+2012')) 
'''
#ax2.axvline(x = 7.3100820633059795, lw = 1.6, color = 'dimgray', linestyle = ':')
#ax2.text(7.5, 0.92, r'$\Longrightarrow$', fontsize = 20)
#ax2.legend(loc = 'upper left', fontsize = 14, frameon = False, borderpad = 0.8)

ax2.set_xscale('log')
ax2.set_xticks(np.logspace(4, 12, 9))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax2.xaxis.set_minor_locator(loc1)
ax2.tick_params(which='major', labelsize = 16, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax2.tick_params(which='minor', labelsize = 16, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)
ax2.set_xticklabels([])

ax3 = fig.add_subplot(3, 1, 3)
bins_x = np.linspace(2, 10, 33)
ax3.plot(10**bins_x, FQ_zoom_sat, lw = 1.6, alpha = 0.8, color = 'navy')
ax3.fill_between(10**bins_x, FQ_zoom_sat-2*FQ_zoom_std_sat, FQ_zoom_sat+2*FQ_zoom_std_sat, 
                 color = 'deepskyblue', alpha = 0.3)
ax3.errorbar(10**bins_x, Bi_mean_sat, yerr = Bi_std_sat, fmt = 's', color = 'black', 
             alpha = 0.5, markersize = 4, capsize = 4.2)
bins_x = np.linspace(5, 11.5, 27)
ax3.plot(10**bins_x, FQ_2048_sat, lw = 1.2, alpha = 0.8, color = 'royalblue', 
         linestyle = '-.')
#ax3.fill_between(bins_x, FQ_2048_sat-FQ_2048_std_sat, FQ_2048_sat+FQ_2048_std_sat, color = 'black', alpha = 0.3)
bins_x = np.linspace(7, 11.5, 19)
ax3.plot(10**bins_x, FQ_1024_sat, lw = 1.2, alpha = 0.8, color = 'deepskyblue', dashes = (5, 2.4))
#ax3.fill_between(bins_x, FQ_1024_sat-FQ_1024_std_sat, FQ_1024_sat+FQ_1024_std_sat, color = 'tomato', alpha = 0.3)

#($V_{\mathrm{Mpeak}}\geqslant 35\mathrm{km\,s^{-1}}$)
#($V_{\mathrm{Mpeak}}\geqslant 70\mathrm{km\,s^{-1}}$)
#($V_{\mathrm{Mpeak}}\geqslant 10\mathrm{km\,s^{-1}}$)

ax3.set_xlim([10**3.6, 10**12.2])
ax3.set_ylim([0, 1])
ax3.set_xlabel(r'$M_{\ast}\,$[$\mathrm{M_{\odot}}$]', fontsize = 20, labelpad = 8)
ax3.set_ylabel(r'$f_{\mathrm{Quench}}$', fontsize = 20, labelpad = 8)
xmin, xmax = ax3.get_xlim()
ymin, ymax = ax3.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax3.text(xmin * 2, ymax - height * 0.04, 
         r'Satellites', 
         horizontalalignment = 'left', verticalalignment = 'top', 
         fontsize = 16, color = 'black', alpha = 0.8)

#ax3.fill_between([6.7913247362250875, 9.118405627198126], [-0.1, -0.1], [1.1, 1.1], facecolor = 'none', hatch = '/', 
#                 edgecolor= 'orangered', linewidth = 1.2)  

W = np.loadtxt(src+'/hmf/TNG/Weisz_14.txt')
Mstar = W[:, -1] * 1e6
Wmax = np.max(Mstar)
Wmin = np.min(Mstar)

Mmin = 10**6.7913247362250875
Mmax = 10**9.118405627198126
ax3.plot([Mmin, Mmax], [0.76, 0.76], color = 'orangered', alpha = 0.8, lw = 1.2)
ax3.plot([Mmin, Mmin], [0.76, 0.8], color = 'orangered', alpha = 0.8, lw = 1.2)
ax3.plot([Mmax, Mmax], [0.76, 0.8], color = 'orangered', alpha = 0.8, lw = 1.2)

ax3.plot([Wmin, Wmax], [0.6, 0.6], color = 'crimson', alpha = 0.8, lw = 1.2)
ax3.plot([Wmin, Wmin], [0.6, 0.64], color = 'crimson', alpha = 0.8, lw = 1.2)
ax3.plot([Wmax, Wmax], [0.6, 0.64], color = 'crimson', alpha = 0.8, lw = 1.2)


'''
ax3.add_patch(Rectangle((Wmin, 0.65), 
                        Wmax - Wmin, 0.1, 
                        color = 'crimson',  
                        ec ='black', 
                        lw = 1.2, alpha = 0.6,
                        label = r'Weisz+2014')) 
ax3.add_patch(Rectangle((, 0.8), 
                        10**9.118405627198126 - 10**6.7913247362250875, 0.1, 
                        color = 'orangered',  
                        ec ='black', 
                        lw = 1.2, alpha = 0.6,
                        label = r'Mao+2020'))
'''
#ax3.legend(loc = 'upper left', fontsize = 14, frameon = False, borderpad = 0.8)

'''
6.7913247362250875, 0.055182041177606056
9.118405627198126, 0.05239857436781148
7.3100820633059795, 0.008512361285705428
'''

ax3.set_xscale('log')
ax3.set_xticks(np.logspace(4, 12, 9))
minorLocator = AutoMinorLocator()
loc1 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax3.xaxis.set_minor_locator(loc1)
ax3.tick_params(which='major', labelsize = 16, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax3.tick_params(which='minor', labelsize = 16, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


fig.savefig('/sdf/home/y/ycwang/figures/FQ_12z.png', dpi = 400, bbox_inches = 'tight')
plt.show()