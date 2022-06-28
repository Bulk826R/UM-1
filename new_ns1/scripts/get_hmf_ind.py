from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from multiprocessing import Pool
#fig=plt.figure()

#####################

h = 0.7
boxsize = 125/h #Mpc units
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'
baseDir = src+'/hmf'
fields = ['scale','id', 'upid', 'mvir', 'rvir', 'mpeak', 'vmax', 'vpeak']

logM_min = 6.5
logM_max = 12.5
Nbins = 25 #Also 141 bins if possible
bin_pivot = np.linspace(logM_min, logM_max, Nbins) 
h_bins = bin_pivot[1] - bin_pivot[0]
logM_bins = np.linspace(logM_min - h_bins/2, logM_max + h_bins/2, Nbins+1)
np.save(baseDir+'/zoom_hmf_mbins.npy', bin_pivot)

def get_hmf(halo_mass):
    hmf = []
    for i in range(len(logM_bins)-1):
        in_bin = np.where((np.log10(halo_mass) >= logM_bins[i]) & (np.log10(halo_mass) < logM_bins[i+1]))[0]
        N_inbin = len(in_bin)
        dn = N_inbin / boxsize**3 # Mpc^-3
        dlogM = logM_bins[i+1] - logM_bins[i]
        hmf.append(dn/dlogM)
    hmf = np.asarray(hmf)
    return hmf


zi_ext = np.loadtxt(baseDir+'/extents2.txt')
x_ext = zi_ext[:, 0]
y_ext = zi_ext[:, 1]
z_ext = zi_ext[:, 2]
Ext = x_ext * y_ext * z_ext # Lagrangian volume factor, * lbox**3 to get volume

Halo_no = np.loadtxt(wr+'/halo2_list.txt')
Halo_no = Halo_no.astype(np.int64)
Halos_zoom = readHlist(src+'/Halo{0:03d}/output/rockstar/hlists/hlist_1.00000.list'.format(Halo_no[0]), fields) 
Mpeak_zoom = Halos_zoom['mpeak'] / h
Mvir_zoom = Halos_zoom['mvir'] / h

HMF_zoom = get_hmf(Mpeak_zoom) / Ext[0]
np.save(baseDir+'/HMF_zoom_{0:03d}.npy'.format(Halo_no[0]), HMF_zoom)
rho = np.sum(Mpeak_zoom) / (boxsize**3 * Ext[0])
Rho = np.asarray([[rho, Halo_no[0]]])

for i in range(1, len(Halo_no)):
    print(i)
    if (Halo_no[i] == 8247) or (Halo_no[i] == 9749) or (Halo_no[i] == 9829):
        d = readHlist(src+'/Halo{0:04d}/output/rockstar/hlists/hlist_1.00000.list'.format(Halo_no[i]), fields) 
        Mpeak_zoom = d['mpeak'] / h
        Mvir_zoom = d['mvir'] / h

        HMF_zoom = get_hmf(Mpeak_zoom) / Ext[i]
        np.save(baseDir+'/HMF_zoom_{0:04d}.npy'.format(Halo_no[i]), HMF_zoom)
        rho = np.sum(Mpeak_zoom) / (boxsize**3 * Ext[i])
        Rho = np.concatenate((Rho, [[rho, Halo_no[i]]]), axis = 0)
    else:
        d = readHlist(src+'/Halo{0:03d}/output/rockstar/hlists/hlist_1.00000.list'.format(Halo_no[i]), fields)  
        Mpeak_zoom = d['mpeak'] / h
        Mvir_zoom = d['mvir'] / h

        HMF_zoom = get_hmf(Mpeak_zoom) / Ext[i]
        np.save(baseDir+'/HMF_zoom_{0:03d}.npy'.format(Halo_no[i]), HMF_zoom)
        rho = np.sum(Mpeak_zoom) / (boxsize**3 * Ext[i])
        Rho = np.concatenate((Rho, [[rho, Halo_no[i]]]), axis = 0)

    
#####################

np.savetxt(baseDir+'/rho_hlist.txt', Rho)


