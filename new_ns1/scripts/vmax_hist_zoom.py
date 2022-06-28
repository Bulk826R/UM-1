from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from multiprocessing import Pool
from astropy.cosmology import FlatLambdaCDM
#from astropy import units as u
cosmo = FlatLambdaCDM(H0=70.00, Om0=0.286)

h = 0.7
omega_m = 0.286
src = '/sdf/group/kipac/u/ycwang/MWmass_new' 
wr = '/sdf/group/kipac/g/cosmo/ki21/ycwang/join_halos/MWmass'

h = 0.7
omega_m = 0.286
Halo_no = np.loadtxt(wr+'/halo2_list.txt')
Halo_no = Halo_no.astype(np.int64)
sz_zoom = np.loadtxt(src+'/Halo416/output/rockstar/outputs/scales.txt')
scales = sz_zoom[:, 1]
z = 1/scales - 1
t = cosmo.lookback_time(z).value
vmax_th = 10.

def read_sfh(chunk_no):
    SFH3 = np.loadtxt(wr+'/vmax{}_np.txt'.format(chunk_no))
    x = t
    
    y = SFH3
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mstar3_med = bin_med
    Mstar3_low = bin_low
    Mstar3_high = bin_high
    
    res3 = np.vstack((x, Mstar3_med, Mstar3_low, Mstar3_high)).T
    with open(wr+'/Vmax{}_np.txt'.format(chunk_no), 'w+') as sfh3:
        np.savetxt(sfh3, res3)
        
        return

chunk = np.arange(1, 3, 1)

if __name__ == '__main__':
    p = Pool(8)
    p.map(read_sfh, chunk)