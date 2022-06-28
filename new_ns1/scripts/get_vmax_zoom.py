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
    SFH3 = np.loadtxt(wr+'/sfh{}.txt'.format(chunk_no))
    x = t
    
    ######
    
    ID = SFH3[:, 0].astype(np.int64)
    halo_no = SFH3[:, -237].astype(np.int64)
    id_non_orp = np.where(ID <= 1e15)[0]
    ids = np.vstack((ID[id_non_orp], halo_no[id_non_orp])).T
    np.savetxt(wr+'/ID{}_np.txt'.format(chunk_no), ids)
    print('Len all, non_orp = ', len(ID), len(id_non_orp))
    
    UpID = SFH3[:, 1].astype(np.int64)
    id_cent = np.where(UpID == -1)[0]
    id_sat = np.where(UpID != -1)[0]
    id_non_orp_cent = np.where((ID <= 1e15) & (UpID == -1))[0]
    id_non_orp_sat = np.where((ID <= 1e15) & (UpID != -1))[0]
    ids = np.vstack((ID[id_non_orp_cent], halo_no[id_non_orp_cent])).T
    np.savetxt(wr+'/ID{}_np_cent.txt'.format(chunk_no), ids)
    ids = np.vstack((ID[id_non_orp_sat], halo_no[id_non_orp_sat])).T
    np.savetxt(wr+'/ID{}_np_sat.txt'.format(chunk_no), ids)
    print('Sat frac {} = '.format(chunk_no), len(id_sat)/len(UpID))
    
    y = SFH3[:, 3:3+len(scales)]
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mstar3_med = bin_med
    Mstar3_low = bin_low
    Mstar3_high = bin_high
    
    M = SFH3[:, 239:239+len(scales)]
    M0 = M[:, -1]
    y = np.transpose(M.T / M0)
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mvir3_med = bin_med
    Mvir3_low = bin_low
    Mvir3_high = bin_high
    
    y = SFH3[:, 475:475+len(scales)]
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    V3_med = bin_med
    V3_low = bin_low
    V3_high = bin_high
    
    y = SFH3[id_non_orp, 3:3+len(scales)]
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mstar3_med_np = bin_med
    Mstar3_low_np = bin_low
    Mstar3_high_np = bin_high
    
    M = SFH3[:, 949:949+len(scales)]
    M0 = M[:, -1]
    y = np.transpose(M.T / M0)
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Msfull3_med = bin_med
    Msfull3_low = bin_low
    Msfull3_high = bin_high
    
    for i in range(len(x)):
        if i==0 and V3_med[i] > vmax_th:
            snap = 0
            break
        elif (V3_med[i] < vmax_th) and (V3_med[i+1] < vmax_th):
            continue
        elif (V3_med[i] < vmax_th) and (V3_med[i+1] >= vmax_th):
            snap = i+1
            break
        
    print('Res snap {} = '.format(chunk_no), snap, scales[snap])
    res3 = np.vstack((x, Mstar3_med, Mstar3_low, Mstar3_high, 
                      Mvir3_med, Mvir3_low, Mvir3_high, 
                      V3_med, V3_low, V3_high, 
                      Mstar3_med_np, Mstar3_low_np, Mstar3_high_np,
                      Msfull3_med, Msfull3_low, Msfull3_high)).T
    
    with open(wr+'/Vmax{}_all.txt'.format(chunk_no), 'w+') as sfh3:
        np.savetxt(sfh3, res3)
        
    y = SFH3[id_cent, 3:3+len(scales)]
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mstar3_med = bin_med
    Mstar3_low = bin_low
    Mstar3_high = bin_high
    
    M = SFH3[id_cent, 239:239+len(scales)]
    M0 = M[:, -1]
    y = np.transpose(M.T / M0)
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mvir3_med = bin_med
    Mvir3_low = bin_low
    Mvir3_high = bin_high
    
    y = SFH3[id_cent, 475:475+len(scales)]
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    V3_med = bin_med
    V3_low = bin_low
    V3_high = bin_high
        
    y = SFH3[id_non_orp_cent, 3:3+len(scales)]
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Mstar3_med_np = bin_med
    Mstar3_low_np = bin_low
    Mstar3_high_np = bin_high
    
    M = SFH3[id_cent, 949:949+len(scales)]
    M0 = M[:, -1]
    y = np.transpose(M.T / M0)
    bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
    bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
    bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
    Msfull3_med = bin_med
    Msfull3_low = bin_low
    Msfull3_high = bin_high
    
    res3 = np.vstack((x, Mstar3_med, Mstar3_low, Mstar3_high, 
                      Mvir3_med, Mvir3_low, Mvir3_high, 
                      V3_med, V3_low, V3_high, 
                      Mstar3_med_np, Mstar3_low_np, Mstar3_high_np,
                      Msfull3_med, Msfull3_low, Msfull3_high)).T
    
    with open(wr+'/Vmax{}_cent.txt'.format(chunk_no), 'w+') as sfh3:
        np.savetxt(sfh3, res3)    
    
    if len(id_sat) > 0:
        y = SFH3[id_sat, 3:3+len(scales)]
        bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
        bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
        bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
        Mstar3_med = bin_med
        Mstar3_low = bin_low
        Mstar3_high = bin_high
        
        M = SFH3[id_sat, 239:239+len(scales)]
        M0 = M[:, -1]
        y = np.transpose(M.T / M0)
        bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
        bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
        bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
        Mvir3_med = bin_med
        Mvir3_low = bin_low
        Mvir3_high = bin_high
        
        y = SFH3[id_sat, 475:475+len(scales)]
        bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
        bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
        bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
        V3_med = bin_med
        V3_low = bin_low
        V3_high = bin_high
        
        y = SFH3[id_non_orp_sat, 3:3+len(scales)]
        bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
        bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
        bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
        Mstar3_med_np = bin_med
        Mstar3_low_np = bin_low
        Mstar3_high_np = bin_high
        
        M = SFH3[id_sat, 949:949+len(scales)]
        M0 = M[:, -1]
        y = np.transpose(M.T / M0)
        bin_med = np.asarray([np.percentile(y[:, i], 50) for i in range(len(x))])
        bin_low = np.asarray([np.percentile(y[:, i], 16) for i in range(len(x))])
        bin_high = np.asarray([np.percentile(y[:, i], 84) for i in range(len(x))])
        Msfull3_med = bin_med
        Msfull3_low = bin_low
        Msfull3_high = bin_high
        
        res3 = np.vstack((x, Mstar3_med, Mstar3_low, Mstar3_high, 
                          Mvir3_med, Mvir3_low, Mvir3_high, 
                          V3_med, V3_low, V3_high, 
                          Mstar3_med_np, Mstar3_low_np, Mstar3_high_np,
                          Msfull3_med, Msfull3_low, Msfull3_high)).T
        
        with open(wr+'/Vmax{}_sat.txt'.format(chunk_no), 'w+') as sfh3:
            np.savetxt(sfh3, res3)   
        
    return

chunk = np.arange(1, 8, 1)

if __name__ == '__main__':
    p = Pool(8)
    p.map(read_sfh, chunk)