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

sc_med = (scales[1:] + scales[:-1])/2
a1 = np.concatenate(([1/1100], sc_med))
a2 = np.concatenate((sc_med, [1.]))
t1 = cosmo.lookback_time(1/a1 - 1).value * 1e9 # yrs
t2 = cosmo.lookback_time(1/a2 - 1).value * 1e9 # yrs
dt = t1 - t2

def f_loss(x):
    return 0.05 * np.log(1 + x/(1.4 * 1e6))

tf = dt * (1 - f_loss(t1))
print('tf = ', tf)

with open(wr+'/sfh1.txt', 'w+') as sfh1:
    print('# This is SFH1 for zoom.', end = '\n', file = sfh1)

with open(wr+'/sfh2.txt', 'w+') as sfh2:
    print('# This is SFH2 for zoom.', end = '\n', file = sfh2)

with open(wr+'/sfh3.txt', 'w+') as sfh3:
    print('# This is SFH3 for zoom.', end = '\n', file = sfh3)

with open(wr+'/sfh4.txt', 'w+') as sfh4:
    print('# This is SFH4 for zoom.', end = '\n', file = sfh4)

with open(wr+'/sfh5.txt', 'w+') as sfh5:
    print('# This is SFH5 for zoom.', end = '\n', file = sfh5)

with open(wr+'/sfh6.txt', 'w+') as sfh6:
    print('# This is SFH6 for zoom.', end = '\n', file = sfh6)

with open(wr+'/sfh7.txt', 'w+') as sfh7:
    print('# This is SFH7 for zoom.', end = '\n', file = sfh7)


def get_ma(halo_no):
    
    if (halo_no == 8247) or (halo_no == 9749) or (halo_no == 9829):
        with open(src+'/Halo{0:04d}'.format(halo_no)+'/output/rockstar/groupcat/sfh_catalog_1.000000.0.txt', 'r') as f:
            sfh_test = np.loadtxt(f)
            ID = sfh_test[:, 0]
            UpID = sfh_test[:, 1]
            VMpeak = sfh_test[:, 10]
            Mstar_obs = sfh_test[:, 19]
            SFH = sfh_test[:, 24:24+len(scales)] #Use this
            Mstar_mainH = sfh_test[:, 496:496+len(scales)]
            M_mainH = sfh_test[:, 968:968+len(scales)]
            Vp_mainH = sfh_test[:, 1440:1440+len(scales)]
            f.close()    
            
    else:
        with open(wr+'/Halo{0:03d}'.format(halo_no)+'/output/rockstar/groupcat/sfh_catalog_1.000000.0.txt', 'r') as f:
            sfh_test = np.loadtxt(f)
            ID = sfh_test[:, 0]
            UpID = sfh_test[:, 1]
            VMpeak = sfh_test[:, 10]
            Mstar_obs = sfh_test[:, 19]
            SFH = sfh_test[:, 24:24+len(scales)] #Use this
            Mstar_mainH = sfh_test[:, 496:496+len(scales)]
            M_mainH = sfh_test[:, 968:968+len(scales)]
            Vp_mainH = sfh_test[:, 1440:1440+len(scales)]
            f.close() 
    
    M0 = Mstar_obs
    M_10 = 0.1 * M0
    M_50 = 0.5 * M0
    M_90 = 0.9 * M0
    
    id1 = np.where((M0 >= 1e4) & (M0 < 1e5))[0]
    id2 = np.where((M0 >= 1e5) & (M0 < 1e6))[0]
    id3 = np.where((M0 >= 1e6) & (M0 < 1e7))[0]
    id4 = np.where((M0 >= 1e7) & (M0 < 1e8))[0]
    id5 = np.where((M0 >= 1e8) & (M0 < 1e9))[0]
    id6 = np.where((M0 >= 1e9) & (M0 < 1e10))[0]
    id7 = np.where((M0 >= 1e10) & (M0 < 1e11))[0]
    
    ID1 = ID[id1]
    ID2 = ID[id2]
    ID3 = ID[id3]
    ID4 = ID[id4]
    ID5 = ID[id5]
    ID6 = ID[id6]
    ID7 = ID[id7]
    
    sfr1 = SFH[id1]
    sfr2 = SFH[id2]
    sfr3 = SFH[id3]
    sfr4 = SFH[id4]
    sfr5 = SFH[id5]
    sfr6 = SFH[id6]
    sfr7 = SFH[id7]
    
    dM1 = sfr1 * dt
    dM2 = sfr2 * dt
    dM3 = sfr3 * dt
    dM4 = sfr4 * dt
    dM5 = sfr5 * dt
    dM6 = sfr6 * dt
    dM7 = sfr7 * dt

    Ms_full1 = np.cumsum(dM1, axis = 1)
    Ms_full2 = np.cumsum(dM2, axis = 1)
    Ms_full3 = np.cumsum(dM3, axis = 1)
    Ms_full4 = np.cumsum(dM4, axis = 1)
    Ms_full5 = np.cumsum(dM5, axis = 1)
    Ms_full6 = np.cumsum(dM6, axis = 1)
    Ms_full7 = np.cumsum(dM7, axis = 1)
    
    upid1 = UpID[id1]
    upid2 = UpID[id2]
    upid3 = UpID[id3]
    upid4 = UpID[id4]
    upid5 = UpID[id5]
    upid6 = UpID[id6]
    upid7 = UpID[id7]

    vp1 = VMpeak[id1]
    vp2 = VMpeak[id2]    
    vp3 = VMpeak[id3]
    vp4 = VMpeak[id4]
    vp5 = VMpeak[id5]
    vp6 = VMpeak[id6]
    vp7 = VMpeak[id7]
    
    M01 = M0[id1]
    M02 = M0[id2]
    M03 = M0[id3]
    M04 = M0[id4]
    M05 = M0[id5]
    M06 = M0[id6]
    M07 = M0[id7]
    
    mm1 = Mstar_mainH[id1].T
    mm2 = Mstar_mainH[id2].T
    mm3 = Mstar_mainH[id3].T
    mm4 = Mstar_mainH[id4].T
    mm5 = Mstar_mainH[id5].T
    mm6 = Mstar_mainH[id6].T
    mm7 = Mstar_mainH[id7].T
    
    mm1 /= mm1[-1]
    mm2 /= mm2[-1]
    mm3 /= mm3[-1]
    mm4 /= mm4[-1]
    mm5 /= mm5[-1]
    mm6 /= mm6[-1]
    mm7 /= mm7[-1]
    
    Ms1 = Mstar_mainH[id1]
    Ms2 = Mstar_mainH[id2]
    Ms3 = Mstar_mainH[id3]
    Ms4 = Mstar_mainH[id4]
    Ms5 = Mstar_mainH[id5]
    Ms6 = Mstar_mainH[id6]
    Ms7 = Mstar_mainH[id7]
    
    Mh1 = M_mainH[id1]
    Mh2 = M_mainH[id2]
    Mh3 = M_mainH[id3]
    Mh4 = M_mainH[id4]
    Mh5 = M_mainH[id5]
    Mh6 = M_mainH[id6]
    Mh7 = M_mainH[id7]
    
    Vp1 = Vp_mainH[id1]
    Vp2 = Vp_mainH[id2]
    Vp3 = Vp_mainH[id3]
    Vp4 = Vp_mainH[id4]
    Vp5 = Vp_mainH[id5]
    Vp6 = Vp_mainH[id6]
    Vp7 = Vp_mainH[id7]

    H1 = halo_no * np.ones_like(id1)
    H2 = halo_no * np.ones_like(id2)
    H3 = halo_no * np.ones_like(id3)
    H4 = halo_no * np.ones_like(id4)
    H5 = halo_no * np.ones_like(id5)
    H6 = halo_no * np.ones_like(id6)
    H7 = halo_no * np.ones_like(id7)

    SFH1 = np.transpose(np.concatenate(([ID1], [upid1], [vp1], mm1, Mh1.T, Vp1.T, Ms1.T, [M01], [H1], Ms_full1.T), axis = 0))
    SFH2 = np.transpose(np.concatenate(([ID2], [upid2], [vp2], mm2, Mh2.T, Vp2.T, Ms2.T, [M02], [H2], Ms_full2.T), axis = 0))
    SFH3 = np.transpose(np.concatenate(([ID3], [upid3], [vp3], mm3, Mh3.T, Vp3.T, Ms3.T, [M03], [H3], Ms_full3.T), axis = 0))
    SFH4 = np.transpose(np.concatenate(([ID4], [upid4], [vp4], mm4, Mh4.T, Vp4.T, Ms4.T, [M04], [H4], Ms_full4.T), axis = 0))
    SFH5 = np.transpose(np.concatenate(([ID5], [upid5], [vp5], mm5, Mh5.T, Vp5.T, Ms5.T, [M05], [H5], Ms_full5.T), axis = 0))
    SFH6 = np.transpose(np.concatenate(([ID6], [upid6], [vp6], mm6, Mh6.T, Vp6.T, Ms6.T, [M06], [H6], Ms_full6.T), axis = 0))
    SFH7 = np.transpose(np.concatenate(([ID7], [upid7], [vp7], mm7, Mh7.T, Vp7.T, Ms7.T, [M07], [H7], Ms_full7.T), axis = 0))
    
    with open(wr+'/sfh1.txt', 'a+') as sfh1:
        np.savetxt(sfh1, SFH1)  
    with open(wr+'/sfh2.txt', 'a+') as sfh2:
        np.savetxt(sfh2, SFH2)  
    with open(wr+'/sfh3.txt', 'a+') as sfh3:
        np.savetxt(sfh3, SFH3)  
    with open(wr+'/sfh4.txt', 'a+') as sfh4:
        np.savetxt(sfh4, SFH4)  
    with open(wr+'/sfh5.txt', 'a+') as sfh5:
        np.savetxt(sfh5, SFH5)  
    with open(wr+'/sfh6.txt', 'a+') as sfh6:
        np.savetxt(sfh6, SFH6)  
    with open(wr+'/sfh7.txt', 'a+') as sfh7:
        np.savetxt(sfh7, SFH7)  
    
    return

if __name__ == '__main__':
    p = Pool(16)
    p.map(get_ma, Halo_no)