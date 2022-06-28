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
BASE_DIR='/sdf/group/kipac/u/ycwang/MWmass_new/Halo416/output/rockstar'
sz_zoom = np.loadtxt(BASE_DIR+'/outputs/scales.txt')
scales = sz_zoom[:, 1]
z = 1/scales - 1
t = cosmo.lookback_time(z).value

ID_host = 8297694
ID_sat_1 = 8304124
ID_sat_2 = 8304070
ID_sat_3 = 8306921
ID_sat_17 = 8296531
ID_sat_r = 8297828
baseDir = '/sdf/group/kipac/u/ycwang/MWmass_new/tracing'

def get_ma(halo_no):
    with open(BASE_DIR+'/groupcat/sfh_catalog_{:.6f}.'.format(scales[-1])+'0.txt', 'r') as f:
        sfh_test = np.loadtxt(f)
        ID = sfh_test[:, 0]
        SFH = sfh_test[:, 24:24+len(scales)] #Use this
        Mstar_mainH = sfh_test[:, 496:496+len(scales)]
        f.close()
        
    if ID_host in ID:
        idx = np.where(ID == ID_host)[0][0]
        res0 = np.vstack((SFH[idx], Mstar_mainH[idx])).T
        np.savetxt(baseDir+'/SFH/sfh_zoom_0.txt', res0)
    
    if ID_sat_1 in ID:
        idx = np.where(ID == ID_sat_1)[0][0]
        res1 = np.vstack((SFH[idx], Mstar_mainH[idx])).T
        np.savetxt(baseDir+'/SFH/sfh_zoom_1.txt', res1)
    
    if ID_sat_2 in ID:
        idx = np.where(ID == ID_sat_2)[0][0]
        res2 = np.vstack((SFH[idx], Mstar_mainH[idx])).T
        np.savetxt(baseDir+'/SFH/sfh_zoom_2.txt', res2)
        
    if ID_sat_3 in ID:
        idx = np.where(ID == ID_sat_3)[0][0]
        res3 = np.vstack((SFH[idx], Mstar_mainH[idx])).T
        np.savetxt(baseDir+'/SFH/sfh_zoom_3.txt', res3)
        
    if ID_sat_17 in ID:
        idx = np.where(ID == ID_sat_17)[0][0]
        res20 = np.vstack((SFH[idx], Mstar_mainH[idx])).T
        np.savetxt(baseDir+'/SFH/sfh_zoom_17.txt', res20)
      
    if ID_sat_r in ID:
        idx = np.where(ID == ID_sat_r)[0][0]
        resR2 = np.vstack((SFH[idx], Mstar_mainH[idx])).T
        np.savetxt(baseDir+'/SFH/sfh_zoom_r.txt', resR2)
        
    return

get_ma(416)
