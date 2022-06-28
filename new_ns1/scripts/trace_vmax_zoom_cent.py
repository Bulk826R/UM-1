from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees
from astropy.cosmology import FlatLambdaCDM
from multiprocessing import Pool
#from astropy import units as u
cosmo = FlatLambdaCDM(H0=70.00, Om0=0.286)
h = 0.7
omega_m = 0.286
fields = ['scale','id', 'pid', 'upid', 'mvir', 'rvir', 'mpeak', 'vmax', 'vpeak', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'M200b',
          'Depth_first_ID','scale_of_last_MM', 'Tree_root_ID', 'Orig_halo_ID', 'Last_progenitor_depthfirst_ID']

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

sim = {} #sim dictionary initialization
def trace(chunk_no):
    IDs = np.loadtxt(wr+'/ID{}_np_cent.txt'.format(chunk_no))
    ID = IDs[:, 0].astype(np.int64)
    halo = IDs[:, 1].astype(np.int64)
    
    idx = np.where(halo == Halo_no[0])[0]
    IDx = ID[idx]
    haloz = readHlist(src+'/Halo{0:03d}/output/rockstar/hlists/hlist_1.00000.list'.format(Halo_no[0]), fields)
    Trees_Dir = (src+'/Halo{0:03d}/output/rockstar/trees'.format(Halo_no[0]))
    sim['Halo{0:03d}'.format(Halo_no[0])] = SimulationAnalysis(trees_dir = Trees_Dir)           
    IDz = haloz['id'].astype(np.int64)
    tree_root_id = haloz['Tree_root_ID'].astype(np.int64)
    idz = np.where(IDz == IDx[0])[0][0]
    tree_halo = sim['Halo{0:03d}'.format(Halo_no[0])].load_main_branch(tree_root_id[idz], 
                                                                       additional_fields = 
                                                                       ['Depth_first_ID', 'desc_id', 
                                                                        'mmp', 'vx', 'vy', 'vz',
                                                                        'pid', 'M200c', 
                                                                        'Last_progenitor_depthfirst_ID'])
    vmax = tree_halo['vmax']
    if len(vmax) < len(scales):
        l0 = len(scales) - len(vmax)
        vmax = np.concatenate((vmax, np.zeros(l0)))
    Vmax = [vmax]
    
    for i in range(len(Halo_no)):
        idx = np.where(halo == Halo_no[i])[0]
        IDx = ID[idx]
    
        if (Halo_no[i] == 8247) or (Halo_no[i] == 9749) or (Halo_no[i] == 9829):    
            haloz = readHlist(src+'/Halo{0:04d}/output/rockstar/hlists/hlist_1.00000.list'.format(Halo_no[i]), fields)
            Trees_Dir = (src+'/Halo{0:04d}/output/rockstar/trees'.format(Halo_no[i]))
            sim['Halo{0:04d}'.format(Halo_no[i])] = SimulationAnalysis(trees_dir = Trees_Dir)            
        else:
            haloz = readHlist(src+'/Halo{0:03d}/output/rockstar/hlists/hlist_1.00000.list'.format(Halo_no[i]), fields)
            Trees_Dir = (src+'/Halo{0:03d}/output/rockstar/trees'.format(Halo_no[i]))
            sim['Halo{0:03d}'.format(Halo_no[i])] = SimulationAnalysis(trees_dir = Trees_Dir)            
        
        IDz = haloz['id'].astype(np.int64)
        tree_root_id = haloz['Tree_root_ID'].astype(np.int64)
        for j in range(len(IDx)):
            idz = np.where(IDz == IDx[j])[0][0]
            if (Halo_no[i] == 8247) or (Halo_no[i] == 9749) or (Halo_no[i] == 9829):    
                tree_halo = sim['Halo{0:04d}'.format(Halo_no[i])].load_main_branch(tree_root_id[idz], 
                                                                                   additional_fields = 
                                                                                   ['Depth_first_ID', 'desc_id', 
                                                                                    'mmp', 'vx', 'vy', 'vz',
                                                                                    'pid', 'M200c', 
                                                                                    'Last_progenitor_depthfirst_ID'])         
            else:
                tree_halo = sim['Halo{0:03d}'.format(Halo_no[i])].load_main_branch(tree_root_id[idz], 
                                                                                   additional_fields = 
                                                                                   ['Depth_first_ID', 'desc_id', 
                                                                                    'mmp', 'vx', 'vy', 'vz',
                                                                                    'pid', 'M200c', 
                                                                                    'Last_progenitor_depthfirst_ID'])
        
            vmax = tree_halo['vmax']
            if len(vmax) < len(scales):
                l0 = len(scales) - len(vmax)
                vmax = np.concatenate((vmax, np.zeros(l0)))
            else:
                continue
            #print(vmax)
            if i == 0 and j == 0:
                continue
            else:
                Vmax = np.concatenate((Vmax, [vmax]), axis = 0)
    
    np.savetxt(wr+'/vmax{}_np_cent.txt'.format(chunk_no), Vmax)
    return 

#a = trace(3)
#print(a)

#chunk = np.arange(1, 8, 1)
chunk = np.arange(1, 3, 1)

if __name__ == '__main__':
    p = Pool(2)
    p.map(trace, chunk)
        