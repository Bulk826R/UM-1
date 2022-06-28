from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
from helpers.SimulationAnalysis import readHlist, SimulationAnalysis, iterTrees

h = 0.7
omega_m = 0.286
BASE_DIR3 = '/sdf/group/kipac/u/ycwang/MWmass_new/Halo416/output/rockstar'
fields = ['scale','id', 'pid', 'upid', 'mvir', 'rvir', 'mpeak', 'vmax', 'vpeak', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'M200b',
          'Depth_first_ID','scale_of_last_MM', 'Tree_root_ID', 'Orig_halo_ID', 'Last_progenitor_depthfirst_ID']



def phase_space_dist(X, V, X0, V0):
    x = X[:, 0] * 1e3
    y = X[:, 1] * 1e3
    z = X[:, 2] * 1e3
    vx = V[:, 0]
    vy = V[:, 1]
    vz = V[:, 2]
    
    x0 = X0[:, 0] * 1e3
    y0 = X0[:, 1] * 1e3
    z0 = X0[:, 2] * 1e3
    vx0 = V0[:, 0]
    vy0 = V0[:, 1]
    vz0 = V0[:, 2]
    
    r0 = np.sqrt(x0**2 + y0**2 + z0**2)
    v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    
    dist = np.sqrt((x-x0)**2 + (y-y0)**2 +(z-z0)**2 + ((vx-vx0)**2 + (vy-vy0)**2 + (vz-vz0)**2) )
    return dist

def phase_space_dist_w(X, V, X0, V0):
    x = X[:, 0] * 1e3
    y = X[:, 1] * 1e3
    z = X[:, 2] * 1e3
    vx = V[:, 0]
    vy = V[:, 1]
    vz = V[:, 2]
    
    x0 = X0[:, 0] * 1e3
    y0 = X0[:, 1] * 1e3
    z0 = X0[:, 2] * 1e3
    vx0 = V0[:, 0]
    vy0 = V0[:, 1]
    vz0 = V0[:, 2]
    
    r0 = np.sqrt(x0**2 + y0**2 + z0**2)
    v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
    
    dist = np.sqrt((x-x0)**2 + (y-y0)**2 +(z-z0)**2 + (r0/v0)**2 * ((vx-vx0)**2 + (vy-vy0)**2 + (vz-vz0)**2) )
    return dist

def real_space_dist(X, X0):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    
    x0 = X0[:, 0]
    y0 = X0[:, 1]
    z0 = X0[:, 2]

    dist = np.sqrt((x-x0)**2 + (y-y0)**2 +(z-z0)**2)
    return dist

def halo_centric_dist(A, A0):
    x = A[:, 0]
    y = A[:, 1]
    z = A[:, 2]
    x0 = A0[:, 0]
    y0 = A0[:, 1]
    z0 = A0[:, 2]
    return np.vstack((x - x0*np.ones_like(x), y - y0*np.ones_like(y), z - z0*np.ones_like(z))).T

##############################################
#Find IDs of Halo416
Trees_Dir_416 = ('{}/trees'.format(BASE_DIR3))

halo_416 = readHlist(os.path.join(BASE_DIR3, 'hlists', 'hlist_1.00000.list'), fields)
ID_416 = halo_416['id']
UpID_416 = halo_416['upid']
tree_root_id_416 = halo_416['Tree_root_ID']
dep_1st_id_416 = halo_416['Depth_first_ID']
ori_halo_416 = halo_416['Orig_halo_ID']
pID_416 = halo_416['pid']
UpID_416 = halo_416['upid']
Mpeak_416 = halo_416['mpeak']
Mvir_416 = halo_416['mvir']
Vpeak_416 = halo_416['vpeak']
Vmax_416 = halo_416['vmax']
Rvir_416 = halo_416['rvir']

x_416 = halo_416['x']
y_416 = halo_416['y']
z_416 = halo_416['z']
X_416 = np.vstack((x_416, y_416, z_416)).T
vx_416 = halo_416['vx']
vy_416 = halo_416['vy']
vz_416 = halo_416['vz']
V_416 = np.vstack((vx_416, vy_416, vz_416)).T

id_max = np.argmax(Mvir_416)
Sat_ID5 = np.where(UpID_416 == ID_416[id_max])[0][0:17]



'''
xx = X_416[1:100]
rsat = np.sqrt((xx[:, 0]-X_416[id_max, 0])**2 + (xx[:, 1]-X_416[id_max, 1])**2 + (xx[:, 2]-X_416[id_max, 2])**2) * 1e3 / h
Rsat = np.sqrt((xx[:, 1]-X_416[id_max, 1])**2 + (xx[:, 2]-X_416[id_max, 2])**2) * 1e3 / h
idr = np.where(Rsat <= 200)[0]
print(idr)
print(xx[idr])
dx = (xx[idr, 0] - X_416[id_max, 0]) * 1e3 / h
dy = (xx[idr, 1] - X_416[id_max, 1]) * 1e3 / h
dz = (xx[idr, 2] - X_416[id_max, 2]) * 1e3 / h
print(dx)
print(dy)
print(dz)
print(rsat[idr], Rsat[idr])


ids = idr[5] # R1:2, R2:5
print('Max ID = ', ID_416[ids+1]) #+1 for skipping the MW host in xx
print('Max pID = ', pID_416[ids+1])
print('Max UpID = ', UpID_416[ids+1])
print('Max Mpeak = ', np.log10(Mpeak_416[ids+1]))
print('Max Mvir = ', np.log10(Mvir_416[ids+1]))
print('Max Vpeak = ', Vpeak_416[ids+1])
print('Max Vmax = ', Vmax_416[ids+1])
print('Max Rvir = ', Rvir_416[ids+1])
print('Max X, V = ', X_416[ids+1], V_416[ids+1])



print('Max ID = ', ID_416[id_max])
print('Max pID = ', pID_416[id_max])
print('Max UpID = ', UpID_416[id_max])
print('Max Mpeak = ', np.log10(Mpeak_416[id_max]))
print('Max Mvir = ', np.log10(Mvir_416[id_max]))
print('Max Vpeak = ', Vpeak_416[id_max])
print('Max Vmax = ', Vmax_416[id_max])
print('Max Rvir = ', Rvir_416[id_max])
print('Max X, V = ', X_416[id_max], V_416[id_max])

print('Sat IDs = ', ID_416[Sat_ID5])
print('Sat pIDs = ', pID_416[Sat_ID5])
print('Sat UpIDs = ', UpID_416[Sat_ID5])
print('Sat Mpeak = ', np.log10(Mpeak_416[Sat_ID5]))
print('Sat Mvir = ', np.log10(Mvir_416[Sat_ID5]))
print('Sat Vpeak = ', Vpeak_416[Sat_ID5])
print('Sat Vmax = ', Vmax_416[Sat_ID5])
print('Sat X, V = \n', X_416[Sat_ID5], '\n', V_416[Sat_ID5])
print('\n')
'''
###################################
###################################

#id_trace = id_max
#id_trace = Sat_ID5[2] # 0, 1, 2, -1(16)
#id_trace = ids
ID_host = 8297694
ID_sat_1 = 8304124
ID_sat_2 = 8304070
ID_sat_3 = 8306921
ID_sat_17 = 8296531
ID_sat_r = 8297828
id_trace = np.where(ID_416 == ID_host)[0][0]

###################################
###################################
'''
if id_trace in Sat_ID5:
    x_416_cent = halo_416['x'][id_max]
    y_416_cent = halo_416['y'][id_max]
    z_416_cent = halo_416['z'][id_max]
    vx_416_cent = halo_416['vx'][id_max]
    vy_416_cent = halo_416['vy'][id_max]
    vz_416_cent = halo_416['vz'][id_max]
    
    x_416_sat_trace = halo_416['x'][id_trace]
    y_416_sat_trace = halo_416['y'][id_trace]
    z_416_sat_trace = halo_416['z'][id_trace]
    vx_416_sat_trace = halo_416['vx'][id_trace]
    vy_416_sat_trace = halo_416['vy'][id_trace]
    vz_416_sat_trace = halo_416['vz'][id_trace]
    X_416_sat = np.vstack((x_416_sat_trace - x_416_cent, y_416_sat_trace - y_416_cent, z_416_sat_trace - z_416_cent)).T
    V_416_sat = np.vstack((vx_416_sat_trace - vx_416_cent, vy_416_sat_trace - vy_416_cent, vz_416_sat_trace - vz_416_cent)).T
    print('Tracer in Halo416 X-cent, V-cent = ', X_416_sat[0], V_416_sat[0])
    print('R cent (kpc) = ', np.sqrt((X_416_sat[0][0]*1e3)**2 + (X_416_sat[0][1]*1e3)**2 + (X_416_sat[0][2]*1e3)**2))
'''
    
print('ID_trace = ', id_trace)
print('Tracer in Halo416 ID = ', ID_416[id_trace])
print('Tracer in Halo416 Mpeak = ', np.log10(Mpeak_416[id_trace]))
print('Tracer in Halo416 Mvir = ', np.log10(Mvir_416[id_trace]))
print('Tracer in Halo416 Vpeak = ', Vpeak_416[id_trace])
print('Tracer in Halo416 Vmax = ', Vmax_416[id_trace])
print('Tracer in Halo416 Vmax = ', Vmax_416[id_trace])
print('Tracer in Halo416 X, V = ', X_416[id_trace], V_416[id_trace])

#####################
#Tracing routine

sim = {} #sim dictionary initialization
sim['Halo416'] = SimulationAnalysis(trees_dir = Trees_Dir_416)

def trace_vmax_zoom(ID):
    tree_halo_416 = sim['Halo416'].load_tree(tree_root_id_416[ID], 
                        additional_fields = 
                        ['Depth_first_ID', 'desc_id', 'mmp', 'vx', 'vy', 'vz',
                         'pid', 'M200c', 'Last_progenitor_depthfirst_ID'])
    A = [tree_halo_416['scale'][0]]
    VMax = [tree_halo_416['vmax'][0]]
    MVir = [tree_halo_416['mvir'][0]]
    RVir = [tree_halo_416['rvir'][0]]
    Xz = [tree_halo_416['x'][0]]
    Yz = [tree_halo_416['y'][0]]
    Zz = [tree_halo_416['z'][0]]
    Vx = [tree_halo_416['vx'][0]]
    Vy = [tree_halo_416['vy'][0]]
    Vz = [tree_halo_416['vz'][0]]
    halo_id = tree_halo_416['id'][0]
    while (halo_id > np.min(tree_halo_416['desc_id'][1:])):
        try:
            id_tree = np.where(tree_halo_416['desc_id'] == halo_id)[0][0]
            a = tree_halo_416['scale'][id_tree]
            vMax = tree_halo_416['vmax'][id_tree]
            mVir = tree_halo_416['mvir'][id_tree]
            rVir = tree_halo_416['rvir'][id_tree]
            xz = tree_halo_416['x'][id_tree]
            yz = tree_halo_416['y'][id_tree]
            zz = tree_halo_416['z'][id_tree]
            vx = tree_halo_416['vx'][id_tree]
            vy = tree_halo_416['vy'][id_tree]
            vz = tree_halo_416['vz'][id_tree]
            A.append(a)
            VMax.append(vMax)
            MVir.append(mVir)
            RVir.append(rVir)
            Xz.append(xz)
            Yz.append(yz)
            Zz.append(zz)
            Vx.append(vx)
            Vy.append(vy)
            Vz.append(vz)
            halo_id = tree_halo_416['id'][id_tree]
        except:
            break
        
    A = np.asarray(A)
    VMax = np.asarray(VMax)
    MVir = np.asarray(MVir)
    RVir = np.asarray(RVir)
    Xz = np.asarray(Xz)
    Yz = np.asarray(Yz)
    Zz = np.asarray(Zz)
    Vx = np.asarray(Vx)
    Vy = np.asarray(Vy)
    Vz = np.asarray(Vz)
    return np.vstack((A, VMax, MVir, RVir, Xz, Yz, Zz, Vx, Vy, Vz)).T
      
v_test = trace_vmax_zoom(id_trace)
np.savetxt('/sdf/group/kipac/u/ycwang/MWmass_new/tracing/v_test_416_'+str(id_trace)+'.txt', v_test)