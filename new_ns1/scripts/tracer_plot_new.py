from __future__ import unicode_literals
import numpy as np
import os
import itertools
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib
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
matplotlib.use('pdf')

def distance(x0, x1, dimensions): #ALWAYS use this function to calculate distance, accounts for periodic box
    delta = x0 - x1
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    delta = np.where(delta < -0.5 * dimensions, dimensions + delta, delta)
    return delta

###################
#Main halo

vt_416_0 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/v_test_416_0.txt')
a_416_0 = vt_416_0[:,0]
vmax_416_0 = vt_416_0[:,1]
mvir_416_0 = vt_416_0[:,2]
z_416_0 = 1/a_416_0 - 1
Rvir_host_zoom = vt_416_0[:, 3]
X_416_0 = vt_416_0[:, 4]
Y_416_0 = vt_416_0[:, 5]
Z_416_0 = vt_416_0[:, 6]

#print(Rvir_host_zoom, z_416_0)

star_zoom_0 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/SFH/sfh_zoom_0.txt')
SFH_zoom_0 = star_zoom_0[:, 0]
Mstar_zoom_0 = star_zoom_0[:, 1]

######

vt_1024_0 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_1024_0.txt')
a_1024_0 = vt_1024_0[:,0]
vmax_1024_0 = vt_1024_0[:,1]
mvir_1024_0 = vt_1024_0[:,2]
z_1024_0 = 1/a_1024_0 - 1
Rvir_host_1024 = vt_1024_0[:, 3]
X_1024_0 = vt_1024_0[:, 4]
Y_1024_0 = vt_1024_0[:, 5]
Z_1024_0 = vt_1024_0[:, 6]

#print(Rvir_host_1024, z_1024_0)

star_1024_0 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_1024_0.txt')
SFH_1024_0 = star_1024_0[:, 0]
Mstar_1024_0 = star_1024_0[:, 1]

######

vt_2048_0 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_2048_0.txt')
a_2048_0 = vt_2048_0[:,0]
vmax_2048_0 = vt_2048_0[:,1]
mvir_2048_0 = vt_2048_0[:,2]
z_2048_0 = 1/a_2048_0 - 1
Rvir_host_2048 = vt_2048_0[:, 3]
X_2048_0 = vt_2048_0[:, 4]
Y_2048_0 = vt_2048_0[:, 5]
Z_2048_0 = vt_2048_0[:, 6]

#print(Rvir_host_2048, z_2048_0)

star_2048_0 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_2048_0.txt')
SFH_2048_0 = star_2048_0[:, 0]
Mstar_2048_0 = star_2048_0[:, 1]

###################
#Sat No.1

vt_416_1 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/v_test_416_1.txt')
a_416_1 = vt_416_1[:,0]
vmax_416_1 = vt_416_1[:,1]
mvir_416_1 = vt_416_1[:,2]
z_416_1 = 1/a_416_1 - 1
X_416_1 = vt_416_1[:, 4]
Y_416_1 = vt_416_1[:, 5]
Z_416_1 = vt_416_1[:, 6]

lenz = np.min([len(X_416_1), len(X_416_0)])
rh_zoom_1 = np.sqrt((X_416_1[0:lenz] - X_416_0[0:lenz])**2 + \
                    (Y_416_1[0:lenz] - Y_416_0[0:lenz])**2 + \
                    (Z_416_1[0:lenz] - Z_416_0[0:lenz])**2) * 1e3
rr1 = rh_zoom_1/Rvir_host_zoom[0:len(rh_zoom_1)]
rR1 = rr1[::-1]
for i in range(len(rR1)):
    if rR1[i] >= 1.:
        continue
    else: 
        break

zin_zoom_1 = z_416_1[::-1][i]
rr_zoom_1 = rr1

star_zoom_1 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/SFH/sfh_zoom_1.txt')
SFH_zoom_1 = star_zoom_1[:, 0]
Mstar_zoom_1 = star_zoom_1[:, 1]

######

vt_1024_1 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_1024_2.txt')
a_1024_1 = vt_1024_1[:,0]
vmax_1024_1 = vt_1024_1[:,1]
mvir_1024_1 = vt_1024_1[:,2]
z_1024_1 = 1/a_1024_1 - 1
X_1024_1 = vt_1024_1[:, 4]
Y_1024_1 = vt_1024_1[:, 5]
Z_1024_1 = vt_1024_1[:, 6]

rh_1024_1 = np.sqrt((X_1024_1 - X_1024_0[0:len(X_1024_1)])**2 + \
                    (Y_1024_1 - Y_1024_0[0:len(Y_1024_1)])**2 + \
                    (Z_1024_1 - Z_1024_0[0:len(Z_1024_1)])**2) * 1e3
rr1 = rh_1024_1/Rvir_host_1024[0:len(rh_1024_1)]
rR1 = rr1[::-1]
for i in range(len(rR1)):
    if rR1[i] >= 1.:
        continue
    else: 
        break

zin_1024_1 = z_1024_1[::-1][i]
rr_1024_1 = rr1

star_1024_1 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_1024_1.txt')
SFH_1024_1 = star_1024_1[:, 0]
Mstar_1024_1 = star_1024_1[:, 1]

######

vt_2048_1 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_2048_2.txt')
a_2048_1 = vt_2048_1[:,0]
vmax_2048_1 = vt_2048_1[:,1]
mvir_2048_1 = vt_2048_1[:,2]
z_2048_1 = 1/a_2048_1 - 1
X_2048_1 = vt_2048_1[:, 4]
Y_2048_1 = vt_2048_1[:, 5]
Z_2048_1 = vt_2048_1[:, 6]

rh_2048_1 = np.sqrt((X_2048_1 - X_2048_0[0:len(X_2048_1)])**2 + \
                    (Y_2048_1 - Y_2048_0[0:len(Y_2048_1)])**2 + \
                    (Z_2048_1 - Z_2048_0[0:len(Z_2048_1)])**2) * 1e3
rr1 = rh_2048_1/Rvir_host_2048[0:len(rh_2048_1)]
rR1 = rr1[::-1]
for i in range(len(rR1)):
    if rR1[i] >= 1.:
        continue
    else: 
        break

zin_2048_1 = z_2048_1[::-1][i]
rr_2048_1 = rr1

star_2048_1 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_2048_1.txt')
SFH_2048_1 = star_2048_1[:, 0]
Mstar_2048_1 = star_2048_1[:, 1]

###################
#Sat No.2

vt_416_2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/v_test_416_6.txt')
a_416_2 = vt_416_2[:,0]
vmax_416_2 = vt_416_2[:,1]
mvir_416_2 = vt_416_2[:,2]
z_416_2 = 1/a_416_2 - 1
X_416_2 = vt_416_2[:, 4]
Y_416_2 = vt_416_2[:, 5]
Z_416_2 = vt_416_2[:, 6]

lenz = np.min([len(X_416_2), len(X_416_0)])
rh_zoom_2 = np.sqrt((X_416_2[0:lenz] - X_416_0[0:lenz])**2 + \
                    (Y_416_2[0:lenz] - Y_416_0[0:lenz])**2 + \
                    (Z_416_2[0:lenz] - Z_416_0[0:lenz])**2) * 1e3
rr2 = rh_zoom_2/Rvir_host_zoom[0:len(rh_zoom_2)]
rR2 = rr2[::-1]
for i in range(len(rR2)):
    if rR2[i] >= 1.:
        continue
    else: 
        break

zin_zoom_2 = z_416_2[::-1][i]
rr_zoom_2 = rr2

star_zoom_2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/SFH/sfh_zoom_2.txt')
SFH_zoom_2 = star_zoom_2[:, 0]
Mstar_zoom_2 = star_zoom_2[:, 1]

######

vt_1024_2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_1024_6.txt')
a_1024_2 = vt_1024_2[:,0]
vmax_1024_2 = vt_1024_2[:,1]
mvir_1024_2 = vt_1024_2[:,2]
z_1024_2 = 1/a_1024_2 - 1
X_1024_2 = vt_1024_2[:, 4]
Y_1024_2 = vt_1024_2[:, 5]
Z_1024_2 = vt_1024_2[:, 6]

rh_1024_2 = np.sqrt((X_1024_2 - X_1024_0[0:len(X_1024_2)])**2 + \
                    (Y_1024_2 - Y_1024_0[0:len(Y_1024_2)])**2 + \
                    (Z_1024_2 - Z_1024_0[0:len(Z_1024_2)])**2) * 1e3
rr2 = rh_1024_2/Rvir_host_1024[0:len(rh_1024_2)]
rR2 = rr2[::-1]
for i in range(len(rR2)):
    if rR2[i] >= 1.:
        continue
    else: 
        break

zin_1024_2 = z_1024_2[::-1][i]
rr_1024_2 = rr2

star_1024_2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_1024_2.txt')
SFH_1024_2 = star_1024_2[:, 0]
Mstar_1024_2 = star_1024_2[:, 1]

######

vt_2048_2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_2048_6.txt')
a_2048_2 = vt_2048_2[:,0]
vmax_2048_2 = vt_2048_2[:,1]
mvir_2048_2 = vt_2048_2[:,2]
z_2048_2 = 1/a_2048_2 - 1
X_2048_2 = vt_2048_2[:, 4]
Y_2048_2 = vt_2048_2[:, 5]
Z_2048_2 = vt_2048_2[:, 6]

rh_2048_2 = np.sqrt((X_2048_2 - X_2048_0[0:len(X_2048_2)])**2 + \
                    (Y_2048_2 - Y_2048_0[0:len(Y_2048_2)])**2 + \
                    (Z_2048_2 - Z_2048_0[0:len(Z_2048_2)])**2) * 1e3
rr2 = rh_2048_2/Rvir_host_2048[0:len(rh_2048_2)]
rR2 = rr2[::-1]
for i in range(len(rR2)):
    if rR2[i] >= 1.:
        continue
    else: 
        break

zin_2048_2 = z_2048_2[::-1][i]
rr_2048_2 = rr2

star_2048_2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_2048_2.txt')
SFH_2048_2 = star_2048_2[:, 0]
Mstar_2048_2 = star_2048_2[:, 1]

###################
#Sat No.3

vt_416_3 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/v_test_416_19.txt')
a_416_3 = vt_416_3[:,0]
vmax_416_3 = vt_416_3[:,1]
mvir_416_3 = vt_416_3[:,2]
z_416_3 = 1/a_416_3 - 1
X_416_3 = vt_416_3[:, 4]
Y_416_3 = vt_416_3[:, 5]
Z_416_3 = vt_416_3[:, 6]

lenz = np.min([len(X_416_3), len(X_416_0)])
rh_zoom_3 = np.sqrt((X_416_3[0:lenz] - X_416_0[0:lenz])**2 + \
                    (Y_416_3[0:lenz] - Y_416_0[0:lenz])**2 + \
                    (Z_416_3[0:lenz] - Z_416_0[0:lenz])**2) * 1e3
rr3 = rh_zoom_3/Rvir_host_zoom[0:len(rh_zoom_3)]
rR3 = rr3[::-1]
for i in range(len(rR3)):
    if rR3[i] >= 1.:
        continue
    else: 
        break

zin_zoom_3 = z_416_3[::-1][i]
rr_zoom_3 = rr3

star_zoom_3 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/SFH/sfh_zoom_3.txt')
SFH_zoom_3 = star_zoom_3[:, 0]
Mstar_zoom_3 = star_zoom_3[:, 1]

######

vt_1024_3 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_1024_27.txt')
a_1024_3 = vt_1024_3[:,0]
vmax_1024_3 = vt_1024_3[:,1]
mvir_1024_3 = vt_1024_3[:,2]
z_1024_3 = 1/a_1024_3 - 1
X_1024_3 = vt_1024_3[:, 4]
Y_1024_3 = vt_1024_3[:, 5]
Z_1024_3 = vt_1024_3[:, 6]

rh_1024_3 = np.sqrt((X_1024_3 - X_1024_0[0:len(X_1024_3)])**2 + \
                    (Y_1024_3 - Y_1024_0[0:len(Y_1024_3)])**2 + \
                    (Z_1024_3 - Z_1024_0[0:len(Z_1024_3)])**2) * 1e3
rr3 = rh_1024_3/Rvir_host_1024[0:len(rh_1024_3)]
rR3 = rr3[::-1]
for i in range(len(rR3)):
    if rR3[i] >= 1.:
        continue
    else: 
        break

zin_1024_3 = z_1024_3[::-1][i]
rr_1024_3 = rr3

star_1024_3 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_1024_3.txt')
SFH_1024_3 = star_1024_3[:, 0]
Mstar_1024_3 = star_1024_3[:, 1]

######

vt_2048_3 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_2048_27.txt')
a_2048_3 = vt_2048_3[:,0]
vmax_2048_3 = vt_2048_3[:,1]
mvir_2048_3 = vt_2048_3[:,2]
z_2048_3 = 1/a_2048_3 - 1
X_2048_3 = vt_2048_3[:, 4]
Y_2048_3 = vt_2048_3[:, 5]
Z_2048_3 = vt_2048_3[:, 6]

rh_2048_3 = np.sqrt((X_2048_3 - X_2048_0[0:len(X_2048_3)])**2 + \
                    (Y_2048_3 - Y_2048_0[0:len(Y_2048_3)])**2 + \
                    (Z_2048_3 - Z_2048_0[0:len(Z_2048_3)])**2) * 1e3
rr3 = rh_2048_3/Rvir_host_2048[0:len(rh_2048_3)]
rR3 = rr3[::-1]
for i in range(len(rR3)):
    if rR3[i] >= 1.:
        continue
    else: 
        break

zin_2048_3 = z_2048_3[::-1][i]
rr_2048_3 = rr3

star_2048_3 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_2048_3.txt')
SFH_2048_3 = star_2048_3[:, 0]
Mstar_2048_3 = star_2048_3[:, 1]

###################
#Sat No.4

vt_416_4 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/v_test_416_287.txt')
a_416_4 = vt_416_4[:,0]
vmax_416_4 = vt_416_4[:,1]
mvir_416_4 = vt_416_4[:,2]
z_416_4 = 1/a_416_4 - 1
X_416_4 = vt_416_4[:, 4]
Y_416_4 = vt_416_4[:, 5]
Z_416_4 = vt_416_4[:, 6]

lenz = np.min([len(X_416_4), len(X_416_0)])
rh_zoom_4 = np.sqrt((X_416_4[0:lenz] - X_416_0[0:lenz])**2 + \
                    (Y_416_4[0:lenz] - Y_416_0[0:lenz])**2 + \
                    (Z_416_4[0:lenz] - Z_416_0[0:lenz])**2) * 1e3
rr4 = rh_zoom_4/Rvir_host_zoom[0:len(rh_zoom_4)]
rR4 = rr4[::-1]
for i in range(len(rR4)):
    if rR4[i] >= 1.:
        continue
    else: 
        break

zin_zoom_4 = z_416_4[::-1][i]
rr_zoom_4 = rr4

star_zoom_4 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/SFH/sfh_zoom_17.txt')
SFH_zoom_4 = star_zoom_4[:, 0]
Mstar_zoom_4 = star_zoom_4[:, 1]

######

vt_2048_4 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/v_test_2048_334.txt')
a_2048_4 = vt_2048_4[:,0]
vmax_2048_4 = vt_2048_4[:,1]
mvir_2048_4 = vt_2048_4[:,2]
z_2048_4 = 1/a_2048_4 - 1
X_2048_4 = vt_2048_4[:, 4]
Y_2048_4 = vt_2048_4[:, 5]
Z_2048_4 = vt_2048_4[:, 6]

rh_2048_4 = np.sqrt((X_2048_4 - X_2048_0[0:len(X_2048_4)])**2 + \
                    (Y_2048_4 - Y_2048_0[0:len(Y_2048_4)])**2 + \
                    (Z_2048_4 - Z_2048_0[0:len(Z_2048_4)])**2) * 1e3

rr4 = rh_2048_4/Rvir_host_2048[0:len(rh_2048_4)]
rR4 = rr4[::-1]
for i in range(len(rR4)):
    if rR4[i] >= 1.:
        continue
    else: 
        break

zin_2048_4 = z_2048_4[::-1][i]
rr_2048_4 = rr4

star_2048_4 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_2048_20.txt')
SFH_2048_4 = star_2048_4[:, 0]
Mstar_2048_4 = star_2048_4[:, 1]

###################
#Sat R2

vt_416_R2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/v_test_416_26.txt')
a_416_R2 = vt_416_R2[:,0]
vmax_416_R2 = vt_416_R2[:,1]
mvir_416_R2 = vt_416_R2[:,2]
z_416_R2 = 1/a_416_R2 - 1
X_416_R2 = vt_416_R2[:, 4]
Y_416_R2 = vt_416_R2[:, 5]
Z_416_R2 = vt_416_R2[:, 6]

lenz = np.min([len(X_416_R2), len(X_416_0)])
rh_zoom_R2 = np.sqrt((X_416_R2[0:lenz] - X_416_0[0:lenz])**2 + \
                    (Y_416_R2[0:lenz] - Y_416_0[0:lenz])**2 + \
                    (Z_416_R2[0:lenz] - Z_416_0[0:lenz])**2) * 1e3
rrR2 = rh_zoom_R2/Rvir_host_zoom[0:len(rh_zoom_R2)]
rRR2 = rrR2[::-1]
for i in range(len(rRR2)):
    if rRR2[i] >= 1.:
        continue
    else: 
        break

zin_zoom_R2 = z_416_R2[::-1][i]
rr_zoom_R2 = rrR2

star_zoom_R2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/tracing/SFH/sfh_zoom_r.txt')
SFH_zoom_R2 = star_zoom_R2[:, 0]
Mstar_zoom_R2 = star_zoom_R2[:, 1]

######

vt_1024_R2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/V_test_1024_Rp2.txt')
a_1024_R2 = vt_1024_R2[:,0]
vmax_1024_R2 = vt_1024_R2[:,1]
mvir_1024_R2 = vt_1024_R2[:,2]
z_1024_R2 = 1/a_1024_R2 - 1
X_1024_R2 = vt_1024_R2[:, 4]
Y_1024_R2 = vt_1024_R2[:, 5]
Z_1024_R2 = vt_1024_R2[:, 6]

rh_1024_R2 = np.sqrt((X_1024_R2 - X_1024_0[0:len(X_1024_R2)])**2 + \
                    (Y_1024_R2 - Y_1024_0[0:len(Y_1024_R2)])**2 + \
                    (Z_1024_R2 - Z_1024_0[0:len(Z_1024_R2)])**2) * 1e3
rrR2 = rh_1024_R2/Rvir_host_1024[0:len(rh_1024_R2)]
rRR2 = rrR2[::-1]
for i in range(len(rRR2)):
    if rRR2[i] >= 1.:
        continue
    else: 
        break

zin_1024_R2 = z_1024_R2[::-1][i]
rr_1024_R2 = rrR2

star_1024_R2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_1024_R2.txt')
SFH_1024_R2 = star_1024_R2[:, 0]
Mstar_1024_R2 = star_1024_R2[:, 1]

######

vt_2048_R2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/V_test_2048_2.txt')
a_2048_R2 = vt_2048_R2[:,0]
vmax_2048_R2 = vt_2048_R2[:,1]
mvir_2048_R2 = vt_2048_R2[:,2]
z_2048_R2 = 1/a_2048_R2 - 1
X_2048_R2 = vt_2048_R2[:, 4]
Y_2048_R2 = vt_2048_R2[:, 5]
Z_2048_R2 = vt_2048_R2[:, 6]

rh_2048_R2 = np.sqrt((X_2048_R2[0:len(X_2048_0)] - X_2048_0)**2 + \
                    (Y_2048_R2[0:len(Y_2048_0)] - Y_2048_0)**2 + \
                    (Z_2048_R2[0:len(Z_2048_0)] - Z_2048_0)**2) * 1e3

rrR2 = rh_2048_R2[0:len(X_2048_0)]/Rvir_host_2048
rRR2 = rrR2[::-1]
for i in range(len(rRR2)):
    if rRR2[i] >= 1.:
        continue
    else: 
        break

z_2048_R2 = z_2048_R2[0:len(X_2048_0)]
zin_2048_R2 = z_2048_R2[::-1][i]
rr_2048_R2 = rrR2
vmax_2048_R2 = vmax_2048_R2[0:len(X_2048_0)]

star_2048_R2 = np.loadtxt('/home/bulk826/Desktop/Stanford/Research1/data/sat/tracing/SFH/sfh_2048_R2.txt')
SFH_2048_R2 = star_2048_R2[:, 0]
Mstar_2048_R2 = star_2048_R2[:, 1]



######################################################
fig.set_size_inches(20, 28)
plt.subplots_adjust(wspace = 0.24, hspace = 0.04)

from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
cosmo = FlatLambdaCDM(H0=70, Om0=0.286)
z_lbt = np.asarray([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 4, 8])
lbt = cosmo.lookback_time(z_lbt)
#print(lbt)
LBT = np.linspace(0, 12, 7)

###################################

ax1 = fig.add_subplot(8, 3, 1)
ax1.plot(cosmo.lookback_time(z_416_0[::-1]), vmax_416_0[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax1.plot(cosmo.lookback_time(z_1024_0[::-1]), vmax_1024_0[::-1], linewidth = '2.', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax1.plot(cosmo.lookback_time(z_2048_0[::-1]), vmax_2048_0[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax1.axhline(y = 10, linewidth = 1.6, color = 'navy', alpha = 0.8, linestyle = ':')
ax1.axhline(y = 60, linewidth = 1.6, color = 'darkturquoise',  alpha = 0.8, linestyle = ':')
ax1.axhline(y = 35, linewidth = 1.6, color = 'crimson', alpha = 0.8, linestyle = ':')

ax1.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 18, labelpad = 8)
ax1.set_xticks(lbt.value)
ax1.set_xticklabels(z_lbt)
ax1.tick_params(labelsize = 14, direction='in')
ax1.set_xticklabels([])
ax1.set_xlim([13.5, 0])
ax1.set_ylim([2, 212])
xmin, xmax = ax1.get_xlim()
ymin, ymax = ax1.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax1.text(xmin + width * 0.02, ymax - height * 0.04, r'MW, $\log_{10} (M_{\mathrm{Peak, zoom}}/\mathrm{M_{\odot}}) = 12.04$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)



ax12= ax1.twiny()
ax12.set_xticks(LBT[::-1])
ax12.set_xlim([13.5, 0])
ax12.tick_params(labelsize = 14, direction='in')
ax12.set_xlabel('Lookback time [Gyrs]', fontsize = 16, labelpad = 8)

######

ax2 = fig.add_subplot(8, 3, 2)
ax2.plot(cosmo.lookback_time(z_416_0[::-1]), Rvir_host_zoom[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax2.plot(cosmo.lookback_time(z_1024_0[::-1]), Rvir_host_1024[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax2.plot(cosmo.lookback_time(z_2048_0[::-1]), Rvir_host_2048[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')

ax2.set_ylabel(r'$R_{\mathrm{vir}}\,$[$\mathrm{kpc}/h$]', fontsize = 18, labelpad = 8)
ax2.tick_params(labelsize = 14, direction='in')
ax2.legend(loc='lower right', fontsize = 14, frameon = False, borderpad = 0.8)
#ax2.legend(loc='lower right', fontsize = 11)
ax2.set_xticks(lbt.value)
ax2.set_xticklabels(z_lbt)
ax2.set_xticklabels([])
ax2.set_xlim([13.5, 0])
xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax2.text(xmin + width * 0.02, ymax - height * 0.04, r'MW', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax22= ax2.twiny()
ax22.set_xticks(LBT[::-1])
ax22.set_xlim([13.5, 0])
ax22.tick_params(labelsize = 14, direction='in')
ax22.set_xlabel('Lookback time [Gyrs]', fontsize = 16, labelpad = 8)

######

ax31 = fig.add_subplot(8, 3, 3)

ax31.plot(cosmo.lookback_time(z_416_0[::-1]), SFH_zoom_0[len(SFH_zoom_0)-len(z_416_0):], linewidth = '2.4', 
          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax31.plot(cosmo.lookback_time(z_1024_0[::-1]), SFH_1024_0[len(SFH_1024_0)-len(z_1024_0):], 
          linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax31.plot(cosmo.lookback_time(z_2048_0[::-1]), SFH_2048_0[len(SFH_2048_0)-len(z_2048_0):], linewidth = '2.2', 
          color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')


ax31.set_ylabel(r'$\mathrm{SFR_{raw}}\,$[$\mathrm{\mathrm{M_{\odot}}\,yr^{-1}}$]', fontsize = 18, labelpad = 8)
ax31.set_ylim([10**(-2.4), 10**2])
ax31.set_yscale('log')
ax31.set_yticks(np.logspace(-2, 2, 5))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax31.yaxis.set_minor_locator(loc2)
ax31.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax31.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)


#ax31.legend(loc='upper right', fontsize = 11)
ax31.set_xticks(lbt.value)
ax31.set_xticklabels(z_lbt)
ax31.set_xticklabels([])
ax31.set_xlim([13.5, 0])
xmin, xmax = ax31.get_xlim()
ymin, ymax = ax31.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax31.text(xmin + width * 0.02, ymax * 0.72, r'MW, $\log_{10} (M_{\mathrm{\ast}}/\mathrm{M_{\odot}}) = 10.58$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax312= ax31.twiny()
ax312.set_xticks(LBT[::-1])
ax312.set_xlim([13.5, 0])
ax312.tick_params(labelsize = 14, direction='in')
ax312.set_xlabel('Lookback time [Gyrs]', fontsize = 16, labelpad = 8)

###################################

ax3 = fig.add_subplot(8, 3, 4)
ax3.plot(cosmo.lookback_time(z_416_1[::-1]), vmax_416_1[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax3.plot(cosmo.lookback_time(z_1024_1[::-1]), vmax_1024_1[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax3.plot(cosmo.lookback_time(z_2048_1[::-1]), vmax_2048_1[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax3.axhline(y = 10, linewidth = 1.6, color = 'navy', alpha = 0.8, linestyle = ':')
ax3.axhline(y = 60, linewidth = 1.6, color = 'darkturquoise',  alpha = 0.8, linestyle = ':')
ax3.axhline(y = 35, linewidth = 1.6, color = 'crimson', alpha = 0.8, linestyle = ':')

ax3.scatter([cosmo.lookback_time(zin_zoom_1).value], [50], s = 64, marker = '^', color = 'navy', alpha = 0.6, lw = 0.)
ax3.scatter([cosmo.lookback_time(zin_1024_1).value], [60], s = 64, marker = '^', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax3.scatter([cosmo.lookback_time(zin_2048_1).value], [50], s = 64, marker = '^', color = 'crimson', alpha = 0.6, lw = 0.)

ax3.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 18, labelpad = 8)
ax3.tick_params(labelsize = 14, direction='in')
#ax3.legend(loc='lower right', fontsize = 11)
ax3.set_xticks(lbt.value)
ax3.set_xticklabels(z_lbt)
ax3.set_xticklabels([])
ax3.set_xlim([13.5, 0])
ax3.set_ylim([2, 135])
xmin, xmax = ax3.get_xlim()
ymin, ymax = ax3.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax3.text(xmin + width * 0.02, ymax - height * 0.04, r'S1, $\log_{10} (M_{\mathrm{Peak, zoom}}/\mathrm{M_{\odot}}) = 11.32$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax32= ax3.twiny()
ax32.set_xticks(LBT[::-1])
ax32.set_xlim([13.5, 0])
ax32.tick_params(labelsize = 14, direction='in')
ax32.set_xticklabels([])

######

ax4 = fig.add_subplot(8, 3, 5)
ax4.plot(cosmo.lookback_time(z_416_1[0:len(rr_zoom_1)][::-1]), rr_zoom_1[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax4.plot(cosmo.lookback_time(z_1024_1[::-1]), rr_1024_1[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax4.plot(cosmo.lookback_time(z_2048_1[::-1]), rr_2048_1[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax4.axhline(y = 1., lw = 1.6, color = 'black', alpha = 0.8, linestyle = ':')

ax4.set_ylabel(r'$r_{\mathrm{3D}}/R_{\mathrm{vir, host}}$', fontsize = 18, labelpad = 8)
ax4.set_yscale('log')
ax4.set_yticks(np.logspace(0, 2, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax4.yaxis.set_minor_locator(loc2)
ax4.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax4.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax4.set_xticks(lbt.value)
ax4.set_xticklabels(z_lbt)
ax4.set_xticklabels([])
ax4.set_xlim([13.5, 0])
xmin, xmax = ax4.get_xlim()
ymin, ymax = ax4.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax4.text(xmin + width * 0.02, ymax  * 0.72, r'S1 (LMC analog)', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax4.scatter([cosmo.lookback_time(zin_zoom_1).value], [5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax4.scatter([cosmo.lookback_time(zin_1024_1).value], [10], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax4.scatter([cosmo.lookback_time(zin_2048_1).value], [5], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)


ax42= ax4.twiny()
ax42.set_xticks(LBT[::-1])
ax42.set_xlim([13.5, 0])
ax42.tick_params(labelsize = 14, direction='in')
ax42.set_xticklabels([])

######

ax61 = fig.add_subplot(8, 3, 6)

ax61.plot(cosmo.lookback_time(z_416_1[::-1]), SFH_zoom_1[len(SFH_zoom_1)-len(z_416_1):], linewidth = '2.4', 
          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax61.plot(cosmo.lookback_time(z_1024_1[::-1]), SFH_1024_1[len(SFH_1024_1)-len(z_1024_1):], 
          linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax61.plot(cosmo.lookback_time(z_2048_1[::-1]), SFH_2048_1[len(SFH_2048_1)-len(z_2048_1):], linewidth = '2.2', 
          color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax61.scatter([cosmo.lookback_time(zin_zoom_1).value], [10**0.], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax61.scatter([cosmo.lookback_time(zin_1024_1).value], [10**0.2], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax61.scatter([cosmo.lookback_time(zin_2048_1).value], [10**0.], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)

ax61.set_ylabel(r'$\mathrm{SFR_{raw}}\,$[$\mathrm{\mathrm{M_{\odot}}\,yr^{-1}}$]', fontsize = 18, labelpad = 8)
ax61.set_ylim([10**(-3.3), 10**1.2])
ax61.set_yscale('log')
ax61.set_yticks(np.logspace(-3, 1, 5))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax61.yaxis.set_minor_locator(loc2)
ax61.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax61.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax61.set_xticks(lbt.value)
ax61.set_xticklabels(z_lbt)
ax61.set_xticklabels([])
ax61.set_xlim([13.5, 0])
#ax61.set_ylim([-3.3, 1.2])
xmin, xmax = ax61.get_xlim()
ymin, ymax = ax61.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax61.text(xmin + width * 0.02, ymax  * 0.72, r'S1, $\log_{10} (M_{\mathrm{\ast}}/\mathrm{M_{\odot}}) = 9.38$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax612= ax61.twiny()
ax612.set_xticks(LBT[::-1])
ax612.set_xlim([13.5, 0])
ax612.tick_params(labelsize = 14, direction='in')
ax612.set_xticklabels([])

###################################

ax5 = fig.add_subplot(8, 3, 7)
ax5.plot(cosmo.lookback_time(z_416_2[::-1]), vmax_416_2[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax5.plot(cosmo.lookback_time(z_1024_2[::-1]), vmax_1024_2[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax5.plot(cosmo.lookback_time(z_2048_2[::-1]), vmax_2048_2[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax5.axhline(y = 10, linewidth = 1.6, color = 'navy', alpha = 0.8, linestyle = ':')
ax5.axhline(y = 60, linewidth = 1.6, color = 'darkturquoise',  alpha = 0.8, linestyle = ':')
ax5.axhline(y = 35, linewidth = 1.6, color = 'crimson', alpha = 0.8, linestyle = ':')
ax5.scatter([cosmo.lookback_time(zin_zoom_2).value], [50], s = 64, marker = '^', color = 'navy', alpha = 0.6, lw = 0.)
ax5.scatter([cosmo.lookback_time(zin_1024_2).value], [50], s = 64, marker = '^', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax5.scatter([cosmo.lookback_time(zin_2048_2).value], [50], s = 64, marker = '^', color = 'crimson', alpha = 0.6, lw = 0.)

ax5.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 18, labelpad = 8)
ax5.tick_params(labelsize = 14, direction='in')
#ax5.legend(loc='lower right', fontsize = 11)
ax5.set_xticks(lbt.value)
ax5.set_xticklabels(z_lbt)
ax5.set_xticklabels([])
ax5.set_xlim([13.5, 0])
ax5.set_ylim([2, 108])
xmin, xmax = ax5.get_xlim()
ymin, ymax = ax5.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax5.text(xmin + width * 0.02, ymax - height * 0.04, r'S2, $\log_{10} (M_{\mathrm{Peak, zoom}}/\mathrm{M_{\odot}}) = 10.90$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax52= ax5.twiny()
ax52.set_xticks(LBT[::-1])
ax52.set_xlim([13.5, 0])
ax52.tick_params(labelsize = 14, direction='in')
ax52.set_xticklabels([])

######

ax6 = fig.add_subplot(8, 3, 8)
ax6.plot(cosmo.lookback_time(z_416_2[0:len(rr_zoom_2)][::-1]), rr_zoom_2[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax6.plot(cosmo.lookback_time(z_1024_2[::-1]), rr_1024_2[::-1], linewidth = '2.', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax6.plot(cosmo.lookback_time(z_2048_2[::-1]), rr_2048_2[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax6.axhline(y = 1., lw = 1.6, color = 'black', alpha = 0.8, linestyle = ':')

ax6.set_ylabel(r'$r_{\mathrm{3D}}/R_{\mathrm{vir, host}}$', fontsize = 18, labelpad = 8)
ax6.set_yscale('log')
ax6.set_yticks(np.logspace(0, 2, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax6.yaxis.set_minor_locator(loc2)
ax6.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax6.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax6.scatter([cosmo.lookback_time(zin_zoom_2).value], [5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax6.scatter([cosmo.lookback_time(zin_1024_2).value], [5], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax6.scatter([cosmo.lookback_time(zin_2048_2).value], [5], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)


ax6.set_xticks(lbt.value)
ax6.set_xticklabels(z_lbt)
ax6.set_xticklabels([])
ax6.set_xlim([13.5, 0])
xmin, xmax = ax6.get_xlim()
ymin, ymax = ax6.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax6.text(xmin + width * 0.02, ymax * 0.72, r'S2 (Bright dwarf)', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax62 = ax6.twiny()
ax62.set_xticks(LBT[::-1])
ax62.set_xlim([13.5, 0])
ax62.tick_params(labelsize = 14, direction='in')
ax62.set_xticklabels([])

######

ax91 = fig.add_subplot(8, 3, 9)

ax91.plot(cosmo.lookback_time(z_416_2[::-1]), SFH_zoom_2[len(SFH_zoom_2)-len(z_416_2):], linewidth = '2.4', 
          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax91.plot(cosmo.lookback_time(z_1024_2[::-1]), SFH_1024_2[len(SFH_1024_2)-len(z_1024_2):], 
          linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax91.plot(cosmo.lookback_time(z_2048_2[::-1]), SFH_2048_2[len(SFH_2048_2)-len(z_2048_2):], linewidth = '2.2', 
          color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax91.scatter([cosmo.lookback_time(zin_zoom_2).value], [10**-0.5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax91.scatter([cosmo.lookback_time(zin_1024_2).value], [10**-0.5], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax91.scatter([cosmo.lookback_time(zin_2048_2).value], [10**-0.5], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)

ax91.set_ylabel(r'$\mathrm{SFR_{raw}}\,$[$\mathrm{\mathrm{M_{\odot}}\,yr^{-1}}$]', fontsize = 18, labelpad = 8)
ax91.set_ylim([10**(-3.5), 10**0.1])
ax91.set_yscale('log')
ax91.set_yticks(np.logspace(-3, 0, 4))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax91.yaxis.set_minor_locator(loc2)
ax91.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax91.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax91.set_xticks(lbt.value)
ax91.set_xticklabels(z_lbt)
ax91.set_xticklabels([])
ax91.set_xlim([13.5, 0])
#ax91.set_ylim([-3.5, -0.1])
xmin, xmax = ax91.get_xlim()
ymin, ymax = ax91.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax91.text(xmin + width * 0.02, ymax  * 0.72, r'S2, $\log_{10} (M_{\mathrm{\ast}}/\mathrm{M_{\odot}}) = 8.78$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax912= ax91.twiny()
ax912.set_xticks(LBT[::-1])
ax912.set_xlim([13.5, 0])
ax912.tick_params(labelsize = 14, direction='in')
ax912.set_xticklabels([])

###################################

ax7 = fig.add_subplot(8, 3, 10)
ax7.plot(cosmo.lookback_time(z_416_3[::-1]), vmax_416_3[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax7.plot(cosmo.lookback_time(z_1024_3[::-1]), vmax_1024_3[::-1], linewidth = '2.', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax7.plot(cosmo.lookback_time(z_2048_3[::-1]), vmax_2048_3[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax7.axhline(y = 10, linewidth = 1.6, color = 'navy', alpha = 0.8, linestyle = ':')
ax7.axhline(y = 60, linewidth = 1.6, color = 'darkturquoise',  alpha = 0.8, linestyle = ':')
ax7.axhline(y = 35, linewidth = 1.6, color = 'crimson', alpha = 0.8, linestyle = ':')
ax7.scatter([cosmo.lookback_time(zin_zoom_3).value], [28], s = 64, marker = '^', color = 'navy', alpha = 0.6, lw = 0.)
ax7.scatter([cosmo.lookback_time(zin_1024_3).value], [32], s = 64, marker = '^', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax7.scatter([cosmo.lookback_time(zin_2048_3).value], [28], s = 64, marker = '^', color = 'crimson', alpha = 0.6, lw = 0.)

ax7.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 18, labelpad = 8)
ax7.tick_params(labelsize = 14, direction='in')
#ax7.legend(loc='lower right', fontsize = 11)
ax7.set_xticks(lbt.value)
ax7.set_xticklabels(z_lbt)
ax7.set_xticklabels([])
ax7.set_xlim([13.5, 0])
xmin, xmax = ax7.get_xlim()
ymin, ymax = ax7.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax7.text(xmin + width * 0.02, ymax - height * 0.04, r'S3, $\log_{10} (M_{\mathrm{Peak, zoom}}/\mathrm{M_{\odot}}) = 10.17$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax72= ax7.twiny()
ax72.set_xticks(LBT[::-1])
ax72.set_xlim([13.5, 0])
ax72.tick_params(labelsize = 14, direction='in')
ax72.set_xticklabels([])

######

ax8 = fig.add_subplot(8, 3, 11)
ax8.plot(cosmo.lookback_time(z_416_3[0:len(rr_zoom_3)][::-1]), rr_zoom_3[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax8.plot(cosmo.lookback_time(z_1024_3[::-1]), rr_1024_3[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax8.plot(cosmo.lookback_time(z_2048_3[::-1]), rr_2048_3[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax8.axhline(y = 1., lw = 1.6, color = 'black', alpha = 0.8, linestyle = ':')

ax8.set_ylabel(r'$r_{\mathrm{3D}}/R_{\mathrm{vir, host}}$', fontsize = 18, labelpad = 8)
ax8.set_yscale('log')
ax8.set_yticks(np.logspace(0, 2, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax8.yaxis.set_minor_locator(loc2)
ax8.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax8.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax8.scatter([cosmo.lookback_time(zin_zoom_3).value], [5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax8.scatter([cosmo.lookback_time(zin_1024_3).value], [10], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax8.scatter([cosmo.lookback_time(zin_2048_3).value], [5], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)

ax8.set_xticks(lbt.value)
ax8.set_xticklabels(z_lbt)
ax8.set_xticklabels([])
ax8.set_xlim([13.5, 0])
xmin, xmax = ax8.get_xlim()
ymin, ymax = ax8.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax8.text(xmin + width * 0.03, ymax * 0.72, r'S3 (Classical Dwarf)', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)
    
ax82= ax8.twiny()
ax82.set_xticks(LBT[::-1])
ax82.set_xlim([13.5, 0])
ax82.tick_params(labelsize = 14, direction='in')
ax82.set_xticklabels([])

######

ax121 = fig.add_subplot(8, 3, 12)

ax121.plot(cosmo.lookback_time(z_416_3[::-1]), SFH_zoom_3[len(SFH_zoom_3)-len(z_416_3):], linewidth = '2.4', 
          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax121.plot(cosmo.lookback_time(z_1024_3[::-1]), SFH_1024_3[len(SFH_1024_3)-len(z_1024_3):], 
          linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax121.plot(cosmo.lookback_time(z_2048_3[::-1]), SFH_2048_3[len(SFH_2048_3)-len(z_2048_3):], linewidth = '2.2', 
          color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax121.scatter([cosmo.lookback_time(zin_zoom_3).value], [10**-1.8], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax121.scatter([cosmo.lookback_time(zin_1024_3).value], [10**-2.0], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax121.scatter([cosmo.lookback_time(zin_2048_3).value], [10**-1.8], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)

ax121.set_ylabel(r'$\mathrm{SFR_{raw}}\,$[$\mathrm{\mathrm{M_{\odot}}\,yr^{-1}}$]', fontsize = 18, labelpad = 8)
ax121.set_ylim([10**(-4.3), 10**(-1.6)])
ax121.set_yscale('log')
ax121.set_yticks(np.logspace(-4, -2, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax121.yaxis.set_minor_locator(loc2)
ax121.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax121.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax121.set_xticks(lbt.value)
ax121.set_xticklabels(z_lbt)
ax121.set_xticklabels([])
ax121.set_xlim([13.5, 0])
xmin, xmax = ax121.get_xlim()
ymin, ymax = ax121.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax121.text(xmin + width * 0.02, ymax * 0.8, r'S3, $\log_{10} (M_{\mathrm{\ast}}/\mathrm{M_{\odot}}) = 7.10$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax1212= ax121.twiny()
ax1212.set_xticks(LBT[::-1])
ax1212.set_xlim([13.5, 0])
ax1212.tick_params(labelsize = 14, direction='in')
ax1212.set_xticklabels([])

###################################

ax9 = fig.add_subplot(8, 3, 13)
ax9.plot(cosmo.lookback_time(z_416_4[::-1]), vmax_416_4[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
#ax9.plot(cosmo.lookback_time(z_2048_4[::-1]), vmax_2048_4[::-1], linewidth = '1.6', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.3, label = r'c125-2048')
ax9.axhline(y = 10, linewidth = 1.6, color = 'navy', alpha = 0.8, linestyle = ':')
#ax9.axhline(y = 35, linewidth = 0.8, color = 'crimson', alpha = 0.8, linestyle = ':')
ax9.scatter([cosmo.lookback_time(zin_zoom_4).value], [16], s = 64, marker = '^', color = 'navy', alpha = 0.6, lw = 0.)
#ax9.scatter([cosmo.lookback_time(zin_2048_4).value], [16], s = 64, marker = '^', color = 'crimson', alpha = 0.3, lw = 0.)

ax9.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 18, labelpad = 8)
ax9.tick_params(labelsize = 14, direction='in')
#ax9.legend(loc='lower right', fontsize = 11)
ax9.set_xticks(lbt.value)
ax9.set_xticklabels(z_lbt)
ax9.set_xticklabels([])
ax9.set_xlim([13.5, 0])
ax9.set_ylim([0, 32])
xmin, xmax = ax9.get_xlim()
ymin, ymax = ax9.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax9.text(xmin + width * 0.02, ymax - height * 0.04, r'S4, $\log_{10} (M_{\mathrm{Peak, zoom}}/\mathrm{M_{\odot}}) = 9.12$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax92= ax9.twiny()
ax92.set_xticks(LBT[::-1])
ax92.set_xlim([13.5, 0])
ax92.tick_params(labelsize = 14, direction='in')
ax92.set_xticklabels([])

######

ax10 = fig.add_subplot(8, 3, 14)
ax10.plot(cosmo.lookback_time(z_416_4[0:len(rr_zoom_4)][::-1]), rr_zoom_4[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
#ax10.plot(cosmo.lookback_time(z_2048_4[::-1]), np.log10(rr_2048_4[::-1]), linewidth = '1.6', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.3, label = r'c125-2048')
ax10.axhline(y = 1., lw = 1.6, color = 'black', alpha = 0.8, linestyle = ':')

ax10.scatter([cosmo.lookback_time(zin_zoom_4).value], [5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax10.set_ylabel(r'$r_{\mathrm{3D}}/R_{\mathrm{vir, host}}$', fontsize = 18, labelpad = 8)
ax10.set_yscale('log')
ax10.set_yticks(np.logspace(0, 2, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax10.yaxis.set_minor_locator(loc2)
ax10.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax10.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax10.set_xticks(lbt.value)
ax10.set_xticklabels(z_lbt)
ax10.set_xticklabels([])
ax10.set_xlim([13.5, 0])
xmin, xmax = ax10.get_xlim()
ymin, ymax = ax10.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax10.text(xmin + width * 0.02, ymax * 0.72, r'S4 (Ultra faint dwarf)', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax102= ax10.twiny()
ax102.set_xticks(LBT[::-1])
ax102.set_xlim([13.5, 0])
ax102.tick_params(labelsize = 14, direction='in')
ax102.set_xticklabels([])

######
   
ax151 = fig.add_subplot(8, 3, 15)

ax151.plot(cosmo.lookback_time(z_416_4[::-1]), SFH_zoom_4[len(SFH_zoom_4)-len(z_416_4):], linewidth = '2.4', 
          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')

#ax151.plot(cosmo.lookback_time(z_416_4[::-1]), Mstar_zoom_4[len(SFH_zoom_4)-len(z_416_4):]/Mstar_zoom_4[-1], linewidth = '1.8', 
#          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax151.scatter([cosmo.lookback_time(zin_zoom_4).value], [10**-3.6], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)

#ax151.plot(cosmo.lookback_time(z_2048_4[::-1]), np.log10(SFH_2048_4[len(SFH_2048_4)-len(z_2048_4):]), linewidth = '1.6', 
#          color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.3, label = r'c125-2048')
#ax151.scatter([cosmo.lookback_time(zin_zoom_4).value], [-2.8], s = 42, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)

ax151.set_ylabel(r'$\mathrm{SFR_{raw}}\,$[$\mathrm{\mathrm{M_{\odot}}\,yr^{-1}}$]', fontsize = 18, labelpad = 8)
ax151.set_ylim([10**(-7.2), 10**(-2.8)])
ax151.set_yscale('log')
ax151.set_yticks(np.logspace(-7, -3, 5))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax151.yaxis.set_minor_locator(loc2)
ax151.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax151.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax151.set_xticks(lbt.value)
ax151.set_xticklabels(z_lbt)
ax151.set_xticklabels([])
ax151.set_xlim([13.5, 0])
#ax151.set_ylim([-7.2, -2.8])
xmin, xmax = ax151.get_xlim()
ymin, ymax = ax151.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax151.text(xmin + width * 0.02, ymax * 0.72, r'S4, $\log_{10} (M_{\mathrm{\ast, zoom}}/\mathrm{M_{\odot}}) = 5.21$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax1512= ax151.twiny()
ax1512.set_xticks(LBT[::-1])
ax1512.set_xlim([13.5, 0])
ax1512.tick_params(labelsize = 14, direction='in')
ax1512.set_xticklabels([])

###################################

#ax22 = fig.add_subplot(8, 3, 22)
ax22 = fig.add_subplot(8, 3, 16)

ax22.plot(cosmo.lookback_time(z_416_R2[::-1]), vmax_416_R2[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax22.plot(cosmo.lookback_time(z_1024_R2[::-1]), vmax_1024_R2[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax22.plot(cosmo.lookback_time(z_2048_R2[::-1]), vmax_2048_R2[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax22.axhline(y = 10, linewidth = 1.6, color = 'navy', alpha = 0.8, linestyle = ':')
ax22.axhline(y = 60, linewidth = 1.6, color = 'darkturquoise',  alpha = 0.8, linestyle = ':')
ax22.axhline(y = 35, linewidth = 1.6, color = 'crimson', alpha = 0.8, linestyle = ':')

ax22.scatter([cosmo.lookback_time(zin_zoom_R2).value], [48], s = 64, marker = '^', color = 'navy', alpha = 0.6, lw = 0.)
ax22.scatter([cosmo.lookback_time(zin_1024_R2).value], [48], s = 64, marker = '^', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax22.scatter([cosmo.lookback_time(zin_2048_R2).value], [48], s = 64, marker = '^', color = 'crimson', alpha = 0.6, lw = 0.)

ax22.set_xlabel(r'$z$', fontsize = 18, labelpad = 8)
ax22.set_ylabel(r'$v_{\mathrm{max}}\,$[$\mathrm{km\,s^{-1}}$]', fontsize = 18, labelpad = 8)
ax22.tick_params(labelsize = 14, direction='in')
#ax22.legend(loc='lower right', fontsize = 11)
ax22.set_xticks(lbt.value)
ax22.set_xticklabels(z_lbt)
ax22.set_xlim([13.5, 0])
ax22.set_ylim([2, 80])
xmin, xmax = ax22.get_xlim()
ymin, ymax = ax22.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax22.text(xmin + width * 0.02, ymax - height * 0.04, r'S5, $\log_{10} (M_{\mathrm{Peak, zoom}}/\mathrm{M_{\odot}}) = 10.16$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax222= ax22.twiny()
ax222.set_xticks(LBT[::-1])
ax222.set_xlim([13.5, 0])
ax222.tick_params(labelsize = 14, direction='in')
ax222.set_xticklabels([])

######

#ax23 = fig.add_subplot(8, 3, 23)
ax23 = fig.add_subplot(8, 3, 17)
ax23.plot(cosmo.lookback_time(z_416_R2[0:len(rr_zoom_R2)][::-1]), rr_zoom_R2[::-1], linewidth = '2.4', color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax23.plot(cosmo.lookback_time(z_1024_R2[::-1]), rr_1024_R2[::-1], linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax23.plot(cosmo.lookback_time(z_2048_R2[::-1]), rr_2048_R2[::-1], linewidth = '2.2', color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax23.axhline(y = 1., lw = 1.6, color = 'black', alpha = 0.8, linestyle = ':')

ax23.set_xlabel(r'$z$', fontsize = 18, labelpad = 8)
ax23.set_ylabel(r'$r_{\mathrm{3D}}/R_{\mathrm{vir, host}}$', fontsize = 18, labelpad = 8)
ax23.set_yscale('log')
ax23.set_yticks(np.logspace(0, 2, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax23.yaxis.set_minor_locator(loc2)
ax23.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax23.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax23.scatter([cosmo.lookback_time(zin_zoom_R2).value], [5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax23.scatter([cosmo.lookback_time(zin_1024_R2).value], [5], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax23.scatter([cosmo.lookback_time(zin_2048_R2).value], [5], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)

ax23.set_xticks(lbt.value)
ax23.set_xticklabels(z_lbt)

ax23.set_xlim([13.5, 0])
xmin, xmax = ax23.get_xlim()
ymin, ymax = ax23.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax23.text(xmin + width * 0.02, ymax * 0.72, r'S5 (Splashback in zoom-in)', 
          horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax232= ax23.twiny()
ax232.set_xticks(LBT[::-1])
ax232.set_xlim([13.5, 0])
ax232.tick_params(labelsize = 14, direction='in')
ax232.set_xticklabels([])

######

#ax24 = fig.add_subplot(8, 3, 24)
ax24 = fig.add_subplot(8, 3, 18)
ax24.plot(cosmo.lookback_time(z_416_R2[::-1]), SFH_zoom_R2[len(SFH_zoom_R2)-len(z_416_R2):], linewidth = '2.4', 
          color = 'navy', alpha = 0.6, label = r'Halo-416 (zoom-in)')
ax24.plot(cosmo.lookback_time(z_1024_R2[::-1]), SFH_1024_R2[len(SFH_1024_R2)-len(z_1024_R2):], 
          linewidth = '2', color = 'darkturquoise', dashes = (5, 2.4), alpha = 0.8, label = r'c125-1024')
ax24.plot(cosmo.lookback_time(z_2048_R2[::-1]), SFH_2048_R2[len(SFH_2048_R2)-len(z_2048_R2):], linewidth = '2.2', 
          color = 'crimson', dashes = (3.2, 0.5, 1.2, 0.5), alpha = 0.8, label = r'c125-2048')
ax24.scatter([cosmo.lookback_time(zin_zoom_R2).value], [10**-1.5], s = 64, marker = 'v', color = 'navy', alpha = 0.6, lw = 0.)
ax24.scatter([cosmo.lookback_time(zin_1024_R2).value], [10**-1.5], s = 64, marker = 'v', color = 'darkturquoise', alpha = 0.6, lw = 0.)
ax24.scatter([cosmo.lookback_time(zin_2048_R2).value], [10**-1.5], s = 64, marker = 'v', color = 'crimson', alpha = 0.6, lw = 0.)

ax24.set_xlabel(r'$z$', fontsize = 18, labelpad = 8)
ax24.set_ylabel(r'$\mathrm{SFR_{raw}}\,$[$\mathrm{\mathrm{M_{\odot}}\,yr^{-1}}$]', fontsize = 18, labelpad = 8)
ax24.set_ylim([10**(-3.52), 10**-1])
ax24.set_yscale('log')
ax24.set_yticks(np.logspace(-3, -1, 3))
minorLocator = AutoMinorLocator()
loc2 = LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numdecs=10, numticks = 10)
ax24.yaxis.set_minor_locator(loc2)
ax24.tick_params(which='major', labelsize = 14, width = 1., length = 6, direction='in', pad = 4, bottom = True, top = True, left = True, right = True)
ax24.tick_params(which='minor', labelsize = 14, width = 1., length = 3, direction='in', bottom = True, top = True, left = True, right = True)

ax24.set_xticks(lbt.value)
ax24.set_xticklabels(z_lbt)
ax24.set_xlim([13.5, 0])
#ax24.set_ylim([-3.52, -1.08])
xmin, xmax = ax24.get_xlim()
ymin, ymax = ax24.get_ylim()
width = xmax - xmin
height = ymax - ymin
ax24.text(xmin + width * 0.02, ymax  * 0.8, r'S5, $\log_{10} (M_{\mathrm{\ast, zoom}}/\mathrm{M_{\odot}}) = 7.71$', 
         horizontalalignment = 'left', verticalalignment = 'top', fontsize = 16, color = 'black', alpha = 0.8)

ax242= ax24.twiny()
ax242.set_xticks(LBT[::-1])
ax242.set_xlim([13.5, 0])
ax242.tick_params(labelsize = 14, direction='in')
ax242.set_xticklabels([])


fig.savefig('/home/bulk826/Desktop/Stanford/Research1/figures/new_ns1/tracer_plot_new.pdf', dpi = 400, bbox_inches = 'tight')
plt.show()